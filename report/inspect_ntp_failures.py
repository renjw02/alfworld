"""Audit committed assistant-only NTP traces without loading a model or using GPUs.

Run from any directory: python3 report/inspect_ntp_failures.py
Requires the local runs/ artifacts. EOS id 248046 is from the recorded Qwen3.5 tokenizer.
"""
import pathlib,json,collections
P=pathlib.Path; repo=P(__file__).resolve().parents[1]; root=repo/'runs/continual-matrix-20261101T135608Z'; out=[]
for p in sorted(root.glob('history-ntp-*')):
 if not p.is_dir():continue
 c=json.loads((p/'commit.json').read_text());cfg=json.loads((p/'config.json').read_text());rs=c['completed'];counts=collections.Counter();examples=[];updates=[];first={}; prefix_check={}
 for r in rs:
  a=p/r['attempt_dir'];e=json.loads((a/'episode.json').read_text());steps=e['steps'];u=json.loads((a/'update.json').read_text()) if (a/'update.json').exists() else {}
  if not prefix_check and u.get('training_text'):
   text=u['training_text']; marker='<|im_start|>assistant\n'; start=text.index(marker)
   prefix_check['training_answer_start']=text[start:start+60]
   with (a/'steps.jsonl').open() as log:
    for line in log:
     event=json.loads(line)
     if event['event']=='step':
      prefix_check['generation_prefix']=event['prompt'][event['prompt'].rfind(marker):]
      break
  active=u.get('optimizer_steps',0)>0
  for s in steps:
   typ='truncated' if s.get('generation_truncated') else 'format' if s.get('format_error') else 'invalid' if s.get('invalid_action') else 'valid'
   counts[typ]+=1
   if typ!='valid':first.setdefault(typ,r['position']+1)
   if active:counts['trained_'+typ]+=1
  if active:
   outcome='success' if r['success'] else 'failure';counts[outcome+'_updates']+=1;counts[outcome+'_optimizer_steps']+=u['optimizer_steps'];counts[outcome+'_targets']+=u['target_tokens']
   counts['eos_in_text']+=u['training_token_ids'].count(248046);counts['eos_targets']+=u['training_labels'].count(248046)
   counts['trained_success_with_errors']+=r['success'] and any(s.get('format_error') or s.get('invalid_action') for s in steps)
   labs=u['training_labels'];sz=cfg['training']['chunk_tokens'];chunks=[labs[i:i+sz] for i in range(0,len(labs)-1,sz-1)];counts['chunks']+=len(u['chunks']);counts['chunks_start_inside_assistant']+=sum(ch[0]!=-100 for ch in chunks if any(x!=-100 for x in ch[1:]))
   updates.append({'episode':r['position']+1,'success':r['success'],'loss':u['loss'],'optimizer_steps':u['optimizer_steps'],'targets':u['target_tokens']})
  actions=collections.Counter(s.get('action') for s in steps if s.get('action'))
  if actions:
   act,n=actions.most_common(1)[0]
   if n>=20:
    s=next(s for s in steps if s.get('action')==act);examples.append({'episode':r['position']+1,'success':r['success'],'action':act,'repetitions':n,'invalid':s.get('invalid_action'),'feedback':s.get('feedback','')[:180],'raw':s['raw_output'][:300],'source':str((a/'episode.json').relative_to(repo))})
  if r['environment_calls']==0:first.setdefault('zero_calls',r['position']+1);counts['zero_call_episodes']+=1
  if active and r['success'] and any(s.get('invalid_action') for s in steps) and not any(x.get('success_error') for x in examples):
   s=next(s for s in steps if s.get('invalid_action'));examples.append({'success_error':True,'episode':r['position']+1,'action':s['action'],'feedback':s.get('feedback','')[:180],'source':str((a/'episode.json').relative_to(repo))})
 assert counts['valid'] + counts['invalid'] + counts['format'] + counts['truncated'] == sum(r['attempts'] for r in rs)
 assert counts['invalid'] == sum(r['invalid_actions'] for r in rs)
 assert counts['format'] + counts['truncated'] == sum(r['format_errors'] for r in rs)
 result={'prefix_check':prefix_check,'name':p.name,'episodes':len(rs),'successes':sum(r['success'] for r in rs),'counts':dict(counts),'first':first,'updates':updates,'examples':examples[:8],'blocks10':[{'start':i+1,'n':len(rs[i:i+10]),'successes':sum(r['success'] for r in rs[i:i+10]),'invalid':sum(r['invalid_actions'] for r in rs[i:i+10]),'format':sum(r['format_errors'] for r in rs[i:i+10])} for i in range(0,len(rs),10)]};out.append(result)
 print(p.name,len(rs),result['successes'],dict(counts),first,flush=True)
(repo/'report/ntp_failure_snapshot.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
