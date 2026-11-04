"""Response-only SFT protocol, exact prefixes, filtering, and real tiny-model updates."""
import copy
import unittest
from unittest.mock import patch

from alfworld.continual.config import Config
from alfworld.continual.methods import Method
from alfworld.continual.model import HFBackend, MockBackend
from alfworld.continual.prompting import initial_message
from alfworld.continual.sft import build_samples, rollout_context
from alfworld.continual.types import Episode


class Tokenizer:
    eos_token_id = 2
    pad_token_id = 0
    all_special_ids = [0, 2]

    def encode(self, text, **kwargs):
        return [ord(c) + 3 for c in text]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        return ''.join(m['role']+':'+m['content']+'\n' for m in messages) + (
            'assistant:\n<think>\n\n</think>\n\n' if add_generation_prompt else '')


def episode(count=8):
    e = Episode('fixture', 'pick', 0, outcome={'success':True})
    obs = 'Room.\nYour task is to: find a mug.'
    e.messages = [initial_message(e.episode_id, obs, ['look'], 'admissible')]
    for i in range(count):
        after = f'Observation after step {i+1}'
        e.steps.append(dict(step=i+1, observation=obs, admissible_commands=['look'],
                            raw_output=f'Reasoning: inspect room {i+1}\nAction: look',
                            action='look', executed_action='look', format_error=None,
                            generation_truncated=False, invalid_action=False,
                            next_observation=after, next_admissible_commands=['look'],
                            feedback=after, score=0, success=i==count-1))
        obs = after
    return e


def renderer():
    backend = HFBackend.__new__(HFBackend)
    backend.tokenizer = Tokenizer()
    return backend


class SFTProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(method='success-only-sft')
        self.config.sft.max_seq_tokens = 8192
        self.renderer = renderer()

    def samples(self, e):
        return build_samples(self.config, e, self.renderer.serialize, self.renderer.tokenizer)

    def test_complete_sample_prefix_history_mask_and_no_future_leakage(self):
        e = episode()
        samples, rejected = self.samples(e)
        self.assertFalse(rejected)
        self.assertEqual(len(samples), 8)
        for index, sample in enumerate(samples):
            prefix = copy.deepcopy(e)
            prefix.steps = prefix.steps[:index]
            prompt, ids = self.renderer.serialize(rollout_context(self.config, prefix), True)
            self.assertEqual(sample['prompt'], prompt)
            self.assertEqual(sample['input_ids'][:sample['prompt_tokens']], ids)
            self.assertEqual(sample['labels'][:len(ids)], [-100]*len(ids))
            target = self.renderer.tokenizer.encode(e.steps[index]['raw_output']) + [2]
            self.assertEqual(sample['labels'][len(ids):], target)
            self.assertNotIn(e.steps[index]['next_observation'], sample['prompt'])
            self.assertIn('Task: find a mug.', sample['prompt'])
            self.assertIn(e.steps[index]['observation'], sample['prompt'])
            self.assertEqual(sample['history_steps'], list(range(max(1,index-4),index+1)))
        self.assertNotIn('inspect room 1', samples[-1]['prompt'])
        self.assertIn('inspect room 3', samples[-1]['prompt'])

    def test_failed_episode_errors_noops_and_history_only(self):
        e = episode()
        e.outcome['success'] = False
        self.assertEqual(self.samples(e), ([], [{'reason':'unsuccessful_episode'}]))
        e.outcome['success'] = True
        e.steps[0].update(generation_truncated=True,format_error='truncated',executed_action=None,
                          next_observation=e.steps[0]['observation'],feedback='Format error: truncated')
        e.steps[1]['observation'] = e.steps[0]['next_observation']
        e.steps[2].update(action='fake',executed_action='fake',raw_output='Reasoning: x\nAction: fake')
        e.steps[3]['feedback'] = 'Nothing happens.'
        e.steps[4]['next_observation'] = e.steps[4]['observation']
        e.steps[5]['observation'] = e.steps[4]['next_observation']
        samples, rejected = self.samples(e)
        self.assertEqual([s['step'] for s in samples], [2,6,7,8])
        self.assertEqual({r['reason'] for r in rejected}, {'format_or_truncated','unexecuted_or_inadmissible','no_effect','unchanged_observation'})
        self.assertIn('Action: fake', samples[1]['prompt'])  # feedback context, never target
        for index in (1,5):
            prefix = copy.deepcopy(e);prefix.steps = prefix.steps[:index]
            expected = self.renderer.serialize(rollout_context(self.config,prefix),True)[0]
            self.assertEqual(next(s for s in samples if s['step']==index+1)['prompt'],expected)

    def test_budget_skips_whole_example_no_chunking_and_gate(self):
        e = episode(1)
        sample = self.samples(e)[0][0]
        self.config.sft.max_seq_tokens = len(sample['input_ids'])-1
        samples, rejected = self.samples(e)
        self.assertFalse(samples)
        self.assertEqual(rejected[0]['reason'],'overlength')
        self.config.sft.max_seq_tokens += 1
        self.assertEqual(self.samples(e)[0][0]['labels'][-1],2)
        backend = MockBackend(self.config);method = Method(self.config,backend)
        e.outcome['success'] = False
        self.assertTrue(method.observe_episode(e)['skipped'])
        self.assertEqual(backend.updates,0)
        self.assertEqual(method.version,0)

    def test_config_invalid_bounds(self):
        self.assertTrue(self.config.validate().trains_parameters)
        self.assertFalse(self.config.is_ntp)
        for value in (True,1,0,1.5):
            self.config.sft.max_seq_tokens=value
            with self.assertRaises(ValueError):self.config.validate()


class SFTModelTests(unittest.TestCase):
    def test_real_loss_lora_resume_and_task_reset(self):
        import torch
        from transformers import Qwen3_5TextConfig,Qwen3_5ForCausalLM
        torch.set_num_threads(2);torch.manual_seed(42)
        config=Config(method='success-only-sft');config.model.device='cpu'
        config.training.chunk_tokens=2  # Must have no effect on SFT sample construction/training.
        config.training.learning_rate=1e-3;config.sft.max_seq_tokens=8192
        config.memory_scope='task_type';config.stream.order='task_blocked'
        arch=Qwen3_5TextConfig(vocab_size=256,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
            num_attention_heads=4,num_key_value_heads=2,head_dim=8,linear_key_head_dim=8,linear_value_head_dim=8,
            linear_num_key_heads=2,linear_num_value_heads=4,layer_types=['linear_attention','full_attention'],
            max_position_embeddings=8192,eos_token_id=2,pad_token_id=0,
            rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':1.,'mrope_section':[1,1,2]})
        backend=HFBackend(config,model=Qwen3_5ForCausalLM(arch),tokenizer=Tokenizer())
        method=Method(config,backend);method.begin_task('pick')
        initial=copy.deepcopy(backend.state_dict());e=episode(2)
        samples,_=build_samples(config,e,backend.serialize,backend.tokenizer)
        # Independently verify shifted loss applies to response + EOS, not prompt.
        s=samples[0]
        with torch.no_grad():
            logits=backend.model(input_ids=torch.tensor([s['input_ids']]),use_cache=False).logits
            manual=torch.nn.functional.cross_entropy(logits[0,s['prompt_tokens']-1:-1].float(),torch.tensor(s['input_ids'][s['prompt_tokens']:]))
        frozen={n:p.detach().clone() for n,p in backend.model.named_parameters() if not p.requires_grad}
        with patch('alfworld.continual.training.token_chunks',side_effect=AssertionError('SFT may not chunk')):
            result=method.observe_episode(e)
        self.assertEqual(result['optimizer_steps'],2)
        self.assertAlmostEqual(result['records'][0]['loss'],manual.item(),places=5)
        self.assertEqual(result['version_after'],1)
        self.assertTrue(any(not torch.equal(initial['adapter'][k],v) for k,v in backend.state_dict()['adapter'].items()))
        for n,p in backend.model.named_parameters():
            if n in frozen:torch.testing.assert_close(p,frozen[n],rtol=0,atol=0)
        saved=copy.deepcopy(backend.state_dict());a=backend.sft_episode(e)
        expected=copy.deepcopy(backend.state_dict()['adapter'])
        backend.load_state_dict(saved);b=backend.sft_episode(e)
        self.assertEqual(a['total_optimizer_steps'],b['total_optimizer_steps'])
        for k,v in expected.items():torch.testing.assert_close(backend.state_dict()['adapter'][k],v,rtol=0,atol=0)
        self.assertTrue(method.begin_task('clean'))
        self.assertEqual(backend.trainer.steps,0)
        for k,v in initial['adapter'].items():torch.testing.assert_close(backend.state_dict()['adapter'][k],v,rtol=0,atol=0)
        e.outcome['success']=False
        self.assertTrue(backend.sft_episode(e)['skipped'])
        self.assertEqual(backend.trainer.steps,0)


if __name__=='__main__':unittest.main()
