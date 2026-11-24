"""Episode-end positive SFT and state-segment-deduplicated action unlikelihood."""
import time

from .negative_ul import NegativeULTrainer, build_sample, candidate_loss, is_no_effect
from .prompting import parse_react
from .trajectory_sft import TrajectorySFTTrainer, build_trajectory


def positive_error(step):
    parsed = parse_react(step['raw_output'], step.get('generation_truncated', False))
    if step.get('format_error') or parsed.error:
        return 'format_or_truncated'
    if step.get('executed_action') is None or step['executed_action'] != parsed.action:
        return 'unexecuted_or_misaligned'
    return 'no_effect' if is_no_effect(step) else None


def unique_negatives(steps):
    # No-effect actions and format errors do not advance the state segment.
    # This is deliberately not a claim of true environment-state equivalence.
    seen, retained = set(), []
    segment = raw = duplicates = 0
    for step in steps:
        if is_no_effect(step):
            raw += 1
            key = (segment, step['executed_action'])
            if key in seen:
                duplicates += 1
            else:
                seen.add(key)
                retained.append(step)
        elif step.get('executed_action') is not None and not step.get('format_error'):
            segment += 1
    return retained, {'negative_feedback_count': raw, 'unique_negative_count': len(retained),
                      'duplicate_negative_count': duplicates}


def build_joint_sample(config, episode, render, tokenizer, context_limit):
    trajectory = build_trajectory(config, episode, render, tokenizer, context_limit,
                                  validity=positive_error)
    steps, stats = unique_negatives(episode.steps)
    negatives, rejected = [], []
    for step in steps:
        sample, reason = build_sample(step, tokenizer, context_limit)
        if sample is None:
            rejected.append({'step': step['step'], 'reason': reason})
        else:
            negatives.append({'step': step['step'], **sample})
    return {'trajectory': trajectory, 'negative_samples': negatives, 'negative_rejected': rejected,
            'trainable_negative_count': len(negatives), **stats}


class JointTrainer(TrajectorySFTTrainer):
    score = NegativeULTrainer.score

    def train_joint(self, sample):
        import torch
        from torch.utils.checkpoint import checkpoint
        started = time.perf_counter()
        trajectory, negatives = sample['trajectory'], sample['negative_samples']
        ids, labels = trajectory['input_ids'], trajectory['labels']
        if (not ids or len(ids) != len(labels) or labels[0] != -100
                or any(y != -100 and y != x for x, y in zip(ids, labels))):
            raise ValueError('Invalid joint SFT labels')
        architecture = getattr(self.model.config, 'text_config', self.model.config)
        if len(ids) > min(self.sft_config.max_seq_tokens, architecture.max_position_embeddings):
            raise ValueError('Joint trajectory exceeds context budget')
        positions = [i for i, y in enumerate(labels) if y != -100]
        weight = self.config.lambda_neg
        records = []
        if not positions and not (negatives and weight > 0):
            return {'skipped': True, 'skip_reason': 'no_valid_targets', 'optimizer_steps': 0,
                    'total_optimizer_steps': self.steps, 'loss': None, 'pos_loss': None,
                    'neg_loss': None, 'records': [], 'seconds': time.perf_counter() - started}
        device = next(self.model.parameters()).device
        self.logits_chunk_tokens = self.sft_config.logits_chunk_tokens
        self.model.train()
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.model.enable_input_require_grads()
        self.model.config.use_cache = False
        try:
            for _ in range(self.sft_config.optimizer_steps):
                self.optimizer.zero_grad(set_to_none=True)
                pos_value = neg_value = 0.0
                if positions:
                    inputs = torch.tensor([ids], device=device)
                    indices = torch.tensor([i - 1 for i in positions], device=device)
                    targets = torch.tensor([labels[i] for i in positions], device=device)
                    hidden = self.model.get_base_model().model(
                        input_ids=inputs, attention_mask=torch.ones_like(inputs),
                        use_cache=False, return_dict=True).last_hidden_state
                    selected = hidden.index_select(1, indices)
                    loss = selected.new_zeros((), dtype=torch.float32)
                    for start in range(0, len(positions), self.logits_chunk_tokens):
                        end = start + self.logits_chunk_tokens
                        loss = loss + checkpoint(self._loss_sum, selected[:, start:end], targets[start:end],
                                                 use_reentrant=False) / len(positions)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Non-finite joint positive loss')
                    pos_value = loss.item()
                    loss.backward()
                    del loss, hidden, selected
                if weight > 0:
                    for negative in negatives:
                        prefix, targets = negative['prefix_token_ids'], negative['candidate_token_ids']
                        with torch.no_grad():
                            scores = torch.stack([self.score(prefix, target) for target in targets])
                        scores.requires_grad_(True)
                        loss = candidate_loss(scores, negative['negative_index'])
                        if not torch.isfinite(scores).all() or not torch.isfinite(loss):
                            raise RuntimeError('Non-finite joint negative loss')
                        neg_value += loss.item() / len(negatives)
                        coefficients, = torch.autograd.grad(loss, scores)
                        for target, coefficient in zip(targets, coefficients):
                            (self.score(prefix, target) * coefficient * weight / len(negatives)).backward()
                norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.config.max_grad_norm,
                                                      error_if_nonfinite=True)
                self.optimizer.step()
                self.steps += 1
                records.append({'optimizer_step': self.steps, 'pos_loss': pos_value, 'neg_loss': neg_value,
                                'loss': pos_value + weight * neg_value, 'gradient_norm': norm.item()})
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self.model.gradient_checkpointing_disable()
            self.model.disable_input_require_grads()
            self.model.config.use_cache = True
            self.model.eval()
        return {'skipped': False, 'skip_reason': None, 'optimizer_steps': len(records),
                'total_optimizer_steps': self.steps, 'target_tokens': len(positions), 'records': records,
                **{k: sum(r[k] for r in records) / len(records) for k in ('loss', 'pos_loss', 'neg_loss')},
                'seconds': time.perf_counter() - started}
