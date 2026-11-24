"""Online action-level unlikelihood over the pre-action candidate set."""
import re
import time

from .prompting import parse_react
from .training import NTPTrainer


def is_no_effect(step):
    return (step.get("format_error") is None and step.get("executed_action") is not None
            and step.get("feedback", "").strip().lower().rstrip(".! ") in
            ("nothing happen", "nothing happens"))


def negative_reasons(step):
    """An executed action must be admissible, change observation, and not be a no-op.

    Compare the environment's raw observations, not the formatted prompt containing
    the admissible-command list. Unexecuted/format-error outputs cannot be trained
    as action candidates. Missing observations are not evidence of no change.
    """
    if step.get("format_error") is not None or step.get("executed_action") is None:
        return []
    reasons = []
    if step["executed_action"] not in step["admissible_commands"]:
        reasons.append("inadmissible_action")
    if is_no_effect(step):
        reasons.append("nothing_happens")
    if ("observation" in step and "next_observation" in step
            and step["observation"] == step["next_observation"]):
        reasons.append("unchanged_observation")
    return reasons


def is_negative(step):
    return bool(negative_reasons(step))


def candidates_for(step):
    return list(dict.fromkeys([*step["admissible_commands"], step["executed_action"]]))


def build_sample(step, tokenizer, context_limit):
    """Reuse rollout tokens through Action:, then tokenize every continuation alike.

    The separating space belongs to the continuation, since BPE commonly merges
    it with the first action word. No generated reasoning token is supervised.
    """
    parsed = parse_react(step["raw_output"], step.get("generation_truncated", False))
    if parsed.error or parsed.action != step["executed_action"]:
        return None, "action_alignment"
    raw = step["raw_output"]
    match = re.search(r"^Action:", raw, re.M)
    prefix_text = raw[:match.end()]
    output = step["output_token_ids"]
    # Use exact generated IDs, not a potentially different retokenization.
    prefix_end = next((i for i in range(1, len(output) + 1)
                       if tokenizer.decode(output[:i], skip_special_tokens=False) == prefix_text), None)
    if prefix_end is None:
        return None, "action_token_boundary"
    eos = tokenizer.eos_token_id
    if eos is None or not output or output[-1] != eos:
        return None, "missing_action_terminator"
    if tokenizer.decode(output[:-1], skip_special_tokens=False) != raw:
        return None, "output_token_alignment"
    candidates = candidates_for(step)
    if len(candidates) < 2:
        return None, "no_alternative_candidates"
    prefix = [*step["input_token_ids"], *output[:prefix_end]]
    targets = [tokenizer.encode(" " + action, add_special_tokens=False) + [eos]
               for action in candidates]
    if any(len(prefix) + len(target) > context_limit for target in targets):
        raise ValueError("Negative-UL candidate exceeds model context capacity; no scoring truncation")
    return {"prefix_token_ids": prefix, "candidate_token_ids": targets,
            "candidates": candidates, "negative_index": candidates.index(parsed.action),
            "negative_added": parsed.action not in step["admissible_commands"]}, None


def candidate_loss(scores, negative_index):
    import torch
    other = torch.cat((scores[:negative_index], scores[negative_index + 1:]))
    # Equivalent to LSE(all) - LSE(other), stable even for a tiny bad mass.
    return torch.nn.functional.softplus(scores[negative_index] - torch.logsumexp(other, 0))


class NegativeULTrainer(NTPTrainer):
    logits_chunk_tokens = 32

    def score(self, prefix, target):
        import torch
        from torch.utils.checkpoint import checkpoint
        device = next(self.model.parameters()).device
        ids = torch.tensor([prefix + target], device=device)
        hidden = self.model.get_base_model().model(
            input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
            return_dict=True).last_hidden_state
        selected = hidden[:, len(prefix) - 1:-1]
        labels = ids[:, len(prefix):]

        def project(h, y):
            logits = self.model.get_base_model().get_output_embeddings()(h).float()
            return -torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum")

        score = selected.new_zeros((), dtype=torch.float32)
        for start in range(0, len(target), self.logits_chunk_tokens):
            h = selected[:, start:start + self.logits_chunk_tokens]
            y = labels[:, start:start + self.logits_chunk_tokens]
            score = score + (checkpoint(project, h, y, use_reentrant=False)
                             if torch.is_grad_enabled() else project(h, y))
        return score

    def train_sample(self, sample):
        import torch
        started = time.perf_counter()
        prefix, targets = sample["prefix_token_ids"], sample["candidate_token_ids"]
        self.model.train()
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.enable_input_require_grads()
        self.model.config.use_cache = False
        self.optimizer.zero_grad(set_to_none=True)
        try:
            records = []
            for _ in range(self.config.negative_ul_steps):
                self.optimizer.zero_grad(set_to_none=True)
                # Exact two-pass chain rule: first compute dL/dz for ALL candidates,
                # then recompute each z with gradients. Only one graph lives at once.
                # Dropout=0 and no intervening optimizer step make both passes equal.
                with torch.no_grad():
                    scores = torch.stack([self.score(prefix, target) for target in targets])
                scores.requires_grad_(True)
                loss = candidate_loss(scores, sample["negative_index"])
                if not torch.isfinite(scores).all() or not torch.isfinite(loss):
                    raise RuntimeError("Non-finite Negative-UL scores/loss")
                coefficients, = torch.autograd.grad(loss, scores)
                for target, coefficient in zip(targets, coefficients):
                    (self.score(prefix, target) * coefficient).backward()
                norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.config.max_grad_norm,
                                                      error_if_nonfinite=True)
                self.optimizer.step()
                self.steps += 1
                records.append({"optimizer_step": self.steps, "loss": loss.item(),
                                "candidate_log_probs": scores.detach().tolist(),
                                "negative_probability": scores.detach().softmax(0)[sample["negative_index"]].item(),
                                "gradient_norm": norm.item()})
            return {"skipped": False, "optimizer_steps": len(records), "substeps": records, "total_optimizer_steps": self.steps,
                    "loss": loss.item(), "candidate_log_probs": scores.detach().tolist(),
                    "negative_probability": scores.detach().softmax(0)[sample["negative_index"]].item(),
                    "gradient_norm": norm.item(), "seconds": time.perf_counter() - started}
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self.model.gradient_checkpointing_disable()
            self.model.disable_input_require_grads()
            self.model.config.use_cache = True
            self.model.eval()
