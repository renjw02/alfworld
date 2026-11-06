"""Whole-episode SFT with interleaved response masks and repeated full updates."""
import time

from .prompting import system_message
from .sft import interaction_error
from .training import NTPTrainer


def build_trajectory(config, episode, render, tokenizer, context_limit):
    messages = [system_message(config.action_mode), *episode.messages]
    responses = [m["content"] for m in messages if m["role"] == "assistant"]
    if responses != [s["raw_output"] for s in episode.steps]:
        raise ValueError("SFT transcript responses must align with recorded steps")
    text, spans = render(messages, False)
    if len(spans) != len(episode.steps):
        raise ValueError("SFT response spans must align with recorded steps")
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    limit = min(config.sft.max_seq_tokens, context_limit)
    if len(ids) > limit:
        raise ValueError(f"SFT trajectory has {len(ids)} tokens, exceeds {limit}; no truncation or chunking")
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("SFT requires an EOS token")
    labels = [-100] * len(ids)
    retained, rejected, targets = [], [], []
    special = set(tokenizer.all_special_ids)
    for step, (start, end) in zip(episode.steps, spans):
        reason = interaction_error(step)
        if reason:
            rejected.append({"step": step["step"], "reason": reason})
            continue
        body = [i for i, (a, b) in enumerate(offsets) if b > a and a >= start and b <= end]
        if any(ids[i] in special for i in body):
            rejected.append({"step": step["step"], "reason": "response_control_tokens"})
            continue
        # A boundary-straddling token cannot be labeled without also supervising
        # template text. Fail explicitly rather than silently dropping body tokens.
        if (not body or offsets[body[0]][0] != start or offsets[body[-1]][1] != end
                or any(a < start < b or a < end < b for a, b in offsets)):
            raise ValueError("SFT response does not align with tokenizer boundaries")
        eos_index = body[-1] + 1
        if eos_index >= len(ids) or ids[eos_index] != eos or offsets[eos_index][0] != end:
            raise ValueError("SFT chat template must end each assistant body with EOS")
        for index in [*body, eos_index]:
            labels[index] = ids[index]
        retained.append(step["step"])
        targets.append({"step": step["step"], "start": body[0], "end": eos_index + 1,
                        "target_tokens": len(body) + 1})
    if labels and labels[0] != -100:
        raise ValueError("The first token cannot be a causal SFT target")
    return {"text": text, "input_ids": ids, "labels": labels, "tokens": len(ids),
            "target_tokens": sum(t != -100 for t in labels),
            "retained_steps": retained, "rejected": rejected, "target_spans": targets}


class TrajectorySFTTrainer(NTPTrainer):
    def __init__(self, model, config, sft_config):
        super().__init__(model, config)
        self.sft_config = sft_config

    def _loss_sum(self, hidden, targets):
        import torch.nn.functional as F
        logits = self.model.get_base_model().get_output_embeddings()(hidden)
        return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                               targets.reshape(-1), reduction="sum")

    def train_trajectory(self, sample):
        import torch
        from torch.utils.checkpoint import checkpoint
        start = time.perf_counter()
        ids, labels = sample["input_ids"], sample["labels"]
        if (len(ids) != len(labels) or not ids or labels[0] != -100
                or any(label != -100 and label != token for token, label in zip(ids, labels))):
            raise ValueError("Invalid whole-trajectory SFT labels")
        architecture = getattr(self.model.config, "text_config", self.model.config)
        if len(ids) > min(self.sft_config.max_seq_tokens, architecture.max_position_embeddings):
            raise ValueError("SFT trajectory exceeds context budget; no truncation or chunking")
        positions = [i for i, label in enumerate(labels) if label != -100]
        records = []
        if positions:
            device = next(self.model.parameters()).device
            inputs = torch.tensor([ids], device=device)
            prediction_positions = torch.tensor([i - 1 for i in positions], device=device)
            targets = torch.tensor([labels[i] for i in positions], device=device)
            self.model.train()
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.enable_input_require_grads()
            self.model.config.use_cache = False
            try:
                for _ in range(self.sft_config.optimizer_steps):
                    self.optimizer.zero_grad(set_to_none=True)
                    # One full causal decoder pass per update. Project only valid
                    # prediction positions to vocabulary logits, in checkpointed
                    # blocks; this changes neither context nor optimizer cadence.
                    hidden = self.model.get_base_model().model(
                        input_ids=inputs, attention_mask=torch.ones_like(inputs),
                        use_cache=False, return_dict=True).last_hidden_state
                    selected = hidden.index_select(1, prediction_positions)
                    loss = selected.new_zeros((), dtype=torch.float32)
                    for offset in range(0, len(positions), self.sft_config.logits_chunk_tokens):
                        end = offset + self.sft_config.logits_chunk_tokens
                        loss = loss + checkpoint(self._loss_sum, selected[:, offset:end], targets[offset:end],
                                                 use_reentrant=False) / len(positions)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite trajectory SFT loss; episode update not committed")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.config.max_grad_norm,
                                                          error_if_nonfinite=True)
                    self.optimizer.step()
                    self.steps += 1
                    records.append({"optimizer_step": self.steps, "loss": loss.item(),
                                    "gradient_norm": norm.item(), "target_tokens": len(positions)})
                    del loss, hidden, selected
            finally:
                self.optimizer.zero_grad(set_to_none=True)
                self.model.gradient_checkpointing_disable()
                self.model.disable_input_require_grads()
                self.model.config.use_cache = True
                self.model.eval()
        return {"skipped": not bool(records), "skip_reason": None if records else "no_valid_targets",
                "optimizer_steps": len(records), "total_optimizer_steps": self.steps,
                "target_tokens": len(positions), "tokens": len(ids), "records": records,
                "loss": sum(r["loss"] for r in records) / len(records) if records else None,
                "seconds": time.perf_counter() - start}
