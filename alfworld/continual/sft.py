"""Per-interaction SFT: exact rollout prefix, current response + EOS targets only.

ALFWorld exposes episode success, not a per-step progress reward. Here a successful
interaction means an executed admissible action without a format error, explicit
no-op feedback, or unchanged observation. This is an auditable proxy, not proof
that each action was necessary for solving the task.
"""
import re
import time

from .prompting import observation_message, parse_react, system_message
from .training import NTPTrainer


def task_description(episode):
    initial = episode.steps[0]["observation"] if episode.steps else episode.messages[0]["content"]
    match = re.search(r"Your task is to\s*:?\s*([^\n]+)", initial, re.IGNORECASE)
    if not match:
        raise ValueError("SFT requires a task description in the initial observation")
    return match[1].strip()


def interaction_error(step):
    parsed = parse_react(step["raw_output"], step.get("generation_truncated", False))
    if step.get("format_error") or parsed.error:
        return "format_or_truncated"
    if step.get("executed_action") != parsed.action or parsed.action not in step["admissible_commands"]:
        return "unexecuted_or_inadmissible"
    if not step.get("success"):
        if step.get("feedback", "").strip().lower() == "nothing happens.":
            return "no_effect"
        if step.get("next_observation") == step["observation"]:
            return "unchanged_observation"
    return None


def context_messages(config, task, history, current_message):
    """Keep the last five chronological attempts, including feedback on errors.

Earlier responses are context only. The final current observation/admissible list
is authoritative. No future outcome or next observation of the target is exposed.
"""
    n = config.sft.history_interactions
    recent = history[-n:] if n else []
    messages = [system_message(config.action_mode), {"role": "user", "content": f"Task: {task}"}]
    for step in recent:
        messages.extend([
            {"role": "user", "content": f"Observation:\n{step['observation']}"},
            {"role": "assistant", "content": step["raw_output"]},
            {"role": "user", "content": f"Feedback:\n{step['feedback']}"},
        ])
    messages.append(dict(current_message))
    return messages


def rollout_context(config, episode):
    if episode.steps:
        previous = episode.steps[-1]
        current = current_observation(config, previous["next_observation"],
                                      previous["next_admissible_commands"], previous)
    else:
        content = episode.messages[0]["content"]
        if content.startswith("BEGIN EPISODE "):
            content = content.split("\n", 1)[1]
        current = {"role": "user", "content": content}
    return context_messages(config, task_description(episode), episode.steps, current)


def current_observation(config, observation, commands, previous=None):
    current = observation_message(observation, commands, config.action_mode,
                                  previous.get("score", 0) if previous else 0)
    if previous and previous.get("format_error"):
        current["content"] = previous["feedback"] + "\n" + current["content"]
    return current


def build_samples(config, episode, serialize, tokenizer):
    """Preserve whole samples; never slice targets or drop any of the five turns.

Oversize samples are skipped with measured lengths. Raise the explicit limit
after examining the pilot instead of silently training on incomplete context.
"""
    if not episode.outcome.get("success"):
        return [], [{"reason": "unsuccessful_episode"}]
    task = task_description(episode)
    samples, rejected = [], []
    for index, step in enumerate(episode.steps):
        error = interaction_error(step)
        if error:
            rejected.append({"step": step["step"], "reason": error})
            continue
        current = current_observation(config, step["observation"], step["admissible_commands"],
                                      episode.steps[index - 1] if index else None)
        messages = context_messages(config, task, episode.steps[:index], current)
        prompt, prompt_ids = serialize(messages, True)
        response_ids = tokenizer.encode(step["raw_output"], add_special_tokens=False)
        eos = tokenizer.eos_token_id
        if eos is None:
            raise ValueError("SFT requires an EOS token")
        # Reject injected chat-control tokens rather than supervising fake turns.
        if set(response_ids).intersection(getattr(tokenizer, "all_special_ids", [eos])):
            rejected.append({"step": step["step"], "reason": "response_control_tokens"})
            continue
        ids = prompt_ids + response_ids + [eos]
        if len(ids) > config.sft.max_seq_tokens:
            rejected.append({"step": step["step"], "reason": "overlength", "tokens": len(ids),
                             "prompt_tokens": len(prompt_ids), "response_tokens": len(response_ids) + 1})
            continue
        samples.append({"step": step["step"], "messages": messages, "prompt": prompt,
                        "response": step["raw_output"], "input_ids": ids,
                        "labels": [-100] * len(prompt_ids) + response_ids + [eos],
                        "prompt_tokens": len(prompt_ids), "target_tokens": len(response_ids) + 1,
                        "history_steps": [s["step"] for s in episode.steps[max(0, index - config.sft.history_interactions):index]]})
    return samples, rejected


class SFTTrainer(NTPTrainer):
    """Mini-batches accumulated into sample-weighted optimizer updates."""

    def __init__(self, model, config, sft_config=None, pad_token_id=0):
        super().__init__(model, config)
        from .config import SFTConfig
        self.sft_config = sft_config or SFTConfig()
        self.pad_token_id = pad_token_id

    def collate(self, samples, device):
        import torch
        length = max(len(s["input_ids"]) - 1 for s in samples)
        targets_length = max(len(s["input_ids"]) - s["prompt_tokens"] for s in samples)
        inputs = torch.full((len(samples), length), self.pad_token_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(inputs)
        targets = torch.full((len(samples), targets_length), -100, dtype=torch.long, device=device)
        for row, sample in enumerate(samples):
            ids = sample["input_ids"][:-1]  # causal shift, EOS remains a target
            response = sample["labels"][sample["prompt_tokens"]:]
            inputs[row, -len(ids):] = torch.tensor(ids, device=device)
            mask[row, -len(ids):] = 1
            targets[row, -len(response):] = torch.tensor(response, device=device)
        positions = (mask.cumsum(-1) - 1).clamp_min(0)
        return inputs, mask, positions, targets

    def train_samples(self, samples):
        import torch
        started = time.perf_counter()
        records = []
        # Validate all examples before making any update.
        for sample in samples:
            ids, labels, prefix = sample["input_ids"], sample["labels"], sample["prompt_tokens"]
            if not 0 < prefix < len(ids) or len(ids) != len(labels):
                raise ValueError("Invalid SFT sequence boundary")
            if labels[:prefix] != [-100] * prefix or labels[prefix:] != ids[prefix:]:
                raise ValueError("SFT must supervise only the complete current response")
        if samples:
            self.model.train()
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.enable_input_require_grads()
            self.model.config.use_cache = False
            device = next(self.model.parameters()).device
            try:
                batch_size = self.sft_config.batch_size
                effective_batch = batch_size * self.sft_config.gradient_accumulation_steps
                for start in range(0, len(samples), effective_batch):
                    window = samples[start:start + effective_batch]
                    self.optimizer.zero_grad(set_to_none=True)
                    window_loss = 0.0
                    micro_batches = 0
                    for offset in range(0, len(window), batch_size):
                        batch = window[offset:offset + batch_size]
                        inputs, mask, positions, targets = self.collate(batch, device)
                        # Left padding aligns response ends. Only response positions
                        # get vocabulary logits; shorter responses mask extra positions.
                        result = self.model(input_ids=inputs, attention_mask=mask,
                                            position_ids=positions, logits_to_keep=targets.shape[1],
                                            use_cache=False)
                        token_losses = torch.nn.functional.cross_entropy(
                            result.logits.float().transpose(1, 2), targets,
                            ignore_index=-100, reduction="none")
                        response_losses = token_losses.sum(-1) / (targets != -100).sum(-1)
                        # Actual sample count handles both a short micro-batch and
                        # the final partial accumulation window without underweighting.
                        loss = response_losses.sum() / len(window)
                        if not torch.isfinite(loss):
                            raise RuntimeError("Non-finite SFT loss; episode update not committed")
                        loss.backward()
                        window_loss += loss.item()
                        micro_batches += 1
                        del loss, response_losses, token_losses, result, inputs, mask, positions, targets
                    norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.config.max_grad_norm,
                                                          error_if_nonfinite=True)
                    self.optimizer.step()
                    self.steps += 1
                    records.append({"sample_steps": [s["step"] for s in window], "loss": window_loss,
                                    "samples": len(window), "micro_batches": micro_batches,
                                    "gradient_norm": norm.item(), "tokens": sum(len(s["input_ids"]) for s in window),
                                    "target_tokens": sum(s["target_tokens"] for s in window),
                                    "optimizer_step": self.steps})
            finally:
                self.optimizer.zero_grad(set_to_none=True)
                self.model.gradient_checkpointing_disable()
                self.model.disable_input_require_grads()
                self.model.config.use_cache = True
                self.model.eval()
        return {"skipped": not bool(records), "optimizer_steps": len(records),
                "total_optimizer_steps": self.steps, "samples_trained": sum(r["samples"] for r in records),
                "batch_size": self.sft_config.batch_size,
                "gradient_accumulation_steps": self.sft_config.gradient_accumulation_steps,
                "micro_batches": sum(r["micro_batches"] for r in records),
                "target_tokens": sum(r["target_tokens"] for r in records),
                "loss": sum(r["loss"] * r["samples"] for r in records) / len(samples) if records else None,
                "loss_reduction": "mean_per_response", "records": records,
                "seconds": time.perf_counter() - started}
