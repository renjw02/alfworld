"""ASCENT: verified online self-distillation (arXiv:2610.05303, Eq. 8).

ASCENT keeps its rollout prefix; ASCENT+ICL supplies a current-episode-only prefix.
Generated response tokens are immutable. Only the teacher receives hindsight.
This backend targets Qwen3_5ForCausalLM, as does the continual HF backend.
"""
import copy
import math
import random
import re
import time
from .training import NTPTrainer


def current_episode_training_steps(episode):
    """Swap only the conditioning prefix; preserve actual rollout response tokens."""
    steps = []
    for step in episode.steps:
        prefix = step.get("training_input_token_ids")
        if not prefix:
            raise ValueError("ASCENT+ICL requires current-episode training prefixes; restart legacy runs")
        steps.append({**step, "input_token_ids": prefix})
    return steps


def episode_goal(episode):
    initial = episode.steps[0]["observation"] if episode.steps else ""
    match = re.search(r"Your task is to\s*:?\s*([^\n]+)", initial, re.IGNORECASE)
    if not match:
        raise ValueError("Could not extract the goal from the initial ALFWorld observation")
    return match[1].strip()


def success_record(episode):
    if not episode.outcome.get("success"):
        raise ValueError("Cannot build a success record from an unsuccessful episode")
    goal = episode_goal(episode)
    kept, removed, responses = [], [], []
    for step in episode.steps:
        valid = (step["format_error"] is None and step["executed_action"] is not None
                 and step["action"] in step["admissible_commands"])
        (kept if valid else removed).append(step["step"])
        if valid:
            responses.append(f"{len(responses) + 1}. {step['raw_output']}")
    text = (f"GOAL ACHIEVED: {goal}\n"
            "The task above was SOLVED by the following successful action sequence:\n"
            + "\n".join(responses)
            + "\nCondition on this successful outcome when choosing the next action.\n\n")
    return text, kept, removed


def forward_kl_sum(student_logits, teacher_logits):
    """Full vocabulary KL(q || p), summed over response positions, in FP32."""
    import torch.nn.functional as F
    log_q = F.log_softmax(teacher_logits.detach().float(), dim=-1)
    log_p = F.log_softmax(student_logits.float(), dim=-1)
    return (log_q.exp() * (log_q - log_p)).sum()


def reverse_kl_sum(student_logits, teacher_logits):
    """Full vocabulary KL(p || q), with gradients through both student factors."""
    import torch.nn.functional as F
    log_q = F.log_softmax(teacher_logits.detach().float(), dim=-1)
    log_p = F.log_softmax(student_logits.float(), dim=-1)
    return (log_p.exp() * (log_p - log_q)).sum()


class AscentReplayBuffer:
    """FIFO of successful training episodes; retain CPU token lists only."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.entries = []

    def prepare(self, steps, privileged_ids, episode_id):
        # An empty episode must neither evict data nor trigger historical updates.
        if not any(s["output_token_ids"] for s in steps):
            return []
        entry = {"episode_id": episode_id, "teacher_token_ids": list(privileged_ids),
                 "steps": [{key: copy.deepcopy(s[key])
                            for key in ("step", "input_token_ids", "output_token_ids")}
                           for s in steps]}
        return (self.entries + [entry])[-self.capacity:]

    def sample(self, entries, batch_size):
        # The runner seeds and checkpoints Python's RNG, including on resume.
        return random.sample(entries, batch_size) if len(entries) > batch_size else list(entries)

    def summary(self, batch, batch_size):
        return {"replay_buffer_size": self.capacity,
                "replay_episodes": len(self.entries),
                "replay_episode_ids": [e["episode_id"] for e in self.entries],
                "replay_turn_counts": [sum(bool(s["output_token_ids"]) for s in e["steps"])
                                       for e in self.entries],
                "replay_batch_size": batch_size,
                "replay_batch_episodes": len(batch),
                "replay_sampled_episode_ids": [e["episode_id"] for e in batch]}


class AscentTrainer(NTPTrainer):
    """Share the frozen base; accumulate all turns before each optimizer step.

    Decoder activations use gradient checkpointing. Output-head/KL blocks are
    checkpointed separately so full-vocabulary activations are recomputed during
    backward instead of retained for the entire response. No prefix is cut.
    """

    def __init__(self, model, config):
        super().__init__(model, config)
        self.replay = AscentReplayBuffer(config.replay_buffer_size)

    def state_dict(self):
        state = super().state_dict()
        state["replay_buffer"] = copy.deepcopy(self.replay.entries)
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.replay.entries = copy.deepcopy(state.get("replay_buffer", []))

    def _response_hidden(self, ids, response_length):
        import torch
        device = next(self.model.parameters()).device
        inputs = torch.tensor([ids], device=device)
        # Bypass only the LM head, not the LoRA-instrumented decoder modules.
        decoder = self.model.get_base_model().model
        hidden = decoder(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                         use_cache=False, return_dict=True).last_hidden_state
        # Input is prompt + response[:-1]. The last L states predict all L tokens.
        # Only transfer response states to the output-head device. Tensor.to keeps
        # the student's gradient path across devices and checkpoint recomputation.
        head_device = self.model.get_base_model().get_output_embeddings().weight.device
        return hidden[:, -response_length:, :].to(head_device)

    def _block_loss(self, student_hidden, teacher_hidden):
        head = self.model.get_base_model().get_output_embeddings()
        import torch
        with torch.no_grad():
            teacher_logits = head(teacher_hidden)
        kl_sum = reverse_kl_sum if self.config.kl_direction == "reverse" else forward_kl_sum
        return kl_sum(head(student_hidden), teacher_logits)

    def train(self, steps, privileged_ids, context_limit, *, episode_id=None):
        """Sample once per update call; give each sampled episode equal weight.

        Each old episode keeps its own immutable prefixes, responses and teacher
        record. Both optimizer steps reuse this batch of successful episodes.
        """
        entries = self.replay.prepare(steps, privileged_ids, episode_id)
        batch = self.replay.sample(entries, self.config.replay_batch_size)
        all_steps, all_records, weights = [], [], []
        for entry in batch:
            valid_steps = [s for s in entry["steps"] if s["output_token_ids"]]
            all_steps.extend(valid_steps)
            all_records.extend([entry["teacher_token_ids"]] * len(valid_steps))
            weights.extend([1 / len(batch) / len(valid_steps)] * len(valid_steps))
        result = self.train_turns(all_steps, all_records, context_limit, turn_weights=weights)
        if not result["skipped"]:
            self.replay.entries = entries
        result.update(self.replay.summary(batch, self.config.replay_batch_size),
                      loss_reduction="mean_of_episode_turn_token_means")
        return result

    def train_turns(self, steps, privileged_ids_by_turn, context_limit, *, turn_weights=None):
        """Shared Eq. 8 optimizer with an explicit privileged prefix for each turn.

        TT-OPSD supplies different turn-local records. Keep episode-level gradient
        accumulation here: calling train separately per turn changes the algorithm.
        """
        import torch
        from torch.utils.checkpoint import checkpoint
        start = time.perf_counter()
        turns = []
        if len(steps) != len(privileged_ids_by_turn):
            raise ValueError("Privileged prefixes must align with every rollout turn")
        if turn_weights is not None and len(turn_weights) != len(steps):
            raise ValueError("Loss weights must align with every rollout turn")
        # Validate the entire episode before changing parameters or optimizer state.
        for step, privileged_ids in zip(steps, privileged_ids_by_turn):
            prompt, response = step["input_token_ids"], step["output_token_ids"]
            if not response:
                continue
            if not prompt:
                raise ValueError("Distillation requires the exact nonempty rollout prefix")
            ids = prompt + response[:-1]
            if len(privileged_ids) + len(ids) > context_limit:
                raise ValueError(f"Distillation teacher context overflow at turn {step['step']}: "
                                 f"{len(privileged_ids) + len(ids)} > {context_limit}; "
                                 "the rollout prefix and privileged record will not be truncated")
            turns.append((step["step"], ids, len(response), privileged_ids))
        if not turns:
            return {"skipped": True, "kl_direction": self.config.kl_direction,
                    "skip_reason": "no_response_tokens", "optimizer_steps": 0,
                    "total_optimizer_steps": self.steps, "target_tokens": 0, "loss": None,
                    "seconds": time.perf_counter() - start}
        weights = ([w for s, w in zip(steps, turn_weights) if s["output_token_ids"]]
                   if turn_weights is not None else [1 / len(turns)] * len(turns))
        if (any(not math.isfinite(w) or w <= 0 for w in weights)
                or not math.isclose(sum(weights), 1.0)):
            raise ValueError("Loss weights must be positive, finite and sum to one")
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.config.use_cache = False
        records = []
        try:
            for _ in range(self.config.optimizer_steps):
                self.optimizer.zero_grad(set_to_none=True)
                losses = []
                for (turn, ids, length, privileged_ids), weight in zip(turns, weights):
                    self.model.eval()
                    # Exit this context before the student's forward AND backward:
                    # checkpoint recomputation must see the trainable adapter.
                    with self.model.disable_adapter(), torch.no_grad():
                        teacher_hidden = self._response_hidden(privileged_ids + ids, length).detach().clone()
                    self.model.train()
                    student_hidden = self._response_hidden(ids, length)
                    loss = student_hidden.new_zeros((), dtype=torch.float32)
                    for offset in range(0, length, self.config.logits_chunk_tokens):
                        end = offset + self.config.logits_chunk_tokens
                        loss = loss + checkpoint(
                            self._block_loss, student_hidden[:, offset:end], teacher_hidden[:, offset:end],
                            use_reentrant=False,
                        )
                    loss = loss / length * weight
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite distillation KL; episode update not committed")
                    loss.backward()
                    losses.append({"turn": turn, "target_tokens": length,
                                   "kl": loss.item() / weight, "loss_weight": weight})
                    del loss, student_hidden, teacher_hidden
                norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.config.max_grad_norm,
                                                      error_if_nonfinite=True)
                self.optimizer.step()
                self.steps += 1
                records.append({"optimizer_step": self.steps, "gradient_norm": norm.item(),
                                "loss": sum(r["kl"] * r["loss_weight"] for r in losses), "turns": losses})
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self.model.gradient_checkpointing_disable()
            self.model.config.use_cache = True
            self.model.eval()
        return {"skipped": False, "kl_direction": self.config.kl_direction,
                "optimizer_steps": len(records), "total_optimizer_steps": self.steps,
                "target_tokens": sum(length for _, _, length, _ in turns), "training_turns": len(turns),
                "updates": records, "loss": sum(r["loss"] for r in records) / len(records),
                "seconds": time.perf_counter() - start}
