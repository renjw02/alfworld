"""Full-transcript or assistant-masked NTP with exactly-once next-token targets."""
import time


def token_chunks(token_ids, size):
    if size < 2:
        raise ValueError("A training chunk needs a context token and a target")
    # The final token of the previous chunk is context, not a repeated target.
    for start in range(0, len(token_ids) - 1, size - 1):
        yield token_ids[start:start + size]


class NTPTrainer:
    def __init__(self, model, config):
        import torch
        self.model = model
        self.config = config
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            self.parameters, lr=config.learning_rate, weight_decay=config.weight_decay,
        )
        self.steps = 0

    def train(self, token_ids, labels=None):
        import torch
        labels = list(token_ids) if labels is None else labels
        if len(labels) != len(token_ids):
            raise ValueError("Labels must align with input tokens")
        start = time.perf_counter()
        device = next(self.model.parameters()).device
        self.model.train()
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()
        self.model.config.use_cache = False
        records = []
        try:
            for chunk, targets in zip(token_chunks(token_ids, self.config.chunk_tokens),
                                      token_chunks(labels, self.config.chunk_tokens)):
                target_count = sum(label != -100 for label in targets[1:])
                if not target_count:
                    continue
                target_tensor = torch.tensor([targets], device=device)
                target_tensor[:, 0] = -100
                inputs = torch.tensor([chunk], device=device)
                self.optimizer.zero_grad(set_to_none=True)
                result = self.model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                    labels=target_tensor, use_cache=False)
                if not torch.isfinite(result.loss):
                    raise RuntimeError("Non-finite NTP loss; episode update not committed")
                result.loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.config.max_grad_norm,
                                                      error_if_nonfinite=True)
                self.optimizer.step()
                self.steps += 1
                records.append({"loss": result.loss.item(), "gradient_norm": norm.item(),
                                "target_tokens": target_count, "optimizer_step": self.steps})
                del result, inputs, target_tensor
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self.model.gradient_checkpointing_disable()
            self.model.config.use_cache = True
            self.model.eval()
        targets = sum(r["target_tokens"] for r in records)
        return {"tokens": len(token_ids), "target_tokens": targets, "chunks": records,
                "optimizer_steps": len(records), "total_optimizer_steps": self.steps,
                "loss": sum(r["loss"] * r["target_tokens"] for r in records) / targets if targets else None,
                "skipped": not bool(targets),
                "seconds": time.perf_counter() - start}

    def state_dict(self):
        return {"optimizer": self.optimizer.state_dict(), "steps": self.steps}

    def load_state_dict(self, state):
        # PyTorch may retain CPU tensor references from the supplied state.
        # Loading must not let later steps mutate a reusable checkpoint snapshot.
        import copy
        self.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        self.steps = state["steps"]
