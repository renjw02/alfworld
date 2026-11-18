"""Action-level forward KL against a frozen, pre-update projected policy."""
import time

from .negative_ul import NegativeULTrainer


def projected_target(scores, negative_index):
    """Normalize only surviving actions, without subtracting near-unit bad mass."""
    import torch
    if scores.ndim != 1 or len(scores) < 2:
        raise ValueError("Projected-KL requires at least two candidate scores")
    if not 0 <= negative_index < len(scores):
        raise ValueError("Invalid negative candidate index")
    if not torch.isfinite(scores).all():
        raise RuntimeError("Non-finite Projected-KL reference scores")
    logits = scores.detach().clone()
    logits[negative_index] = -torch.inf
    return logits.softmax(0)


def candidate_loss(scores, target):
    # Only evaluate log(q) on positive support; avoid 0 * log(0).
    q = target.detach()
    support = q > 0
    return (q[support] * (q[support].log() - scores.log_softmax(0)[support])).sum()


class ProjectedKLTrainer(NegativeULTrainer):
    def __init__(self, model, config, projected_kl):
        super().__init__(model, config)
        self.optimizer_steps = projected_kl.optimizer_steps

    def train_sample(self, sample):
        import torch
        started = time.perf_counter()
        prefix, targets = sample["prefix_token_ids"], sample["candidate_token_ids"]
        negative = sample["negative_index"]
        self.model.train()
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.enable_input_require_grads()
        self.model.config.use_cache = False
        self.optimizer.zero_grad(set_to_none=True)
        records = []
        try:
            with torch.no_grad():
                reference = torch.stack([self.score(prefix, target) for target in targets])
                q = projected_target(reference, negative)
            for index in range(self.optimizer_steps):
                self.optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    scores = (reference.clone() if index == 0 else
                              torch.stack([self.score(prefix, target) for target in targets]))
                scores.requires_grad_(True)
                loss = candidate_loss(scores, q)
                if not torch.isfinite(scores).all() or not torch.isfinite(loss):
                    raise RuntimeError("Non-finite Projected-KL scores/loss")
                coefficients, = torch.autograd.grad(loss, scores)
                for target, coefficient in zip(targets, coefficients):
                    (self.score(prefix, target) * coefficient).backward()
                norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.config.max_grad_norm,
                                                      error_if_nonfinite=True)
                self.optimizer.step()
                self.steps += 1
                records.append({"optimizer_step": self.steps, "loss": loss.item(),
                                "candidate_log_probs": scores.detach().tolist(),
                                "negative_probability": scores.detach().softmax(0)[negative].item(),
                                "gradient_norm": norm.item()})
            return {"skipped": False, "optimizer_steps": len(records),
                    "total_optimizer_steps": self.steps,
                    "loss": records[-1]["loss"], "substeps": records,
                    "reference_log_probs": reference.tolist(), "target_probabilities": q.tolist(),
                    "seconds": time.perf_counter() - started}
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self.model.gradient_checkpointing_disable()
            self.model.disable_input_require_grads()
            self.model.config.use_cache = True
            self.model.eval()
