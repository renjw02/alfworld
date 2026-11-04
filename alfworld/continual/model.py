"""Local Transformers backend and an explicitly labeled control-flow test backend."""
import hashlib
import json
from pathlib import Path
import time
from .types import Generation
from .training import NTPTrainer
from .ascent import AscentTrainer, success_record
from .context import fit_context


def layer_device_map(num_layers, devices):
    """Keep each decoder layer intact and place contiguous blocks on each GPU."""
    if num_layers < len(devices):
        raise ValueError("More GPUs than decoder layers")
    mapping = {"model.embed_tokens": devices[0], "model.rotary_emb": devices[0],
               "model.norm": devices[-1], "lm_head": devices[0]}
    for index in range(num_layers):
        mapping[f"model.layers.{index}"] = devices[index * len(devices) // num_layers]
    return mapping


class HFBackend:
    def __init__(self, config, model=None, tokenizer=None):
        import torch
        from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM
        self.config = config
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(config.model.path, local_files_only=True)
        if model is None:
            if not config.model.device.startswith("cuda") or not torch.cuda.is_available():
                raise RuntimeError("The 9B backend requires a CUDA device; use --backend mock for CPU control-flow tests")
            device_map = {"": config.model.device}
            if config.model.devices:
                config.validate()
                available = []
                for index, device in enumerate(config.model.devices):
                    free, total = torch.cuda.mem_get_info(device)
                    cap = free - 3 * 1024**3
                    if config.model.gpu_memory_gib:
                        cap = min(cap, config.model.gpu_memory_gib[index] * 1024**3)
                    if cap < 8 * 1024**3:
                        raise RuntimeError(f"Insufficient memory on {device} after reserving 3 GiB")
                    torch.cuda.set_per_process_memory_fraction(cap / total, device)
                    available.append(cap)
                if sum(available) < 24 * 1024**3:
                    raise RuntimeError("Less than 24 GiB available across selected GPUs")
                architecture = AutoConfig.from_pretrained(config.model.path, local_files_only=True)
                architecture = getattr(architecture, "text_config", architecture)
                device_map = layer_device_map(architecture.num_hidden_layers, config.model.devices)
            else:
                free, _ = torch.cuda.mem_get_info(config.model.device)
                if free < 24 * 1024**3:
                    raise RuntimeError("Less than 24 GiB GPU memory is free. No quantization or device fallback is applied.")
            model, loading = Qwen3_5ForCausalLM.from_pretrained(
                config.model.path, local_files_only=True, dtype=torch.bfloat16,
                device_map=device_map, output_loading_info=True,
            )
            if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
                raise RuntimeError(f"Incomplete model load: {loading}")
            unexpected = loading.get("unexpected_keys", [])
            if any("visual" not in key and "vision" not in key and "mtp" not in key for key in unexpected):
                raise RuntimeError(f"Unexpected language-model weights: {unexpected}")
        self.model = model
        self.trainer = None
        self.target_modules = []
        self.model.requires_grad_(False)
        if config.trains_parameters:
            training = config.ascent if config.is_ascent else config.training
            from peft import LoraConfig, get_peft_model
            self.model = get_peft_model(model, LoraConfig(
                task_type="CAUSAL_LM", r=training.rank,
                lora_alpha=training.alpha, lora_dropout=training.dropout,
                target_modules=["down_proj"] if config.is_ascent else "all-linear",
                exclude_modules=["lm_head"], bias="none",
            ))
            self.target_modules = [name for name, module in self.model.named_modules() if hasattr(module, "lora_A")]
            if not self.target_modules:
                raise RuntimeError("LoRA matched no modules")
            self.trainer = (AscentTrainer(self.model, training) if config.is_ascent
                            else NTPTrainer(self.model, training))
        self.model.eval()
        self.model.config.use_cache = True

    def _render(self, messages, generation):
        # Some Qwen templates strip <think> regions from earlier assistant text.
        # Protect message bodies so even malformed raw generations remain exact.
        protected = []
        replacements = {}
        for i, message in enumerate(messages):
            marker = f"ALFWORLD_BODY_{i}_" + hashlib.sha256(message["content"].encode()).hexdigest()
            replacements[marker] = message["content"]
            protected.append({**message, "content": marker})
        text = self.tokenizer.apply_chat_template(
            protected, tokenize=False, add_generation_prompt=generation, enable_thinking=False,
        )
        # Replace in one pass so text inside a body can never trigger a second replacement.
        import re
        pattern = "|".join(re.escape(marker) for marker in replacements)
        spans, parts = [], []
        cursor = length = 0
        roles = {marker: message["role"] for marker, message in zip(replacements, messages)}
        for match in re.finditer(pattern, text):
            prefix, body = text[cursor:match.start()], replacements[match[0]]
            parts.extend((prefix, body))
            length += len(prefix)
            if roles[match[0]] == "assistant":
                spans.append((length, length + len(body)))
            length += len(body)
            cursor = match.end()
        parts.append(text[cursor:])
        return "".join(parts), spans

    def serialize(self, messages, generation):
        text, _ = self._render(messages, generation)
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return text, ids

    def generate(self, messages, *, max_new_tokens=None):
        import torch
        max_new_tokens = self.config.model.max_new_tokens if max_new_tokens is None else max_new_tokens
        text_config = getattr(self.model.config, "text_config", self.model.config)
        maximum = min(self.config.model.max_input_tokens, text_config.max_position_embeddings - max_new_tokens)
        if maximum < 1:
            raise ValueError("Output budget exceeds model context length and leaves no input space")
        input_messages, text, ids, truncation = fit_context(
            messages, maximum, self.serialize, self.tokenizer, self.config.model.context_overflow)
        device = next(self.model.parameters()).device
        inputs = torch.tensor([ids], device=device)
        eos = self.tokenizer.eos_token_id
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=inputs, attention_mask=torch.ones_like(inputs),
                max_new_tokens=max_new_tokens, do_sample=False,
                use_cache=True, eos_token_id=eos, pad_token_id=self.tokenizer.pad_token_id or eos,
            )
        output_ids = output[0, len(ids):].tolist()
        truncated = len(output_ids) >= max_new_tokens and output_ids[-1] != eos
        # Strip only terminal EOS; preserve any generated control tokens in the raw record.
        decode_ids = output_ids[:-1] if output_ids and output_ids[-1] == eos else output_ids
        return Generation(self.tokenizer.decode(decode_ids, skip_special_tokens=False), text, ids,
                          output_ids, time.perf_counter() - start, truncated, truncation, input_messages)

    def train_episode(self, messages):
        if self.trainer is None:
            raise RuntimeError("Training requested without a LoRA trainer")
        text, spans = self._render(messages, False)
        if self.config.assistant_only:
            encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            ids = encoded["input_ids"]
            # Only tokens wholly inside raw assistant bodies are prediction targets.
            labels = [token if end > start and any(a <= start and end <= b for a, b in spans)
                      else -100 for token, (start, end) in zip(ids, encoded["offset_mapping"])]
        else:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            labels = list(ids)
        if labels:
            labels[0] = -100
        result = self.trainer.train(ids, labels)
        result.update(training_text=text, training_token_ids=ids, training_labels=labels,
                      loss_scope="assistant" if self.config.assistant_only else "full")
        return result

    def distill_episode(self, episode):
        if not self.config.is_ascent or not episode.outcome.get("success"):
            raise ValueError("ASCENT distillation requires a verified successful episode")
        text, kept, removed = success_record(episode)
        privileged_ids = self.tokenizer.encode(text, add_special_tokens=False)
        architecture = getattr(self.model.config, "text_config", self.model.config)
        result = self.trainer.train(episode.steps, privileged_ids, architecture.max_position_embeddings)
        result.update(kind="ascent", teacher_record=text, teacher_token_ids=privileged_ids,
                      retained_turns=kept, filtered_turns=removed,
                      loss_scope="all_generated_response_tokens_including_eos",
                      loss_reduction="mean_of_turn_token_means")
        return result

    def state_dict(self):
        if self.trainer is None:
            return {}
        from peft import get_peft_model_state_dict
        import torch
        def cpu_copy(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().clone()
            if isinstance(value, dict):
                return {k: cpu_copy(v) for k, v in value.items()}
            if isinstance(value, list):
                return [cpu_copy(v) for v in value]
            if isinstance(value, tuple):
                return tuple(cpu_copy(v) for v in value)
            return value
        return cpu_copy({"adapter": get_peft_model_state_dict(self.model), "trainer": self.trainer.state_dict()})

    def load_state_dict(self, state):
        if self.trainer is not None:
            from peft import set_peft_model_state_dict
            result = set_peft_model_state_dict(self.model, state["adapter"])
            if result.unexpected_keys:
                raise ValueError(f"Unexpected adapter keys: {result.unexpected_keys}")
            self.trainer.load_state_dict(state["trainer"])

    def metadata(self):
        path = Path(self.config.model.path)
        hashes = {}
        for name in ("config.json", "model.safetensors.index.json", "tokenizer_config.json", "chat_template.jinja"):
            if (path / name).exists():
                hashes[name] = hashlib.sha256((path / name).read_bytes()).hexdigest()
        return {"backend": "hf", "model_path": str(path.resolve()), "config_hashes": hashes,
                "weights": [{"name": p.name, "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                            for p in sorted(path.glob("*.safetensors"))],
                "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                "lora_modules": self.target_modules}


class MockBackend:
    """Never reported as a model experiment: only checks the protocol and real environment."""
    def __init__(self, config):
        self.config = config
        self.updates = 0

    def generate(self, messages, *, max_new_tokens=None):
        prompt = json.dumps(messages, ensure_ascii=False)
        from .cheatsheet import CURATOR_SYSTEM
        if messages[0]["content"] == CURATOR_SYSTEM:
            text = "<cheatsheet>Mock reference: inspect observations and use current admissible commands.</cheatsheet>"
            return Generation(text, prompt, list(prompt.encode()), list(text.encode()), 0.0)
        action = "look"
        if "Admissible commands:\n" in messages[-1]["content"]:
            candidates = messages[-1]["content"].split("Admissible commands:\n")[-1].splitlines()
            if candidates:
                action = candidates[0]
        text = f"Reasoning: This is a control-flow smoke test; inspect the environment.\nAction: {action}"
        return Generation(text, prompt, list(prompt.encode()), list(text.encode()), 0.0)

    def train_episode(self, messages):
        self.updates += 1
        return {"mock": True, "optimizer_steps": 0, "training_text": json.dumps(messages), "loss": None}

    def distill_episode(self, episode):
        text, kept, removed = success_record(episode)
        self.updates += 1
        return {"kind": "ascent", "mock": True, "optimizer_steps": 0, "loss": None,
                "teacher_record": text, "retained_turns": kept, "filtered_turns": removed,
                "seconds": 0.0, "skipped": False}

    def state_dict(self):
        return {"updates": self.updates}

    def load_state_dict(self, state):
        self.updates = state["updates"]

    def metadata(self):
        return {"backend": "mock", "scientific_result": False, "token_encoding": "UTF-8 bytes, not model tokens"}


def make_backend(config):
    return MockBackend(config) if config.model.backend == "mock" else HFBackend(config)
