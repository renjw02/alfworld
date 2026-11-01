"""Sequential evaluate-then-update protocol with transactional episode commits."""
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import numpy as np
from .environment import TextEnvironment
from .cheatsheet import curator_messages
from .logging import AttemptLog, commit_checkpoint, load_checkpoint, run_lock, write_json
from .methods import Method
from .model import make_backend
from .prompting import initial_message, observation_message, parse_react, terminal_message, system_message
from .stream import data_root, prepare_manifest, validate_manifest
from .types import Episode


def seed_all(seed, config):
    random.seed(seed)
    np.random.seed(seed)
    if config.model.backend == "hf":
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)


def rng_state(config):
    state = {"python": random.getstate(), "numpy": np.random.get_state()}
    if config.model.backend == "hf":
        import torch
        state["torch"] = torch.get_rng_state()
        if config.model.device.startswith("cuda"):
            state["cuda"] = torch.cuda.get_rng_state(config.model.device)
    return state


def restore_rng(state, config):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if "torch" in state:
        import torch
        torch.set_rng_state(state["torch"])
        if "cuda" in state:
            torch.cuda.set_rng_state(state["cuda"], config.model.device)


def metadata():
    packages = {}
    for name in ("alfworld", "textworld", "torch", "transformers", "peft", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"git_commit": commit, "git_dirty": dirty, "packages": packages,
            "python": sys.version, "protocol": "valid_seen_streaming_adaptation"}


def run_episode(config, entry, position, method, log, env_factory=TextEnvironment):
    episode = Episode(entry["episode_id"], entry["task_type"], position)
    env = env_factory(data_root(config.stream) / entry["game"], config.max_steps, config.stream.seed + position)
    score = 0.0
    calls = 0
    success = False
    end_reason = "step_limit"
    started = time.perf_counter()
    try:
        observation, info = env.reset()
        commands = list(info["admissible_commands"])
        episode.messages.append(initial_message(episode.episode_id, observation, commands, config.action_mode))
        log.append({"event": "reset", "episode_id": episode.episode_id, "position": position,
                    "observation": observation, "admissible_commands": commands})
        for step in range(1, config.max_steps + 1):
            messages = method.build_context(episode)
            # Retain the attempted input even if inference raises before producing a step.
            write_json(log.path / "last_request.json", {"step": step, "messages": messages,
                                                       "model_version": method.version})
            generated = method.backend.generate(messages)
            parsed = parse_react(generated.text, generated.truncated)
            prior_observation = observation
            prior_commands = commands[:]
            record = {"event": "step", "step": step, "model_version": method.version,
                      "observation": prior_observation, "messages": messages, "prompt": generated.prompt,
                      "input_token_ids": generated.input_ids, "output_token_ids": generated.output_ids,
                      "input_tokens": len(generated.input_ids), "output_tokens": len(generated.output_ids),
                      "raw_output": generated.text, "reasoning": parsed.reasoning, "action": parsed.action,
                      "format_error": parsed.error, "generation_truncated": generated.truncated,
                      "executed_action": None, "admissible_commands": prior_commands,
                      "invalid_action": parsed.error is None and parsed.action not in prior_commands,
                      "generation_seconds": generated.seconds}
            # If env.step fails, preserve the generated reasoning/action as well as its input.
            write_json(log.path / "last_generation.json", record)
            # Keep raw model output, including malformed output, as the training target.
            episode.messages.append({"role": "assistant", "content": generated.text})
            if parsed.error:
                feedback = "Format error: " + parsed.error + " Use Reasoning: followed by one final Action: line. No action was executed."
                episode.messages.append({"role": "user", "content": feedback})
                done = False
            else:
                observation, score, done, info = env.step(parsed.action)
                score = float(score)
                calls += 1
                commands = list(info["admissible_commands"])
                success = bool(info.get("won", False))
                record["executed_action"] = parsed.action
                feedback = observation
                episode.messages.append(observation_message(observation, commands, config.action_mode, score))
            if success:
                end_reason = "success"
            elif step == config.max_steps:
                end_reason = "step_limit"
            elif done:
                end_reason = "environment_end"
            record.update(feedback=feedback, next_observation=observation, next_admissible_commands=commands,
                          score=score, environment_done=bool(done), success=success,
                          episode_done=bool(done or success or step == config.max_steps))
            log.append(record)
            # Exact prompts live on disk; only compact records are retained in memory.
            episode.steps.append({k: v for k, v in record.items() if k not in
                                  ("messages", "prompt", "input_token_ids", "output_token_ids")})
            if done or success:
                break
        episode.outcome = {"success": success, "end_reason": end_reason, "score": score,
                           "attempts": step, "environment_calls": calls,
                           "interaction_seconds": time.perf_counter() - started}
        episode.messages.append(terminal_message(episode.outcome))
        write_json(log.path / "episode.json", episode.to_dict())
        log.append({"event": "terminal", **episode.outcome})
        return episode
    finally:
        env.close()


def run(config, resume=False, backend_factory=make_backend, env_factory=TextEnvironment):
    config.validate()
    root = Path(config.output_dir).resolve()
    with run_lock(root):
        if (root / "config.json").exists():
            if not resume:
                raise ValueError("Output directory already contains a run; use --resume or a new directory")
            saved_config = json.loads((root / "config.json").read_text())
            # Older runs predate this storage-only setting; their checkpoints remain readable.
            saved_config.setdefault("checkpoint_interval", 10)
            saved_config.setdefault("cheatsheet_max_new_tokens", 2048)
            if saved_config != config.to_dict():
                raise ValueError("Resume configuration differs from the saved configuration")
            saved_metadata = json.loads((root / "metadata.json").read_text())
            if saved_metadata["packages"] != metadata()["packages"]:
                raise ValueError("Installed package versions changed since the original run")
            manifest = validate_manifest(json.loads((root / "manifest.json").read_text()), config.stream)
        else:
            if resume:
                raise ValueError("Cannot resume: no saved run")
            if any(p.name != ".lock" for p in root.iterdir()):
                raise ValueError("Output directory is not empty")
            manifest = prepare_manifest(config.stream)
            write_json(root / "manifest.json", manifest)
            write_json(root / "config.json", config.to_dict())
            write_json(root / "metadata.json", metadata())
        # Keep all traces from one invocation together, including resumed attempts.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        trace_root = root / "traces" / stamp
        trace_root.mkdir(parents=True, exist_ok=False)
        write_json(trace_root / "invocation.json", {
            "started_at_utc": stamp, "resume": resume, "config": config.to_dict(),
            "metadata": metadata(), "system_message": system_message(config.action_mode),
        })
        print(f"Execution traces: {trace_root}", flush=True)
        seed_all(config.stream.seed, config)
        backend = backend_factory(config)
        backend_metadata = backend.metadata()
        if (root / "model.json").exists():
            if json.loads((root / "model.json").read_text()) != backend_metadata:
                raise ValueError("Model identity changed since the original run")
        else:
            write_json(root / "model.json", backend_metadata)
        method = Method(config, backend)
        position, completed = 0, []
        if (root / "commit.json").exists():
            state, completed = load_checkpoint(root)
            method.load_state_dict(state["method"])
            if "backend" in state:
                backend.load_state_dict(state["backend"])
            restore_rng(state["rng"], config)
            position = state["next_position"]
        else:
            commit_checkpoint(root, {"next_position": 0, "method": method.state_dict(),
                                    "rng": rng_state(config)}, [], model_checkpoint=False)
        entries = manifest["episodes"][:config.stream.limit]
        for index in range(position, len(entries)):
            log = AttemptLog(trace_root, index)
            try:
                episode = run_episode(config, entries[index], index, method, log, env_factory)
                # Evaluation is complete before any adaptation takes place.
                if config.method == "dc-cumulative":
                    write_json(log.path / "curator_request.json", {
                        "messages": curator_messages(method.cheatsheet, episode, config.cheatsheet_max_new_tokens),
                        "max_new_tokens": config.cheatsheet_max_new_tokens,
                        "cheatsheet_version": method.cheatsheet_version,
                    })
                update = method.observe_episode(episode)
                if update is not None:
                    write_json(log.path / "update.json", update)
                summary = {"position": index, "episode_id": episode.episode_id,
                           "task_type": episode.task_type, "attempt_dir": str(log.path.relative_to(root)),
                           **episode.outcome, "model_version_after": method.version,
                           "input_tokens": sum(s["input_tokens"] for s in episode.steps),
                           "output_tokens": sum(s["output_tokens"] for s in episode.steps),
                           "format_errors": sum(s["format_error"] is not None for s in episode.steps),
                           "invalid_actions": sum(s["invalid_action"] for s in episode.steps),
                           "reasoning_characters": sum(len(s["reasoning"] or "") for s in episode.steps),
                           "generation_seconds": sum(s["generation_seconds"] for s in episode.steps),
                           "training_seconds": update.get("seconds", 0.0) if update else 0.0}
                if config.method == "dc-cumulative":
                    summary.update(cheatsheet_version_after=method.cheatsheet_version,
                                   curator_input_tokens=update["input_tokens"],
                                   curator_output_tokens=update["output_tokens"],
                                   curator_generation_seconds=update["generation_seconds"],
                                   cheatsheet_update_accepted=update["accepted"])
                completed.append(summary)
                training = config.is_ntp
                save_model = training and ((index + 1) % config.checkpoint_interval == 0
                                           or index + 1 == len(entries))
                if not training or save_model:
                    state = {"next_position": index + 1, "method": method.state_dict(),
                             "rng": rng_state(config)}
                    if save_model:
                        state["backend"] = backend.state_dict()
                    # Training results become committed only with their matching weights.
                    commit_checkpoint(root, state, completed, model_checkpoint=save_model)
                print(f"[{index + 1}/{len(entries)}] {config.method}: {episode.outcome}", flush=True)
            except BaseException as exc:
                write_json(log.path / "error.json", {"type": type(exc).__name__, "message": str(exc),
                                                    "traceback": traceback.format_exc(), "committed": False})
                raise
            finally:
                log.close()
        from .analysis import summarize
        return summarize(root)
