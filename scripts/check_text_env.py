"""Run a real ALFWorld text game without Torch or a GPU."""

import argparse
import random
from pathlib import Path

import yaml

from alfworld.info import ALFWORLD_DATA
from alfworld.agents.environment import get_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[1] / "configs/base_config.yaml")
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    with args.config.open() as stream:
        config = yaml.safe_load(stream)
    config["env"]["type"] = "AlfredTWEnv"
    config["general"]["use_cuda"] = False
    config["dataset"]["num_train_games"] = 1
    # Request admissible commands without requiring a training agent or Torch.
    config["general"]["training_method"] = "dqn"
    game = get_environment("AlfredTWEnv")(config, train_eval="train")
    if not game.game_files:
        raise RuntimeError(
            f"No playable games in {ALFWORLD_DATA}. "
            "Run: uv run alfworld-download --text-only"
        )

    env = game.init_env(batch_size=1)
    rng = random.Random(42)
    try:
        env.seed(42)
        observations, infos = env.reset()
        assert len(observations) == 1 and observations[0]
        print(observations[0])
        for step in range(1, args.steps + 1):
            commands = infos["admissible_commands"][0]
            assert commands, "The game returned no admissible commands"
            action = rng.choice(commands)
            observations, scores, dones, infos = env.step([action])
            assert len(observations) == 1 and observations[0]
            print(f"Step {step}: {action} (score={scores[0]}, done={dones[0]})")
            if dones[0]:
                break
        print(f"PASS: text environment reset and {step} step(s) completed.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
