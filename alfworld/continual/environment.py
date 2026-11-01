"""One explicitly selected TextWorld game; no expert, implicit cycling, or auto reset."""
import uuid
import textworld
import textworld.gym
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler


class TextEnvironment:
    def __init__(self, game, max_steps, seed):
        env_id = textworld.gym.register_game(
            str(game), textworld.EnvInfos(won=True, admissible_commands=True, score=True),
            max_episode_steps=max_steps, wrappers=[AlfredDemangler(shuffle=False)],
            name="continual-" + uuid.uuid4().hex,
        )
        self.env = textworld.gym.make(env_id)
        self.env.seed(seed)
        self.env_id = env_id

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()
        # TextWorld maintains its own Gym registry. Release each episode's entry.
        textworld.gym.registry.pop(self.env_id, None)
