"""Exercise real PyTorch in a fresh process without leaking global settings."""
import os
import subprocess
import sys
import unittest


class DeterminismTests(unittest.TestCase):
    def test_seed_all_enforces_determinism_and_repeats_updates(self):
        subprocess.run([sys.executable, "-c", """
import os
import random
import numpy as np
import torch
from alfworld.continual.config import Config
from alfworld.continual.runner import seed_all

config = Config()
config.model.backend = 'hf'
seed_all(42, config)
assert torch.are_deterministic_algorithms_enabled()
assert not torch.is_deterministic_algorithms_warn_only_enabled()
assert not torch.backends.cudnn.benchmark
assert torch.backends.cudnn.deterministic
assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
devices = ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else [])
for device in devices:
    def trial():
        seed_all(42, config)
        layer = torch.nn.Linear(32, 16, device=device)
        x = torch.randn(8, 32, device=device)
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)
        layer(x).square().mean().backward()
        optimizer.step()
        return random.random(), np.random.random(), layer(x).detach()
    first, second = trial(), trial()
    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2]), device
print('Deterministic forward/backward/update passed:', devices)
"""], check=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
