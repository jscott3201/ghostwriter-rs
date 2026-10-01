"""Owned random state for adapter initialization, actual training and fresh reload."""
from contextlib import contextmanager
import random


@contextmanager
def seeded_runtime(seed):
    """Apply the recorded seed before PEFT initialization and restore caller CPU RNG states."""
    import numpy as np
    import torch
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        try:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            yield
        finally:
            try:
                random.setstate(python_state)
            finally:
                np.random.set_state(numpy_state)
