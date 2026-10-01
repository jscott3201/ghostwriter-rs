"""One pinned, offline adapter with explicit unshifted causal language-model labels."""
import os

# Must precede imports of Transformers, Datasets, Hub, and TRL, including via library usage.
for _name in (
    "HF_HUB_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "HF_DATASETS_OFFLINE",
    "TRANSFORMERS_OFFLINE",
):
    os.environ[_name] = "1"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

__version__ = "0.1.0"
