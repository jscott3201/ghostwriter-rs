"""This suite always requires the actual isolated Gemma profile and pinned local files."""
import os
from pathlib import Path

import pytest

from ghostwriter_trl.tokenizer import load_tokenizer


@pytest.fixture(scope="session")
def profile():
    return "gemma4_e2b_text_v1"


@pytest.fixture(scope="session")
def tokenizer(profile):
    directory = os.environ.get("GW_TRL_GEMMA_TOKENIZER")
    if not directory:
        pytest.fail("GW_TRL_GEMMA_TOKENIZER must name the pinned Gemma text-profile files")
    return load_tokenizer(Path(directory), profile=profile)
