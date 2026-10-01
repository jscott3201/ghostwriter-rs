"""Qualification requires actual pinned local dependencies, tokenizer, and the Rust binary."""
import os
from pathlib import Path
import pytest

from ghostwriter_trl.tokenizer import load_tokenizer
from .prepared_fixtures import current_fixtures


@pytest.fixture(scope="session")
def profile():
    return "qwen3_text_v1"


@pytest.fixture(scope="session")
def tokenizer():
    directory = os.environ.get("GW_TRL_TOKENIZER")
    if not directory:
        pytest.fail("GW_TRL_TOKENIZER must name the exact locally acquired pinned fixture")
    return load_tokenizer(Path(directory))


@pytest.fixture(scope="session")
def gw():
    executable = Path(os.environ.get("GW_TRL_GW", "../../target/debug/gw")).resolve()
    if not executable.is_file():
        pytest.fail("build gw and set GW_TRL_GW to that exact executable")
    return executable


@pytest.fixture(scope="session")
def historical_fixture_dir():
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixture_dir(historical_fixture_dir, gw, tokenizer, profile, tmp_path_factory):
    return current_fixtures(tmp_path_factory.mktemp("prepared-current"), historical_fixture_dir,
                            gw, tokenizer, profile)
