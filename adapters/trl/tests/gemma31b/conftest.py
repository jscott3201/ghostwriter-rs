"""Actual official local assets are required; absence is a failed prerequisite."""
import os
from pathlib import Path
import pytest
from ghostwriter_trl.artifact import read_snapshot
from ghostwriter_trl.gemma31b.tokenizer import load_tokenizer

@pytest.fixture(scope='session')
def tokenizer():
    directory = os.environ.get('GW_TRL_GEMMA31B_TOKENIZER')
    if not directory:
        pytest.fail('GW_TRL_GEMMA31B_TOKENIZER must name exact official local assets')
    return load_tokenizer(Path(directory))

@pytest.fixture(scope='session')
def serial_source(gw):
    return read_snapshot(Path(__file__).parents[1] / 'fixtures/v5-gemma31b-serial.parquet', gw)
