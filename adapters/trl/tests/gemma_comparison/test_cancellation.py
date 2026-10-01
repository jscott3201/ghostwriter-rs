"""Cancellation settles the owned native process before returning to the caller."""
import pytest
from ghostwriter_trl.comparison.bridge import capture_population
from .test_pair import fixture_population


def test_cancel_during_generation_closes_and_settles_waiting_native(gw):
    database, registration = fixture_population()
    process = None
    with pytest.raises(KeyboardInterrupt):
        with capture_population(gw, database, registration, "test") as bridge:
            process = bridge._process
            assert len(bridge.population["members"]) == 32
            assert process.poll() is None
            raise KeyboardInterrupt
    assert process.poll() is not None
    assert process.stdout.closed
