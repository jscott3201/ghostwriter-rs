"""Public standalone reader failure controls with weak-reference host sentinels."""
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
import gc
import weakref

import pytest

from ghostwriter_trl.artifact import ContractError
from ghostwriter_trl.cuda_lora import bundle, environment, model, runtime, state


@pytest.mark.parametrize("failure", ["state", "attach", "load", "measure"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_failed_public_reload_releases_objects_and_preserves_failures(tmp_path, monkeypatch, failure, cleanup_fails):
    class Sentinel:
        pass
    refs = []
    def allocate():
        value = Sentinel()
        refs.append(weakref.ref(value))
        return value
    @contextmanager
    def captured(path):
        try:
            yield tmp_path, tmp_path / "snapshot"
        finally:
            if cleanup_fails:
                raise OSError("capture cleanup failure")
    prepared = tmp_path / "prepared"
    prepared.write_bytes(b"prepared")
    report = {"declarations": {"cuda_dependencies": {}, "cuda_runtime": {}, "final_state": {}},
              "base_model": {}, "final_adapter": {}, "prepared_build_id": "expected"}
    monkeypatch.setattr(bundle, "_captured", captured)
    monkeypatch.setattr(bundle, "native_report", lambda *a: report)
    monkeypatch.setattr(bundle, "_extract", lambda *a: {"prepared.gwsft": prepared, "base/config.json": prepared,
        "base/model.safetensors": prepared, "final/adapter_model.safetensors": prepared})
    monkeypatch.setattr(bundle, "verify_prepared", lambda *a: SimpleNamespace(build_id="different"))
    monkeypatch.setattr(bundle, "read_config", lambda *a: ({}, None))
    monkeypatch.setattr(runtime, "runtime", nullcontext)
    monkeypatch.setattr(environment, "observe", lambda: ({}, {}))
    def load(*args):
        base = allocate()
        if failure == "load":
            raise ContractError("load semantic failure")
        return base, {}
    def attach(base, *args):
        attached = allocate()
        if failure == "attach":
            raise ContractError("attach semantic failure")
        return attached, {}
    def measure(actual):
        if failure == "measure":
            raise ContractError("measure semantic failure")
        return {}
    monkeypatch.setattr(bundle, "load_base", load)
    monkeypatch.setattr(bundle, "reload_adapter", attach)
    monkeypatch.setattr(state, "measure", measure)
    with pytest.raises(ContractError) as held:
        bundle.read_checkpoint(prepared, prepared, None)
    gc.collect()
    assert refs and all(ref() is None for ref in refs)
    if cleanup_fails:
        assert "capture cleanup failure" in str(held.value)
        assert "semantic failure" in str(held.value) or "differs from captured" in str(held.value)
