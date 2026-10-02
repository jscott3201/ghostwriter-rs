"""Host checks for parameter-only conversion on the real reduced official architecture."""
import torch

from ghostwriter_trl.cuda_lora import model as cuda_model
from ghostwriter_trl.cuda_lora.capture import fixture
from ghostwriter_trl.cuda_lora.state import measure
from ghostwriter_trl.lora.shapes import EMBEDDING, HEAD


def test_parameter_only_materialization_preserves_buffers_and_aliases(monkeypatch):
    # CPU placement exercises the allocation/copy implementation without claiming CUDA execution.
    monkeypatch.setattr(cuda_model, "require_cuda", lambda: torch.device("cpu"))
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with fixture(None) as source:
            path, config, _, _ = source._consume()
            first, _ = cuda_model.load_base(config, path / "model.safetensors")
            state = measure(first)
            assert all(p.dtype == torch.bfloat16 and not p.requires_grad for p in first.parameters())
            assert all(b.dtype == torch.float32 for b in first.buffers())
            params = dict(first.named_parameters(remove_duplicate=False))
            assert params[EMBEDDING] is params[HEAD]
            assert any(not row["persistent"] and "inv_freq" in name for name, row in state["buffers"]["tensors"].items())
            second, _ = cuda_model.load_base(config, path / "model.safetensors")
            assert measure(second) == state
            assert first is not second
    finally:
        torch.set_num_threads(previous)
