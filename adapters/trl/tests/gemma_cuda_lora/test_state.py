"""Small host controls for the actual CUDA producer's complete state measurement."""
import torch
from ghostwriter_trl.cuda_lora.state import measure


def test_nonpersistent_buffer_and_alias_are_bound():
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False))
    model.register_parameter("tied", model.weight)
    model.register_buffer("rope", torch.tensor([2.0]), persistent=False)
    before = measure(model)
    model.rope.add_(1)
    after = measure(model)
    assert before["frozen"] == after["frozen"]
    assert before["buffers"] != after["buffers"]
    assert before["frozen"]["tensors"]["weight"]["alias"] == "tied"
    assert before["buffers"]["tensors"]["rope"]["persistent"] is False


def test_storage_dtype_is_separate_from_equal_float_values():
    model = torch.nn.Linear(2, 2, bias=False).requires_grad_(False)
    model.weight.data.fill_(1)
    before = measure(model)
    model.weight.data = model.weight.data.to(torch.bfloat16)
    assert before["frozen"]["state_id"] != measure(model)["frozen"]["state_id"]
