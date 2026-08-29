# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.model_loader.base_loader import (
    addr_to_layer,
    init_map,
    layer_to_addr,
    locate_splited_weights_on_layer,
)


class _LayeredModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.public_weight = nn.Parameter(torch.zeros(1))
        self.layers = nn.ModuleList([nn.Linear(1, 1), nn.Linear(1, 1)])
        self.register_buffer("public_buffer", torch.ones(1))


def test_locate_splited_weights_builds_postprocessed_layer_address_map() -> None:
    model = _LayeredModel()
    init_map(2)

    locate_splited_weights_on_layer(model)

    assert list(layer_to_addr) == ["unknown", "pub", "layers.0", "layers.1"]
    assert layer_to_addr["unknown"] == []
    public_addresses = {
        model.public_weight.data_ptr(),
        model.public_buffer.data_ptr(),
    }
    assert public_addresses <= set(layer_to_addr["pub"])
    assert {parameter.data_ptr() for parameter in model.layers[0].parameters()} <= set(
        layer_to_addr["layers.0"]
    )
    assert {parameter.data_ptr() for parameter in model.layers[1].parameters()} <= set(
        layer_to_addr["layers.1"]
    )
    assert set(addr_to_layer) == {
        address for addresses in layer_to_addr.values() for address in addresses
    }


def test_locate_splited_weights_rejects_out_of_range_layer() -> None:
    model = _LayeredModel()
    init_map(1)

    try:
        locate_splited_weights_on_layer(model)
    except RuntimeError as error:
        assert "out-of-range layer 1" in str(error)
    else:
        raise AssertionError("multiproc_pipe accepted a model layer outside its map")


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("qwen2.py", "Qwen2Model"),
        ("deepseek_v2.py", "DeepseekV2Model"),
    ],
)
def test_multiproc_pipe_model_waits_use_allocator_health_checks(
    module_name: str,
    class_name: str,
) -> None:
    repository = Path(__file__).resolve().parents[3]
    tree = ast.parse(
        (repository / "vllm" / "model_executor" / "models" / module_name).read_text()
    )
    model_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    forward = next(
        node
        for node in model_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )

    calls = [node.func for node in ast.walk(forward) if isinstance(node, ast.Call)]
    assert any(
        isinstance(function, ast.Attribute)
        and function.attr == "wait_for_layer"
        and isinstance(function.value, ast.Name)
        and function.value.id == "allocator"
        for function in calls
    )
    assert not any(
        isinstance(function, ast.Name) and function.id == "get_layer_ready_event"
        for function in calls
    )
