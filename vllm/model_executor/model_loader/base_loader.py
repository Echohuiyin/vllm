# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod

import regex as re
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.platforms import current_platform
from vllm.tracing import instrument
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import set_default_torch_dtype

logger = init_logger(__name__)

_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")

# multiproc_pipe uses parameter addresses as the stable bridge between model
# loading, allocator records, and the first forward after resume. Keep these
# dictionaries process-local: every NPU worker owns one model and one Copier.
layer_to_addr: dict[str, list[int]] = {}
addr_to_layer: dict[int, str] = {}


def init_map(num_layers: int) -> None:
    """Initialize the process-local layer/address maps for multiproc_pipe."""
    if num_layers <= 0:
        raise ValueError("multiproc_pipe requires at least one model layer")
    layer_to_addr.clear()
    # Allocator blocks that cannot be associated with a model tensor still
    # belong to the immutable weight image.  The Copier restores this bucket
    # before public and layer-local weights.
    layer_to_addr["unknown"] = []
    layer_to_addr["pub"] = []
    layer_to_addr.update({f"layers.{index}": [] for index in range(num_layers)})
    addr_to_layer.clear()


def locate_splited_weights_on_layer(model: nn.Module) -> None:
    """Record post-processed parameter and buffer addresses by model layer.

    The intentionally retained ``splited`` spelling is part of the original
    multiproc_pipe integration API.
    """
    if not layer_to_addr:
        raise RuntimeError("multiproc_pipe layer map was not initialized")

    for addresses in layer_to_addr.values():
        addresses.clear()
    addr_to_layer.clear()

    def layer_order(layer_name: str) -> int:
        if layer_name == "unknown":
            return -2
        return -1 if layer_name == "pub" else int(layer_name.split(".")[1])

    named_tensors = list(model.named_parameters(remove_duplicate=False))
    named_tensors.extend(model.named_buffers(remove_duplicate=False))
    for name, tensor in named_tensors:
        address = tensor.data_ptr()
        if address == 0:
            continue
        layer_name = "pub"
        if (match := _LAYER_PATTERN.search(name)) is not None:
            layer_index = int(match.group(1))
            candidate = f"layers.{layer_index}"
            if candidate not in layer_to_addr:
                raise RuntimeError(
                    f"Weight {name!r} belongs to out-of-range layer {layer_index}"
                )
            layer_name = candidate

        current = addr_to_layer.get(address)
        if current is None or layer_order(layer_name) < layer_order(current):
            addr_to_layer[address] = layer_name

    for address, layer_name in addr_to_layer.items():
        layer_to_addr[layer_name].append(address)
    for addresses in layer_to_addr.values():
        addresses.sort()


def _multiproc_pipe_enabled(vllm_config: VllmConfig) -> bool:
    additional_config = vllm_config.additional_config
    return (
        isinstance(additional_config, dict)
        and additional_config.get("multiproc_pipe", False) is True
    )


class BaseModelLoader(ABC):
    """Base class for model loaders."""

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

    @abstractmethod
    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Load weights into a model. This standalone API allows
        inplace weights loading for an already-initialized model"""
        raise NotImplementedError

    @instrument(span_name="Load model")
    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        """Load a model with the given configurations."""
        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = (
            device_config.device if load_config.device is None else load_config.device
        )
        target_device = torch.device(load_device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(
                    vllm_config=vllm_config, model_config=model_config, prefix=prefix
                )

            log_model_inspection(model)

            logger.debug("Loading weights on %s ...", load_device)
            # Quantization does not happen in `load_weights` but after it
            self.load_weights(model, model_config)

            # Log peak GPU memory after loading weights. This is needed
            # to have test coverage on peak memory for online quantization.
            if current_platform.is_cuda():
                peak_memory = torch.accelerator.max_memory_allocated()
                logger.debug_once(
                    "Peak GPU memory after loading weights: %s GiB",
                    format_gib(peak_memory),
                    scope="local",
                )

            process_weights_after_loading(model, model_config, target_device)

            if _multiproc_pipe_enabled(vllm_config):
                init_map(model_config.get_num_layers(vllm_config.parallel_config))
                locate_splited_weights_on_layer(model)

        return model.eval()


def log_model_inspection(model: nn.Module) -> None:
    """Log model structure if VLLM_LOG_MODEL_INSPECTION=1."""
    if not envs.VLLM_LOG_MODEL_INSPECTION:
        return

    from vllm.model_inspection import format_model_inspection

    logger.info("vLLM model structure:\n%s", format_model_inspection(model))
