# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm.v1.engine import core as engine_core_module
from vllm.v1.engine.core import EngineCore
from vllm.v1.kv_cache_interface import KVCacheConfig


def test_platform_adjusts_kv_config_before_scheduler_and_worker_init(monkeypatch):
    events = []
    generated_config = KVCacheConfig(
        num_blocks=2, kv_cache_tensors=[], kv_cache_groups=[]
    )
    adjusted_config = KVCacheConfig(
        num_blocks=1, kv_cache_tensors=[], kv_cache_groups=[]
    )
    model_executor = MagicMock()
    model_executor.get_kv_cache_specs.return_value = [{}]
    model_executor.initialize_from_config.side_effect = lambda configs: events.append(
        ("initialize", configs)
    )

    engine_core = EngineCore.__new__(EngineCore)
    engine_core.model_executor = model_executor
    engine_core.available_gpu_memory_for_kv_cache = -1
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=1024),
        cache_config=SimpleNamespace(num_gpu_blocks=None),
        validate_block_size=MagicMock(),
    )

    def get_configs(*args):
        events.append(("generate", args))
        return [generated_config]

    def adjust_configs(config, executor, configs):
        events.append(("adjust", configs))
        assert config is vllm_config
        assert executor is model_executor
        return [adjusted_config]

    def scheduler_config(configs):
        events.append(("scheduler", configs))
        return configs[0]

    monkeypatch.setattr(engine_core_module, "get_kv_cache_configs", get_configs)
    monkeypatch.setattr(
        engine_core_module.current_platform, "adjust_kv_cache_configs", adjust_configs
    )
    monkeypatch.setattr(
        engine_core_module, "generate_scheduler_kv_cache_config", scheduler_config
    )

    result = engine_core._initialize_kv_caches(vllm_config)

    assert result is adjusted_config
    assert [event[0] for event in events] == [
        "generate",
        "adjust",
        "scheduler",
        "initialize",
    ]
    assert model_executor.initialize_from_config.call_args.args == ([adjusted_config],)
    assert vllm_config.cache_config.num_gpu_blocks == 1
