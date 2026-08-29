# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest

from vllm.entrypoints.llm import LLM
from vllm.entrypoints.serve.sleep.api_router import resume as http_resume
from vllm.entrypoints.serve.sleep.api_router import suspend as http_suspend
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.core_client import AsyncMPClient, InprocClient, SyncMPClient
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.executor.abstract import Executor


class _TestExecutor(Executor):
    collective_rpc_mock: Mock

    def _init_executor(self) -> None:
        pass

    def collective_rpc(
        self, method, timeout=None, args=(), kwargs=None, non_block: bool = False
    ):
        return self.collective_rpc_mock(
            method,
            timeout=timeout,
            args=args,
            kwargs=kwargs,
            non_block=non_block,
        )

    def check_health(self) -> None:
        pass


def _make_executor() -> _TestExecutor:
    executor = object.__new__(_TestExecutor)
    executor.is_sleeping = False
    executor.sleep_mode = None
    executor.sleeping_tags = set()
    executor.collective_rpc_mock = Mock(return_value=[])
    return executor


def test_executor_suspend_resume_cycle() -> None:
    executor = _make_executor()

    executor.suspend(level=1)

    assert executor.is_sleeping
    assert executor.sleep_mode == "suspend"
    assert executor.sleeping_tags == {"weights", "kv_cache"}
    executor.collective_rpc_mock.assert_called_once_with(
        "suspend", timeout=None, args=(), kwargs={"level": 1}, non_block=False
    )

    executor.resume(tags=["weights"])

    assert not executor.is_sleeping
    assert executor.sleep_mode is None
    assert not executor.sleeping_tags
    assert executor.collective_rpc_mock.call_args_list == [
        call(
            "suspend",
            timeout=None,
            args=(),
            kwargs={"level": 1},
            non_block=False,
        ),
        call(
            "resume",
            timeout=None,
            args=(),
            kwargs={"tags": ["weights"]},
            non_block=False,
        ),
    ]


def test_executor_rejects_crossed_sleep_and_suspend_pairs() -> None:
    executor = _make_executor()
    executor.suspend()

    with pytest.raises(RuntimeError, match="completed with resume"):
        executor.wake_up()

    executor = _make_executor()
    executor.sleep()

    with pytest.raises(RuntimeError, match="completed with wake_up"):
        executor.resume()


@pytest.mark.parametrize(("level", "clear_cache"), [(0, False), (1, True)])
def test_engine_core_suspend_resume_order(level: int, clear_cache: bool) -> None:
    engine_core = object.__new__(EngineCore)
    calls = Mock()
    engine_core.model_executor = Mock()
    engine_core.model_executor.suspend.side_effect = calls.executor_suspend
    engine_core.model_executor.resume.side_effect = calls.executor_resume

    def pause_scheduler(**kwargs):
        calls.pause_scheduler(**kwargs)
        return None

    engine_core.pause_scheduler = Mock(side_effect=pause_scheduler)
    engine_core.resume_scheduler = Mock(side_effect=lambda: calls.resume_scheduler())

    assert engine_core.suspend(level=level, mode="abort") is None
    engine_core.resume(tags=["weights"])

    assert calls.mock_calls == [
        call.pause_scheduler(mode="abort", clear_cache=clear_cache),
        call.executor_suspend(level),
        call.executor_resume(["weights"]),
        call.resume_scheduler(),
    ]


def test_engine_core_waits_for_async_pause_before_suspend() -> None:
    engine_core = object.__new__(EngineCore)
    pause_future: Future[None] = Future()
    engine_core.pause_scheduler = Mock(return_value=pause_future)
    engine_core.model_executor = Mock()
    engine_core.model_executor.suspend.return_value = None

    result = engine_core.suspend(level=1, mode="wait")

    assert isinstance(result, Future)
    engine_core.model_executor.suspend.assert_not_called()

    pause_future.set_result(None)

    assert result.result() is None
    engine_core.model_executor.suspend.assert_called_once_with(1)


def test_offline_suspend_resume_arguments_reach_engine_core_client() -> None:
    llm = object.__new__(LLM)
    llm.llm_engine = Mock()

    llm.suspend(level=1, mode="keep")
    llm.resume(tags=["weights"])

    llm.llm_engine.suspend.assert_called_once_with(level=1, mode="keep")
    llm.llm_engine.resume.assert_called_once_with(["weights"])

    engine = object.__new__(LLMEngine)
    engine.engine_core = Mock()
    engine.logger_manager = Mock()

    engine.suspend(level=1, mode="keep")
    engine.resume(tags=["weights"])

    engine.engine_core.suspend.assert_called_once_with(1, "keep")
    engine.engine_core.resume.assert_called_once_with(["weights"])
    assert engine.logger_manager.record_sleep_state.call_args_list == [
        call(1, 1),
        call(0, 0),
    ]

    engine.engine_core.reset_mock()
    engine.logger_manager.reset_mock()
    engine.suspend(level=2)
    engine.engine_core.suspend.assert_not_called()
    engine.logger_manager.record_sleep_state.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query_tags", "expected_tags"),
    [([], None), (["weights"], ["weights"])],
)
async def test_http_suspend_resume_arguments_reach_async_engine(
    query_tags: list[str], expected_tags: list[str] | None
) -> None:
    engine_client = Mock()
    engine_client.suspend = AsyncMock()
    engine_client.resume = AsyncMock()
    query_params = Mock()
    query_params.get.side_effect = lambda name, default: {
        "level": "1",
        "mode": "keep",
    }.get(name, default)
    query_params.getlist.return_value = query_tags
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(engine_client=engine_client),
        ),
        query_params=query_params,
    )

    await http_suspend(request)
    await http_resume(request)

    engine_client.suspend.assert_awaited_once_with(1, "keep")
    engine_client.resume.assert_awaited_once_with(expected_tags)


@pytest.mark.asyncio
async def test_async_engine_suspend_resume_arguments_reach_core_client() -> None:
    engine = object.__new__(AsyncLLM)
    engine.engine_core = Mock()
    engine.engine_core.suspend_async = AsyncMock()
    engine.engine_core.resume_async = AsyncMock()
    engine.logger_manager = Mock()

    await engine.suspend(level=1, mode="keep")
    await engine.resume(tags=["weights"])

    engine.engine_core.suspend_async.assert_awaited_once_with(1, "keep")
    engine.engine_core.resume_async.assert_awaited_once_with(["weights"])
    assert engine.logger_manager.record_sleep_state.call_args_list == [
        call(1, 1),
        call(0, 0),
    ]

    engine.engine_core.suspend_async.reset_mock()
    engine.logger_manager.reset_mock()
    await engine.suspend(level=2)
    engine.engine_core.suspend_async.assert_not_awaited()
    engine.logger_manager.record_sleep_state.assert_not_called()


def test_sync_core_clients_forward_suspend_resume_arguments() -> None:
    inproc_client = object.__new__(InprocClient)
    inproc_client.engine_core = Mock()
    inproc_client.engine_core.suspend.return_value = None

    inproc_client.suspend(level=1, mode="keep")
    inproc_client.resume(tags=["weights"])

    inproc_client.engine_core.suspend.assert_called_once_with(1, "keep")
    inproc_client.engine_core.resume.assert_called_once_with(["weights"])

    mp_client = object.__new__(SyncMPClient)
    mp_client.call_utility = Mock()

    mp_client.suspend(level=1, mode="keep")
    mp_client.resume(tags=["weights"])

    assert mp_client.call_utility.call_args_list == [
        call("suspend", 1, "keep"),
        call("resume", ["weights"]),
    ]


@pytest.mark.asyncio
async def test_async_core_client_forwards_suspend_resume_arguments() -> None:
    client = object.__new__(AsyncMPClient)
    client.call_utility_async = AsyncMock()

    await client.suspend_async(level=1, mode="keep")
    await client.resume_async(tags=["weights"])

    assert client.call_utility_async.await_args_list == [
        call("suspend", 1, "keep"),
        call("resume", ["weights"]),
    ]
