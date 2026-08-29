# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from collections import deque
from concurrent.futures import Future
from threading import Thread
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call, patch

import pytest

from vllm.entrypoints.llm import LLM
from vllm.entrypoints.serve.sleep.api_router import resume as http_resume
from vllm.entrypoints.serve.sleep.api_router import suspend as http_suspend
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.core_client import (
    AsyncMPClient,
    InprocClient,
    SyncMPClient,
    _fail_utility_results,
)
from vllm.v1.engine.exceptions import EngineDeadError
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.engine.utils import CoreEngineProcManager
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc
from vllm.v1.worker.worker_base import WorkerFatalError, WorkerRetryableError


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
    engine_core.vllm_config = Mock()
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


def test_engine_core_validates_sleep_before_pausing_scheduler() -> None:
    engine_core = object.__new__(EngineCore)
    engine_core.vllm_config = Mock()
    engine_core.pause_scheduler = Mock()

    with (
        patch(
            "vllm.v1.engine.core.current_platform.validate_sleep_level",
            side_effect=ValueError("unsupported level"),
        ),
        pytest.raises(ValueError, match="unsupported level"),
    ):
        engine_core.sleep(level=2)

    engine_core.pause_scheduler.assert_not_called()


@pytest.mark.parametrize("level", [-1, 3, True, 1.0])
def test_engine_core_rejects_invalid_sleep_level_before_pausing(level) -> None:
    engine_core = object.__new__(EngineCore)
    engine_core.pause_scheduler = Mock()

    with pytest.raises(ValueError, match="0, 1, or 2"):
        engine_core.sleep(level=level)

    engine_core.pause_scheduler.assert_not_called()


@pytest.mark.parametrize("method", ["wake_up", "resume"])
def test_engine_core_validates_wake_before_calling_executor(method: str) -> None:
    engine_core = object.__new__(EngineCore)
    engine_core.vllm_config = Mock()
    engine_core.model_executor = Mock()
    engine_core.resume_scheduler = Mock()

    validator = "validate_wake_tags" if method == "wake_up" else "validate_resume_tags"
    with (
        patch(
            f"vllm.v1.engine.core.current_platform.{validator}",
            side_effect=ValueError("unsupported tags"),
        ),
        pytest.raises(ValueError, match="unsupported tags"),
    ):
        getattr(engine_core, method)(tags=["weights"])

    getattr(engine_core.model_executor, method).assert_not_called()
    engine_core.resume_scheduler.assert_not_called()


def test_engine_core_waits_for_async_pause_before_suspend() -> None:
    engine_core = object.__new__(EngineCore)
    engine_core.vllm_config = Mock()
    pause_future: Future[None] = Future()
    engine_core.pause_scheduler = Mock(return_value=pause_future)
    engine_core.model_executor = Mock()
    engine_core.model_executor.is_sleeping = False
    engine_core.model_executor.suspend.return_value = None
    engine_core.resume_scheduler = Mock()
    engine_core.is_scheduler_paused = Mock(return_value=True)

    result = engine_core.suspend(level=1, mode="wait")

    assert isinstance(result, Future)
    engine_core.model_executor.suspend.assert_not_called()
    assert not engine_core.is_sleeping()

    with pytest.raises(RuntimeError, match="transition is still in progress"):
        engine_core.resume()

    with pytest.raises(RuntimeError, match="transition is still in progress"):
        EngineCore.resume_scheduler(engine_core)

    engine_core.model_executor.resume.assert_not_called()
    engine_core.resume_scheduler.assert_not_called()

    pause_future.set_result(None)

    assert result.result() is None
    engine_core.model_executor.suspend.assert_called_once_with(1)

    engine_core.model_executor.is_sleeping = True
    assert engine_core.is_sleeping()
    engine_core.resume()
    engine_core.model_executor.resume.assert_called_once_with(None)
    engine_core.resume_scheduler.assert_called_once_with()


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


def test_worker_fatal_error_is_reported_before_process_exit() -> None:
    worker_proc = object.__new__(WorkerProc)
    worker_proc.rpc_broadcast_mq = Mock()
    worker_proc.rpc_broadcast_mq.dequeue.return_value = ("suspend", (), {}, None)
    worker_proc.worker = Mock()
    worker_proc.worker.suspend.side_effect = WorkerFatalError("ambiguous Famem state")
    worker_proc.enqueue_output = Mock()
    worker_proc.rank = 0

    with pytest.raises(WorkerFatalError, match="ambiguous Famem state"):
        worker_proc.worker_busy_loop()

    response = worker_proc.enqueue_output.call_args.args[0]
    assert isinstance(response, RuntimeError)
    assert str(response) == "ambiguous Famem state"


def test_worker_retryable_error_uses_distinct_response_status() -> None:
    worker_proc = object.__new__(WorkerProc)
    worker_proc.worker_response_mq = Mock()

    worker_proc.enqueue_output(WorkerRetryableError("lease is busy"))

    worker_proc.worker_response_mq.enqueue.assert_called_once_with(
        (WorkerProc.ResponseStatus.RETRYABLE_FAILURE, "lease is busy")
    )


def test_lifecycle_rpc_failure_terminates_complete_worker_group() -> None:
    executor = object.__new__(MultiprocExecutor)
    executor.vllm_config = Mock(additional_config={"multiproc_pipe": True})
    executor.rpc_broadcast_mq = Mock()
    response_mq = Mock()
    response_mq.dequeue.return_value = (
        WorkerProc.ResponseStatus.FAILURE,
        "rank entered an ambiguous state",
    )
    executor.response_mqs = [response_mq]
    executor.futures_queue = deque()
    executor._init_failure_state()
    failure_callback = Mock()
    executor.failure_callback = failure_callback
    executor.shutdown = Mock()

    with pytest.raises(RuntimeError, match="ambiguous state"):
        executor.collective_rpc("suspend")

    assert executor.is_failed
    executor.shutdown.assert_called_once_with()
    assert executor.failure_callback is None
    failure_callback.assert_called_once_with()


@pytest.mark.parametrize(
    "statuses,retryable",
    [
        ([WorkerProc.ResponseStatus.RETRYABLE_FAILURE] * 2, True),
        (
            [
                WorkerProc.ResponseStatus.SUCCESS,
                WorkerProc.ResponseStatus.RETRYABLE_FAILURE,
            ],
            False,
        ),
    ],
)
def test_lifecycle_rpc_retries_only_when_every_rank_rejects_without_state_change(
    statuses, retryable
) -> None:
    executor = object.__new__(MultiprocExecutor)
    executor.vllm_config = Mock(additional_config={"multiproc_pipe": True})
    executor.rpc_broadcast_mq = Mock()
    executor.response_mqs = [Mock(), Mock()]
    for mq, status in zip(executor.response_mqs, statuses, strict=True):
        mq.dequeue.return_value = (
            status,
            None if status is WorkerProc.ResponseStatus.SUCCESS else "lease is busy",
        )
    executor.futures_queue = deque()
    executor._init_failure_state()
    executor.shutdown = Mock()

    with pytest.raises(RuntimeError) as error:
        executor.collective_rpc("resume")

    assert isinstance(error.value, WorkerRetryableError) is retryable
    assert executor.is_failed is not retryable
    assert executor.shutdown.called is not retryable


def test_lifecycle_rpc_timeout_terminates_complete_worker_group() -> None:
    executor = object.__new__(MultiprocExecutor)
    executor.vllm_config = Mock(additional_config={"multiproc_pipe": True})
    executor.rpc_broadcast_mq = Mock()
    response_mq = Mock()
    response_mq.dequeue.side_effect = TimeoutError("rank did not reply")
    executor.response_mqs = [response_mq]
    executor.futures_queue = deque()
    executor._init_failure_state()
    executor.shutdown = Mock()

    with pytest.raises(TimeoutError, match="RPC call to suspend timed out"):
        executor.collective_rpc("suspend", timeout=0.1)

    assert executor.is_failed
    executor.shutdown.assert_called_once_with()


def test_lifecycle_rpc_enqueue_timeout_terminates_complete_worker_group() -> None:
    executor = object.__new__(MultiprocExecutor)
    executor.vllm_config = Mock(additional_config={"multiproc_pipe": True})
    executor.rpc_broadcast_mq = Mock()
    executor.rpc_broadcast_mq.enqueue.side_effect = TimeoutError(
        "workers stopped reading"
    )
    executor.response_mqs = []
    executor.futures_queue = deque()
    executor._init_failure_state()
    executor.shutdown = Mock()

    with pytest.raises(TimeoutError, match="workers stopped reading"):
        executor.collective_rpc("suspend", timeout=0.1)

    assert executor.is_failed
    executor.shutdown.assert_called_once_with()
    enqueue_timeout = executor.rpc_broadcast_mq.enqueue.call_args.kwargs["timeout"]
    assert 0 < enqueue_timeout <= 0.1


@pytest.mark.parametrize("response_timeout", [False, True])
def test_lifecycle_rpc_error_is_retryable_without_multiproc_pipe(
    response_timeout: bool,
) -> None:
    executor = object.__new__(MultiprocExecutor)
    executor.vllm_config = Mock(additional_config={})
    executor.rpc_broadcast_mq = Mock()
    response_mq = Mock()
    if response_timeout:
        response_mq.dequeue.side_effect = TimeoutError("rank did not reply")
    else:
        response_mq.dequeue.return_value = (
            WorkerProc.ResponseStatus.FAILURE,
            "wake-up validation failed before changing mappings",
        )
    executor.response_mqs = [response_mq]
    executor.futures_queue = deque()
    executor._init_failure_state()
    executor.shutdown = Mock()

    expected_error = "timed out" if response_timeout else "validation failed"
    with pytest.raises((RuntimeError, TimeoutError), match=expected_error):
        executor.collective_rpc("wake_up", timeout=0.1)

    assert not executor.is_failed
    executor.shutdown.assert_not_called()


def test_lifecycle_rpc_failure_notifies_engine_when_shutdown_fails() -> None:
    executor = object.__new__(MultiprocExecutor)
    executor._init_failure_state()
    failure_callback = Mock()
    executor.failure_callback = failure_callback
    executor.shutdown = Mock(side_effect=RuntimeError("queue cleanup failed"))

    with pytest.raises(RuntimeError, match="queue cleanup failed"):
        executor._handle_worker_failure()

    assert executor.is_failed
    assert executor.failure_callback is None
    failure_callback.assert_called_once_with()


def test_concurrent_worker_failures_shutdown_and_notify_once() -> None:
    executor = object.__new__(MultiprocExecutor)
    executor._init_failure_state()
    failure_callback = Mock()
    executor.failure_callback = failure_callback
    executor.shutdown = Mock()

    executor._handle_worker_failure()
    executor._handle_worker_failure()

    executor.shutdown.assert_called_once_with()
    failure_callback.assert_called_once_with()


def test_resume_scheduler_rejects_physically_sleeping_executor() -> None:
    engine_core = object.__new__(EngineCore)
    engine_core.model_executor = Mock(is_sleeping=True)
    engine_core.scheduler = Mock()

    with pytest.raises(RuntimeError, match="call wake_up or resume first"):
        engine_core.resume_scheduler()

    engine_core.scheduler.set_pause_state.assert_not_called()


def test_engine_death_fails_pending_sync_utility_call() -> None:
    future: Future[None] = Future()
    utility_results = {1: future}

    _fail_utility_results(utility_results)

    assert not utility_results
    with pytest.raises(EngineDeadError):
        future.result(timeout=0.1)


def test_multiproc_pipe_engine_core_process_uses_frontend_parent_guard() -> None:
    context = Mock()
    process = context.Process.return_value
    process.exitcode = None
    config = SimpleNamespace(
        additional_config={"multiproc_pipe": True},
        parallel_config=SimpleNamespace(data_parallel_size=1, use_ray=False),
    )

    with (
        patch("vllm.v1.engine.utils.get_mp_context", return_value=context),
        patch("vllm.v1.engine.utils.weakref.finalize", return_value=Mock()),
        patch("vllm.v1.engine.utils.os.getpid", return_value=321),
        patch.object(CoreEngineProcManager, "finished_procs", return_value={}),
    ):
        CoreEngineProcManager(
            local_engine_count=1,
            start_index=0,
            local_start_index=0,
            vllm_config=config,
            local_client=True,
            handshake_address="ipc:///unused",
            executor_class=_TestExecutor,
            log_stats=False,
        )

    process_kwargs = context.Process.call_args.kwargs["kwargs"]
    assert process_kwargs["expected_parent_pid"] == 321
    process.start.assert_called_once_with()

    with (
        patch("vllm.v1.engine.utils.get_mp_context", return_value=Mock()),
        patch("vllm.v1.engine.utils.threading.current_thread", return_value=object()),
        patch("vllm.v1.engine.utils.threading.main_thread", return_value=object()),
        pytest.raises(RuntimeError, match="frontend main thread"),
    ):
        CoreEngineProcManager(
            1,
            0,
            0,
            config,
            True,
            "ipc:///unused",
            _TestExecutor,
            False,
        )


@pytest.mark.asyncio
async def test_engine_death_fails_pending_async_utility_call_thread_safely() -> None:
    future = asyncio.get_running_loop().create_future()
    utility_results = {1: future}

    monitor = Thread(target=_fail_utility_results, args=(utility_results,))
    monitor.start()
    monitor.join(timeout=1.0)

    assert not monitor.is_alive()
    assert not utility_results
    with pytest.raises(EngineDeadError):
        await asyncio.wait_for(future, timeout=0.1)
