import io
import os
import subprocess
import sys
import time
from unittest.mock import Mock

import pytest

from ml_training.hybrid import Runtime, Worker
from ml_training.processes import run_isolated


@pytest.mark.skipif(os.name != "posix", reason="The local model UI uses POSIX process groups")
def test_timeout_kills_descendants_holding_output_pipes(tmp_path):
    child = "import time; print('worker-ready', flush=True); time.sleep(4)"
    parent = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(4)"
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        run_isolated([sys.executable, "-c", parent], cwd=tmp_path, env=os.environ.copy(), timeout=0.5)
    assert "worker-ready" in caught.value.output
    # Killing only the parent leaves the child's inherited pipe open for four seconds.
    assert time.monotonic() - started < 2


def test_isolated_command_preserves_output_and_exit_status(tmp_path):
    result = run_isolated(
        [sys.executable, "-c", "import sys; print('result'); print('diagnostic',file=sys.stderr); sys.exit(7)"],
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout=5,
    )
    assert result.returncode == 7
    assert result.stdout == "result\n" and result.stderr == "diagnostic\n"


def test_worker_escalates_shutdown_and_closes_all_handles():
    worker = Worker.__new__(Worker)
    worker.process = Mock()
    worker.process.poll.return_value = None
    worker.process.stdin, worker.process.stdout = io.StringIO(), io.StringIO()
    worker.process.wait.side_effect = [
        subprocess.TimeoutExpired("worker", 15),
        subprocess.TimeoutExpired("worker", 15),
        0,
    ]
    worker.selector, worker.log = Mock(), io.StringIO()
    worker.close()
    worker.process.terminate.assert_called_once()
    worker.process.kill.assert_called_once()
    assert worker.process.stdin.closed and worker.process.stdout.closed and worker.log.closed
    worker.selector.close.assert_called_once()


def test_runtime_closes_remaining_workers_after_one_cleanup_fails():
    runtime = Runtime.__new__(Runtime)
    first, second = Mock(), Mock()
    first.close.side_effect = OSError("failed cleanup")
    runtime.workers = [first, second]
    with pytest.raises(ExceptionGroup, match="worker cleanup"):
        runtime.close()
    second.close.assert_called_once()
