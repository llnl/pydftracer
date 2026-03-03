import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_SOURCE = REPO_ROOT / "python"


def _run_subprocess(code: str, env_overrides: dict[str, str], timeout: int = 15) -> None:
    env = os.environ.copy()
    env.update(env_overrides)

    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        env["PYTHONPATH"] = f"{PYTHON_SOURCE}{os.pathsep}{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = str(PYTHON_SOURCE)

    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0, (
        f"Subprocess failed with code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )


@pytest.mark.subprocess
@pytest.mark.parametrize("enable_value", ["0", "1"])
def test_base_imports_finish_quickly(enable_value: str) -> None:
    code = """
from dftracer.python import ai, dftracer
print("ok")
"""
    _run_subprocess(
        code,
        env_overrides={"DFTRACER_ENABLE": enable_value},
        timeout=15,
    )


@pytest.mark.subprocess
def test_base_and_lazy_dynamo_import_do_not_import_torch() -> None:
    code = """
import sys
import dftracer.python
assert "torch" not in sys.modules, sorted(
    name for name in sys.modules if name.startswith("torch")
)

from dftracer.python import Dynamo, create_backend, dynamo
assert dynamo is not None
assert Dynamo is not None
assert create_backend is not None
assert "torch" not in sys.modules, sorted(
    name for name in sys.modules if name.startswith("torch")
)
print("ok")
"""
    _run_subprocess(
        code,
        env_overrides={"DFTRACER_ENABLE": "1"},
        timeout=15,
    )


@pytest.mark.subprocess
def test_dynamo_compile_loads_torch_only_when_explicitly_used() -> None:
    code = """
import builtins
import os

os.environ["DFTRACER_ENABLE"] = "1"

original_import = builtins.__import__

def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torch" or name.startswith("torch."):
        raise ImportError("blocked torch import for regression test")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = _blocked_import

from dftracer.python import dynamo

try:
    @dynamo.compile
    def identity(x):
        return x
except RuntimeError as exc:
    assert "PyTorch is not available" in str(exc), str(exc)
else:
    raise AssertionError("Expected RuntimeError when torch import is blocked")

print("ok")
"""
    _run_subprocess(
        code,
        env_overrides={"DFTRACER_ENABLE": "1"},
        timeout=15,
    )
