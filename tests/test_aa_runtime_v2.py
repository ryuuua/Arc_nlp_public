from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from neureps_arc_nlp.aa_runtime_v2 import (
    configure_torch_runtime_v2,
    normalize_torch_device_v2,
    strict_bool_v2,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, True), (False, False), ("true", True), (" FALSE ", False)],
)
def test_strict_bool_v2_accepts_only_explicit_booleans(value: Any, expected: bool) -> None:
    assert strict_bool_v2(value, name="flag") is expected


@pytest.mark.parametrize("value", [1, 0, None, "yes", "0", [], object()])
def test_strict_bool_v2_rejects_truthiness_coercion(value: Any) -> None:
    with pytest.raises(ValueError, match="flag must be a boolean"):
        strict_bool_v2(value, name="flag")


@pytest.mark.parametrize(
    ("value", "expected"),
    [("cpu", ("cpu", None)), (" CUDA ", ("cuda", None)), ("cuda:0", ("cuda:0", 0)), ("cuda:12", ("cuda:12", 12))],
)
def test_normalize_torch_device_v2_accepts_exact_grammar(
    value: str, expected: tuple[str, int | None]
) -> None:
    assert normalize_torch_device_v2(value) == expected


@pytest.mark.parametrize("value", ["", "mps", "gpu", "cuda:-1", "cuda:1.5", "cpu:0", 0, None])
def test_normalize_torch_device_v2_rejects_unknown_devices(value: Any) -> None:
    with pytest.raises(ValueError, match="device must be one of"):
        normalize_torch_device_v2(value)


class _FakeCuda:
    def __init__(self, *, available: bool, count: int, initialized: bool = False) -> None:
        self._available = available
        self._count = count
        self._initialized = initialized
        self.seed: int | None = None

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._count

    def is_initialized(self) -> bool:
        return self._initialized

    def manual_seed_all(self, seed: int) -> None:
        self.seed = seed

    def get_device_properties(self, index: int) -> Any:
        return SimpleNamespace(name=f"fake-{index}", uuid=f"uuid-{index}", total_memory=1024)


class _FakeTorch:
    __version__ = "fake"
    version = SimpleNamespace(cuda="fake-cuda")

    def __init__(self, *, available: bool = True, count: int = 1, initialized: bool = False) -> None:
        self.cuda = _FakeCuda(available=available, count=count, initialized=initialized)
        self.backends = SimpleNamespace(
            cudnn=SimpleNamespace(deterministic=False, benchmark=True, allow_tf32=True),
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
        )
        self._deterministic = False
        self._warn_only = True
        self.seed: int | None = None

    def manual_seed(self, seed: int) -> None:
        self.seed = seed

    def use_deterministic_algorithms(self, enabled: bool, *, warn_only: bool) -> None:
        self._deterministic = enabled
        self._warn_only = warn_only

    def are_deterministic_algorithms_enabled(self) -> bool:
        return self._deterministic

    def is_deterministic_algorithms_warn_only_enabled(self) -> bool:
        return self._warn_only

    def device(self, name: str) -> str:
        return name


def test_configure_torch_runtime_v2_resolves_exact_cuda_ordinal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    fake = _FakeTorch(count=2)
    runtime = configure_torch_runtime_v2(
        seed=7, device="cuda:1", deterministic=True, torch_module=fake
    )
    assert runtime.device == "cuda:1"
    assert runtime.metadata["device_used"] == "cuda:1"
    assert runtime.metadata["device_uuid"] == "uuid-1"
    assert runtime.metadata["deterministic_algorithms_enabled"] is True
    assert runtime.metadata["deterministic_warn_only"] is False
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert fake.backends.cuda.matmul.allow_tf32 is False
    assert fake.backends.cudnn.allow_tf32 is False


def test_configure_torch_runtime_v2_rejects_unavailable_cuda_ordinal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with pytest.raises(RuntimeError, match="visible device count is 1"):
        configure_torch_runtime_v2(
            seed=0, device="cuda:1", deterministic=True, torch_module=_FakeTorch(count=1)
        )


def test_configure_torch_runtime_v2_rejects_late_workspace_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(RuntimeError, match="before CUDA initialization"):
        configure_torch_runtime_v2(
            seed=0,
            device="cuda:0",
            deterministic=True,
            torch_module=_FakeTorch(initialized=True),
        )


def test_configure_torch_runtime_v2_rejects_late_workspace_on_import_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    fake = _FakeTorch(initialized=True)
    monkeypatch.setitem(sys.modules, "torch", fake)
    with pytest.raises(RuntimeError, match="before CUDA initialization"):
        configure_torch_runtime_v2(
            seed=0,
            device="cuda:0",
            deterministic=True,
        )
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ


def test_configure_torch_runtime_v2_disables_sticky_deterministic_state() -> None:
    fake = _FakeTorch(available=False, count=0)
    fake._deterministic = True
    runtime = configure_torch_runtime_v2(
        seed=3, device="cpu", deterministic=False, torch_module=fake
    )
    assert runtime.metadata["deterministic_algorithms_enabled"] is False
    assert fake.backends.cudnn.deterministic is False
