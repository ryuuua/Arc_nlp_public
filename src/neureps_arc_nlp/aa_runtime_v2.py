from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

_CUDA_DEVICE_RE = re.compile(r"cuda(?::(?P<index>[0-9]+))?\Z")
_VALID_CUBLAS_WORKSPACE_CONFIGS = frozenset({":16:8", ":4096:8"})


@dataclass(frozen=True)
class TorchRuntimeV2:
    """Resolved fail-closed Torch runtime for versioned AA solvers."""

    device: Any
    metadata: dict[str, object]


def strict_bool_v2(value: object, *, name: str) -> bool:
    """Parse an explicit v2 boolean without Python truthiness coercion."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError(f"{name} must be a boolean or the explicit string 'true' or 'false'.")


def normalize_torch_device_v2(device: object) -> tuple[str, int | None]:
    """Validate the public v2 device grammar without importing Torch."""

    if not isinstance(device, str):
        raise ValueError("device must be one of: 'cpu', 'cuda', or 'cuda:<index>'.")
    normalized = device.strip().lower()
    if normalized == "cpu":
        return normalized, None
    match = _CUDA_DEVICE_RE.fullmatch(normalized)
    if match is None:
        raise ValueError("device must be one of: 'cpu', 'cuda', or 'cuda:<index>'.")
    index_text = match.group("index")
    return normalized, (None if index_text is None else int(index_text))


def configure_torch_runtime_v2(
    *,
    seed: int,
    device: Any,
    deterministic: Any,
    torch_module: Any | None = None,
) -> TorchRuntimeV2:
    """Resolve a strict AA-v2 Torch runtime and return auditable metadata.

    CUDA determinism is configured before any availability or device query.
    If CUDA was initialized before the required cuBLAS workspace contract was
    present, the function fails instead of claiming deterministic execution.
    """

    normalized_device, requested_index = normalize_torch_device_v2(device)
    deterministic_flag = strict_bool_v2(deterministic, name="deterministic")

    workspace_added_by_this_call = False
    if normalized_device.startswith("cuda") and deterministic_flag:
        configured_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if configured_workspace in (None, ""):
            if torch_module is not None and _cuda_is_initialized(torch_module):
                raise RuntimeError(
                    "CUBLAS_WORKSPACE_CONFIG must be set before CUDA initialization "
                    "for deterministic AA v2 execution."
                )
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            workspace_added_by_this_call = True
        elif configured_workspace not in _VALID_CUBLAS_WORKSPACE_CONFIGS:
            allowed = ", ".join(sorted(_VALID_CUBLAS_WORKSPACE_CONFIGS))
            raise RuntimeError(
                "Invalid CUBLAS_WORKSPACE_CONFIG for deterministic AA v2 execution; "
                f"expected one of: {allowed}."
            )

    if torch_module is None:
        try:
            import torch as imported_torch
        except Exception as exc:  # pragma: no cover - import failure is environment-specific
            raise RuntimeError("PyTorch is required for the requested AA v2 runtime.") from exc
        torch = imported_torch
    else:
        torch = torch_module

    if (
        workspace_added_by_this_call
        and normalized_device.startswith("cuda")
        and _cuda_is_initialized(torch)
    ):
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be set before CUDA initialization "
            "for deterministic AA v2 execution."
        )

    resolved_index: int | None = None
    if normalized_device.startswith("cuda"):
        if not bool(torch.cuda.is_available()):
            raise RuntimeError("CUDA was requested for AA v2, but PyTorch reports no CUDA device.")
        device_count = int(torch.cuda.device_count())
        resolved_index = 0 if requested_index is None else requested_index
        if resolved_index < 0 or resolved_index >= device_count:
            raise RuntimeError(
                f"CUDA device index {resolved_index} is unavailable; visible device count is {device_count}."
            )
        resolved_device = f"cuda:{resolved_index}"
    else:
        resolved_device = "cpu"

    torch.manual_seed(int(seed))
    if resolved_index is not None:
        torch.cuda.manual_seed_all(int(seed))

    torch.use_deterministic_algorithms(deterministic_flag, warn_only=False)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic_flag
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False

    torch_device = torch.device(resolved_device)
    metadata: dict[str, object] = {
        "runtime_contract": "aa_torch_runtime_v2",
        "device_requested": normalized_device,
        "device_used": resolved_device,
        "device_index": resolved_index,
        "deterministic_requested": deterministic_flag,
        "deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
            if hasattr(torch, "is_deterministic_algorithms_warn_only_enabled")
            else False
        ),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_benchmark": bool(getattr(torch.backends.cudnn, "benchmark", False)),
        "cudnn_deterministic": bool(getattr(torch.backends.cudnn, "deterministic", False)),
        "torch_version": str(getattr(torch, "__version__", "unknown")),
        "cuda_version": str(getattr(getattr(torch, "version", None), "cuda", None)),
    }
    if resolved_index is not None:
        properties = torch.cuda.get_device_properties(resolved_index)
        metadata.update(
            {
                "device_name": str(getattr(properties, "name", "unknown")),
                "device_uuid": _optional_text(getattr(properties, "uuid", None)),
                "device_total_memory_bytes": int(getattr(properties, "total_memory", 0)),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            }
        )
    return TorchRuntimeV2(device=torch_device, metadata=metadata)


def _cuda_is_initialized(torch_module: object) -> bool:
    cuda = getattr(torch_module, "cuda", None)
    is_initialized = getattr(cuda, "is_initialized", None)
    return bool(is_initialized()) if callable(is_initialized) else False


def _optional_text(value: object) -> str | None:
    return None if value is None else str(value)
