from __future__ import annotations

import importlib
from dataclasses import dataclass

INTERNAL_SUPPORT_MODULE = "gpm_vlm_internal_multimodal"
READINESS_READY = "READY"
READINESS_PARTIAL = "PARTIAL"
READINESS_NOT_READY = "NOT READY"


@dataclass(frozen=True)
class ImportProbeResult:
    ok: bool
    error: str
    module: object | None = None


@dataclass(frozen=True)
class GPMDependencyStatus:
    pillow: ImportProbeResult
    llama_cpp: ImportProbeResult
    internal_support: ImportProbeResult
    llama_cpp_version: str
    readiness: str


def probe_import(module_name: str) -> ImportProbeResult:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - import errors are environment dependent.
        return ImportProbeResult(ok=False, error=str(exc), module=None)
    return ImportProbeResult(ok=True, error="", module=module)


def probe_package_local_import(module_name: str, package_name: str) -> ImportProbeResult:
    normalized_package = str(package_name or "").strip()
    if not normalized_package:
        return probe_import(module_name)

    # Prefer package-local resolution for GPM internal modules, but keep an
    # absolute fallback for environments that expose modules at top level.
    try:
        module = importlib.import_module(f".{module_name}", package=normalized_package)
        return ImportProbeResult(ok=True, error="", module=module)
    except Exception as rel_exc:  # pragma: no cover - import errors are environment dependent.
        absolute_probe = probe_import(module_name)
        if absolute_probe.ok:
            return absolute_probe
        return ImportProbeResult(ok=False, error=str(rel_exc), module=None)


def llama_cpp_version(llama_cpp_module: object | None) -> str:
    if llama_cpp_module is None:
        return ""
    return str(getattr(llama_cpp_module, "__version__", "") or "").strip()


def detect_llama_cuda_status(llama_cpp_module: object | None) -> str:
    if llama_cpp_module is None:
        return "UNKNOWN"

    probe_names = ("llama_supports_cuda", "llama_supports_cublas")
    saw_explicit_false = False
    for probe_name in probe_names:
        probe_fn = getattr(llama_cpp_module, probe_name, None)
        if not callable(probe_fn):
            continue
        try:
            value = bool(probe_fn())
        except Exception:
            continue
        if value:
            return "YES"
        saw_explicit_false = True

    system_info_fn = getattr(llama_cpp_module, "llama_print_system_info", None)
    if callable(system_info_fn):
        try:
            raw_info = system_info_fn()
            if isinstance(raw_info, bytes):
                text = raw_info.decode("utf-8", errors="replace")
            else:
                text = str(raw_info)
            if "cuda" in text.casefold():
                return "YES"
        except Exception:
            pass

    if saw_explicit_false:
        return "NO"
    return "UNKNOWN"


def compute_startup_readiness(
    *,
    pillow_import_ok: bool,
    llama_cpp_import_ok: bool,
    internal_support_import_ok: bool,
) -> str:
    if not pillow_import_ok:
        return READINESS_NOT_READY
    if llama_cpp_import_ok and internal_support_import_ok:
        return READINESS_READY
    return READINESS_PARTIAL


def collect_dependency_status() -> GPMDependencyStatus:
    pillow_probe = probe_import("PIL")
    llama_probe = probe_import("llama_cpp")
    internal_probe = probe_package_local_import(INTERNAL_SUPPORT_MODULE, __package__ or "")
    readiness = compute_startup_readiness(
        pillow_import_ok=pillow_probe.ok,
        llama_cpp_import_ok=llama_probe.ok,
        internal_support_import_ok=internal_probe.ok,
    )
    return GPMDependencyStatus(
        pillow=pillow_probe,
        llama_cpp=llama_probe,
        internal_support=internal_probe,
        llama_cpp_version=llama_cpp_version(llama_probe.module),
        readiness=readiness,
    )


def _probe_status_text(probe: ImportProbeResult) -> str:
    return "OK" if probe.ok else "MISSING"


def print_startup_diagnostics() -> None:
    status = collect_dependency_status()
    print("[GPM startup] Dependency diagnostics")
    print(f"[GPM startup] Pillow import: {_probe_status_text(status.pillow)}")
    print(f"[GPM startup] llama_cpp import: {_probe_status_text(status.llama_cpp)}")
    print(f"[GPM startup] llama-cpp-python version: {status.llama_cpp_version or '<unknown>'}")
    print(
        f"[GPM startup] Internal support import ({INTERNAL_SUPPORT_MODULE}): "
        f"{_probe_status_text(status.internal_support)}"
    )
    if not status.internal_support.ok:
        print(f"[GPM startup] Internal support detail: {status.internal_support.error}")
    print(f"[GPM startup] Internal scanner readiness: {status.readiness}")
