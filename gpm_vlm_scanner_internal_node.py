from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .gpm_vlm_backend import (
    BACKEND_GGUF,
    OVERWRITE_FAMILY,
    OVERWRITE_SKIP_EXISTING,
    scan_images_with_preset,
)
from .gpm_vlm_model_discovery import (
    AUTO_MMPROJ_OPTION,
    discover_gguf_model_choices,
    discover_mmproj_choices,
    resolve_model_and_mmproj_paths,
)
from .gpm_vlm_runtime_base import RUNTIME_MODE_INTERNAL
from .gpm_vlm_scanner_node import (
    WRITE_SCAN_REPORT_MODES,
    WRITE_SCAN_REPORT_OFF,
    WRITE_SCAN_REPORT_ON,
    _build_status_text,
    _preset_choices,
    _write_scan_report,
)
from .gpm_vlm_presets import GPMVLMPresetStore
from .gpm_vlm_presets import get_preset_generation_settings

EXECUTION_MODE_SUBPROCESS = "SUBPROCESS (Recommended: releases VRAM after scan)"


_INTERNAL_STARTUP_ERROR_PREFIXES = (
    "internal runtime requires llama-cpp-python",
    "internal model file was not found",
    "internal mmproj file was not found",
    "selected GGUF appears to be a text-only model",
    "selected GGUF does not look like a supported internal vision model",
    "model/mmproj may be mismatched",
    "internal_model_name is required for internal mode",
    "mmproj_name is required for internal mode",
    "internal model family could not be inferred safely",
    "internal runtime family inference failed",
    "internal runtime chat handler resolution failed",
    "internal scan correctness is currently verified for Qwen2.5-VL only",
    "unsupported internal multimodal model family",
    "installed llama-cpp-python build does not support internal family",
    "internal runtime failed to load model",
    "internal runtime model is not loaded",
    "internal runtime inference failed",
    "mmproj auto-pair failed",
    "no mmproj GGUF files were found",
    "no GGUF VLM model was selected",
    "selected GGUF model was not found",
    "selected mmproj model was not found",
)


def _build_internal_status_text(summary: dict[str, Any]) -> str:
    error_text = str(summary.get("error", "")).strip()
    if error_text and error_text.startswith(_INTERNAL_STARTUP_ERROR_PREFIXES):
        return f"Startup failed: {error_text}"
    return _build_status_text(summary)


def _model_choices() -> list[str]:
    return discover_gguf_model_choices()


def _mmproj_choices() -> list[str]:
    return discover_mmproj_choices()


def _empty_scan_error(preset_id: str, message: str) -> dict[str, Any]:
    return {
        "ok": False,
        "error": message,
        "preset_id": preset_id,
        "total_found": 0,
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "failures": [],
    }


def _build_internal_scan_request(
    *,
    root_folder: str,
    preset_id: str,
    overwrite_mode: str,
    scan_limit: int,
    model_name: str,
    mmproj_name: str,
    timeout_seconds: int,
    n_ctx: int,
    n_gpu_layers: int,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
    threads: int,
    batch_size: int,
    debug_mode: bool,
    write_scan_report: str,
    model_path_resolved: str = "",
    mmproj_path_resolved: str = "",
) -> dict[str, Any]:
    return {
        "root_folder": root_folder,
        "preset_id": preset_id,
        "overwrite_mode": overwrite_mode,
        "scan_limit": int(scan_limit),
        "write_scan_report": write_scan_report,
        "model_name": model_name,
        "mmproj_name": mmproj_name,
        "model_path_resolved": str(model_path_resolved or ""),
        "mmproj_path_resolved": str(mmproj_path_resolved or ""),
        "timeout_seconds": int(timeout_seconds),
        "n_ctx": int(n_ctx),
        "n_gpu_layers": int(n_gpu_layers),
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "threads": int(threads),
        "batch_size": int(batch_size),
        "debug_mode": bool(debug_mode),
    }


def _run_internal_scan_in_process(
    *,
    request: dict[str, Any],
    keep_model_loaded: bool,
    unload_on_complete: bool,
) -> dict[str, Any]:
    preset_id = str(request.get("preset_id", "")).strip()
    summary: dict[str, Any]
    try:
        store = GPMVLMPresetStore()
        preset = store.get_preset(preset_id)
        if preset is None:
            summary = _empty_scan_error(preset_id, f"preset not found: {preset_id}")
        elif not str(request.get("model_name", "")).strip():
            summary = _empty_scan_error(preset_id, "model_name is required for internal GGUF mode")
        elif not str(request.get("mmproj_name", "")).strip():
            summary = _empty_scan_error(
                preset_id,
                "mmproj_name is required for internal GGUF mode. Use auto-pair or select manually.",
            )
        else:
            preset_temperature, preset_top_p, preset_max_tokens = get_preset_generation_settings(preset)
            raw_temperature = request.get("temperature", None)
            raw_top_p = request.get("top_p", None)
            raw_max_tokens = request.get("max_tokens", None)
            effective_temperature = preset_temperature if raw_temperature is None else float(raw_temperature)
            effective_top_p = preset_top_p if raw_top_p is None else float(raw_top_p)
            effective_max_tokens = preset_max_tokens if raw_max_tokens is None else int(raw_max_tokens)
            summary = scan_images_with_preset(
                root_folder=str(request.get("root_folder", "")),
                preset=preset,
                overwrite_mode=str(request.get("overwrite_mode", OVERWRITE_SKIP_EXISTING)),
                backend_mode=BACKEND_GGUF,
                gguf_model_name="",
                timeout_seconds=int(request.get("timeout_seconds", 180)),
                scan_limit=int(request.get("scan_limit", 0)),
                runtime_mode=RUNTIME_MODE_INTERNAL,
                internal_model_name=str(request.get("model_name", "")),
                internal_mmproj_name=str(request.get("mmproj_name", "")),
                internal_model_path_override=str(request.get("model_path_resolved", "")).strip(),
                internal_mmproj_path_override=str(request.get("mmproj_path_resolved", "")).strip(),
                internal_n_ctx=int(request.get("n_ctx", 4096)),
                internal_n_gpu_layers=int(request.get("n_gpu_layers", -1)),
                internal_temperature=effective_temperature,
                internal_top_p=effective_top_p,
                internal_max_tokens=effective_max_tokens,
                internal_threads=int(request.get("threads", 0)),
                internal_batch_size=int(request.get("batch_size", 512)),
                internal_keep_model_loaded=keep_model_loaded,
                internal_unload_on_complete=unload_on_complete,
                internal_debug_mode=bool(request.get("debug_mode", False)),
            )
    except Exception as exc:
        summary = _empty_scan_error(preset_id, f"internal scan execution failed: {exc}")
    return summary


def _tail_text(text: str, limit: int = 4000) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[-limit:]


def _run_internal_scan_subprocess(
    *,
    request: dict[str, Any],
    timeout_seconds: int,
) -> dict[str, Any]:
    preset_id = str(request.get("preset_id", "")).strip()
    worker_path = Path(__file__).resolve().parent / "gpm_vlm_internal_worker.py"
    start_ts = time.time()
    with tempfile.TemporaryDirectory(prefix="gpm_internal_worker_") as temp_dir:
        temp_root = Path(temp_dir)
        request_path = temp_root / "request.json"
        output_path = temp_root / "output.json"
        request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")

        cmd = [
            sys.executable,
            str(worker_path),
            "--request-json",
            str(request_path),
            "--output-json",
            str(output_path),
        ]
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(Path(__file__).resolve().parent),
                capture_output=True,
                text=True,
                timeout=max(5, int(timeout_seconds)),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = round(time.time() - start_ts, 3)
            summary = _empty_scan_error(preset_id, "internal scan worker timed out")
            summary["worker_return_code"] = None
            summary["worker_elapsed_seconds"] = elapsed
            summary["worker_timed_out"] = True
            summary["worker_stdout_tail"] = _tail_text(getattr(exc, "stdout", "") or "")
            summary["worker_stderr_tail"] = _tail_text(getattr(exc, "stderr", "") or "")
            return summary
        except Exception as exc:
            elapsed = round(time.time() - start_ts, 3)
            summary = _empty_scan_error(preset_id, f"internal scan worker launch failed: {exc}")
            summary["worker_return_code"] = None
            summary["worker_elapsed_seconds"] = elapsed
            return summary

        elapsed = round(time.time() - start_ts, 3)
        if not output_path.exists():
            summary = _empty_scan_error(preset_id, "internal scan worker did not produce output JSON")
            summary["worker_return_code"] = int(completed.returncode)
            summary["worker_elapsed_seconds"] = elapsed
            summary["worker_stdout_tail"] = _tail_text(completed.stdout)
            summary["worker_stderr_tail"] = _tail_text(completed.stderr)
            return summary

        try:
            loaded = json.loads(output_path.read_text(encoding="utf-8"))
        except Exception as exc:
            summary = _empty_scan_error(preset_id, f"internal scan worker output JSON is invalid: {exc}")
            summary["worker_return_code"] = int(completed.returncode)
            summary["worker_elapsed_seconds"] = elapsed
            summary["worker_stdout_tail"] = _tail_text(completed.stdout)
            summary["worker_stderr_tail"] = _tail_text(completed.stderr)
            return summary

        if not isinstance(loaded, dict):
            summary = _empty_scan_error(preset_id, "internal scan worker output JSON is not an object")
            summary["worker_return_code"] = int(completed.returncode)
            summary["worker_elapsed_seconds"] = elapsed
            summary["worker_stdout_tail"] = _tail_text(completed.stdout)
            summary["worker_stderr_tail"] = _tail_text(completed.stderr)
            return summary

        loaded["worker_return_code"] = int(completed.returncode)
        loaded["worker_elapsed_seconds"] = elapsed
        if completed.returncode != 0:
            loaded["worker_failed"] = True
            loaded["worker_exit_error"] = f"internal scan worker exited with nonzero code ({completed.returncode})"
            stdout_tail = _tail_text(completed.stdout)
            stderr_tail = _tail_text(completed.stderr)
            if stdout_tail:
                loaded["worker_stdout_tail"] = stdout_tail
            if stderr_tail:
                loaded["worker_stderr_tail"] = stderr_tail
        return loaded


def _normalize_execution_mode(execution_mode: str) -> str:
    label = str(execution_mode or "").strip().upper()
    if label.startswith("IN_PROCESS"):
        return "in_process"
    return "subprocess"


def _run_internal_scan(
    *,
    root_folder: str,
    preset_id: str,
    overwrite_mode: str,
    scan_limit: int,
    write_scan_report: str,
    model_name: str,
    mmproj_name: str,
    timeout_seconds: int,
    n_ctx: int,
    n_gpu_layers: int,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
    threads: int,
    batch_size: int,
    keep_model_loaded: bool,
    unload_on_complete: bool,
    debug_mode: bool,
    node_runtime_lifecycle_mode: str,
    execution_mode: str,
) -> tuple[str, str]:
    normalized_execution_mode = _normalize_execution_mode(execution_mode)
    lifecycle_fields = {
        "internal_keep_model_loaded_requested": bool(keep_model_loaded),
        "internal_unload_on_complete_requested": bool(unload_on_complete),
        "node_runtime_lifecycle_mode": str(node_runtime_lifecycle_mode).strip() or "internal_unknown",
        "internal_execution_mode": normalized_execution_mode,
    }
    resolved_model_path, resolved_mmproj_path, resolve_error = resolve_model_and_mmproj_paths(
        model_name=model_name,
        mmproj_name=mmproj_name,
    )
    if resolve_error:
        summary = _empty_scan_error(preset_id, resolve_error)
        summary.update(lifecycle_fields)
        if write_scan_report == WRITE_SCAN_REPORT_ON:
            report_error = _write_scan_report(summary)
            if report_error:
                summary["report_write_error"] = report_error
        status_text = _build_internal_status_text(summary)
        report_error = str(summary.get("report_write_error", "")).strip()
        if report_error:
            status_text = f"{status_text} | Report: {report_error}"
        return json.dumps(summary, indent=2, ensure_ascii=False), status_text
    if resolved_model_path is None or resolved_mmproj_path is None:
        summary = _empty_scan_error(preset_id, "internal runtime model selection failed")
        summary.update(lifecycle_fields)
        return json.dumps(summary, indent=2, ensure_ascii=False), _build_internal_status_text(summary)

    request = _build_internal_scan_request(
        root_folder=root_folder,
        preset_id=preset_id,
        overwrite_mode=overwrite_mode,
        scan_limit=scan_limit,
        model_name=model_name,
        mmproj_name=mmproj_name,
        timeout_seconds=timeout_seconds,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        threads=threads,
        batch_size=batch_size,
        debug_mode=debug_mode,
        write_scan_report=write_scan_report,
        model_path_resolved=str(resolved_model_path),
        mmproj_path_resolved=str(resolved_mmproj_path),
    )
    if normalized_execution_mode == "subprocess":
        summary = _run_internal_scan_subprocess(request=request, timeout_seconds=timeout_seconds)
    else:
        summary = _run_internal_scan_in_process(
            request=request,
            keep_model_loaded=keep_model_loaded,
            unload_on_complete=unload_on_complete,
        )
    if not isinstance(summary, dict):
        summary = _empty_scan_error(preset_id, "internal scan returned invalid summary payload")
    summary.update(lifecycle_fields)

    if write_scan_report == WRITE_SCAN_REPORT_ON:
        report_error = _write_scan_report(summary)
        if report_error:
            summary["report_write_error"] = report_error

    status_text = _build_internal_status_text(summary)
    report_error = str(summary.get("report_write_error", "")).strip()
    if report_error:
        status_text = f"{status_text} | Report: {report_error}"

    return json.dumps(summary, indent=2, ensure_ascii=False), status_text


class GPMVLMScannerInternal:
    @classmethod
    def INPUT_TYPES(cls):
        model_choices = _model_choices()
        mmproj_choices = _mmproj_choices()
        default_mmproj = AUTO_MMPROJ_OPTION if AUTO_MMPROJ_OPTION in mmproj_choices else mmproj_choices[0]
        return {
            "required": {
                "root_folder": ("STRING", {"default": "", "multiline": False}),
                "preset_id": (_preset_choices(), {"default": "builtin-sdxl"}),
                "overwrite_mode": (
                    [OVERWRITE_SKIP_EXISTING, OVERWRITE_FAMILY],
                    {"default": OVERWRITE_SKIP_EXISTING},
                ),
                "scan_limit": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "write_scan_report": (WRITE_SCAN_REPORT_MODES, {"default": WRITE_SCAN_REPORT_OFF}),
                "model_name": (model_choices, {"default": model_choices[0]}),
                "mmproj_name": (mmproj_choices, {"default": default_mmproj}),
                "timeout_seconds": ("INT", {"default": 180, "min": 5, "max": 3600, "step": 1}),
                "debug_mode": (["OFF", "ON"], {"default": "OFF"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("summary_json", "status_text")
    FUNCTION = "scan"
    CATEGORY = "GPM"

    def scan(
        self,
        root_folder: str,
        preset_id: str,
        overwrite_mode: str,
        scan_limit: int,
        write_scan_report: str,
        model_name: str,
        mmproj_name: str,
        timeout_seconds: int,
        debug_mode: str,
        **_legacy_kwargs: Any,
    ):
        return _run_internal_scan(
            root_folder=root_folder,
            preset_id=preset_id,
            overwrite_mode=overwrite_mode,
            scan_limit=scan_limit,
            write_scan_report=write_scan_report,
            model_name=model_name,
            mmproj_name=mmproj_name,
            timeout_seconds=timeout_seconds,
            n_ctx=4096,
            n_gpu_layers=-1,
            temperature=None,
            top_p=None,
            max_tokens=None,
            threads=0,
            batch_size=512,
            keep_model_loaded=True,
            unload_on_complete=True,
            debug_mode=debug_mode == "ON",
            node_runtime_lifecycle_mode="basic_internal_fixed_defaults",
            execution_mode=EXECUTION_MODE_SUBPROCESS,
        )


class GPMVLMScannerInternalAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        model_choices = _model_choices()
        mmproj_choices = _mmproj_choices()
        default_mmproj = AUTO_MMPROJ_OPTION if AUTO_MMPROJ_OPTION in mmproj_choices else mmproj_choices[0]
        return {
            "required": {
                "root_folder": ("STRING", {"default": "", "multiline": False}),
                "preset_id": (_preset_choices(), {"default": "builtin-sdxl"}),
                "overwrite_mode": (
                    [OVERWRITE_SKIP_EXISTING, OVERWRITE_FAMILY],
                    {"default": OVERWRITE_SKIP_EXISTING},
                ),
                "scan_limit": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "write_scan_report": (WRITE_SCAN_REPORT_MODES, {"default": WRITE_SCAN_REPORT_OFF}),
                "model_name": (model_choices, {"default": model_choices[0]}),
                "mmproj_name": (mmproj_choices, {"default": default_mmproj}),
                "timeout_seconds": ("INT", {"default": 180, "min": 5, "max": 3600, "step": 1}),
                "n_ctx": ("INT", {"default": 4096, "min": 256, "max": 32768, "step": 256}),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 200, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 512, "min": 32, "max": 2048, "step": 1}),
                "threads": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
                "batch_size": ("INT", {"default": 512, "min": 32, "max": 8192, "step": 32}),
                "debug_mode": (["OFF", "ON"], {"default": "OFF"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("summary_json", "status_text")
    FUNCTION = "scan"
    CATEGORY = "GPM"

    def scan(
        self,
        root_folder: str,
        preset_id: str,
        overwrite_mode: str,
        scan_limit: int,
        write_scan_report: str,
        model_name: str,
        mmproj_name: str,
        timeout_seconds: int,
        n_ctx: int,
        n_gpu_layers: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        threads: int,
        batch_size: int,
        debug_mode: str,
        **_legacy_kwargs: Any,
    ):
        return _run_internal_scan(
            root_folder=root_folder,
            preset_id=preset_id,
            overwrite_mode=overwrite_mode,
            scan_limit=scan_limit,
            write_scan_report=write_scan_report,
            model_name=model_name,
            mmproj_name=mmproj_name,
            timeout_seconds=timeout_seconds,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            threads=threads,
            batch_size=batch_size,
            keep_model_loaded=True,
            unload_on_complete=True,
            debug_mode=debug_mode == "ON",
            node_runtime_lifecycle_mode="advanced_internal_subprocess_fixed_defaults",
            execution_mode=EXECUTION_MODE_SUBPROCESS,
        )
