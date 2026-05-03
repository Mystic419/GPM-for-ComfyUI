from __future__ import annotations

import json
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
    debug_mode: bool,
) -> tuple[str, str]:
    store = GPMVLMPresetStore()
    preset = store.get_preset(preset_id)
    if preset is None:
        summary = _empty_scan_error(preset_id, f"preset not found: {preset_id}")
    elif not str(model_name).strip():
        summary = _empty_scan_error(preset_id, "model_name is required for internal GGUF mode")
    elif not str(mmproj_name).strip():
        summary = _empty_scan_error(
            preset_id,
            "mmproj_name is required for internal GGUF mode. Use auto-pair or select manually.",
        )
    else:
        preset_temperature, preset_top_p, preset_max_tokens = get_preset_generation_settings(preset)
        effective_temperature = preset_temperature if temperature is None else float(temperature)
        effective_top_p = preset_top_p if top_p is None else float(top_p)
        effective_max_tokens = preset_max_tokens if max_tokens is None else int(max_tokens)
        summary = scan_images_with_preset(
            root_folder=root_folder,
            preset=preset,
            overwrite_mode=overwrite_mode,
            backend_mode=BACKEND_GGUF,
            gguf_model_name="",
            timeout_seconds=timeout_seconds,
            scan_limit=scan_limit,
            runtime_mode=RUNTIME_MODE_INTERNAL,
            internal_model_name=model_name,
            internal_mmproj_name=mmproj_name,
            internal_n_ctx=n_ctx,
            internal_n_gpu_layers=n_gpu_layers,
            internal_temperature=effective_temperature,
            internal_top_p=effective_top_p,
            internal_max_tokens=effective_max_tokens,
            internal_threads=threads,
            internal_batch_size=batch_size,
            internal_keep_model_loaded=keep_model_loaded,
            internal_debug_mode=debug_mode,
        )

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
            keep_model_loaded=False,
            debug_mode=debug_mode == "ON",
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
                "keep_model_loaded": (["OFF", "ON"], {"default": "OFF"}),
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
        keep_model_loaded: str,
        debug_mode: str,
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
            keep_model_loaded=keep_model_loaded == "ON",
            debug_mode=debug_mode == "ON",
        )
