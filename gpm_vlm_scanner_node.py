from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .gpm_vlm_backend import (
    BACKEND_GGUF,
    OVERWRITE_FAMILY,
    OVERWRITE_SKIP_EXISTING,
    scan_images_with_preset,
)
from .gpm_vlm_presets import GPMVLMPresetStore

WRITE_SCAN_REPORT_OFF = "OFF"
WRITE_SCAN_REPORT_ON = "ON"
WRITE_SCAN_REPORT_MODES = [WRITE_SCAN_REPORT_OFF, WRITE_SCAN_REPORT_ON]
SCAN_REPORT_FILENAME = "gpm_scan_report.txt"


def _preset_choices() -> list[str]:
    # Preset dropdown values are resolved from current preset storage at INPUT_TYPES time.
    # New presets may require a ComfyUI refresh/reload before appearing in this dropdown.
    # This is acceptable for the current phase until preset CRUD/UI wiring is added.
    store = GPMVLMPresetStore()
    presets = store.list_presets()
    ids = [str(item.get("id", "")).strip() for item in presets if isinstance(item, dict)]
    cleaned = [item for item in ids if item]
    return cleaned or ["builtin-sdxl", "builtin-pony", "builtin-natural-language"]


def _build_status_text(summary: dict[str, Any]) -> str:
    total_found = int(summary.get("total_found", 0))
    processed = int(summary.get("processed", 0))
    skipped = int(summary.get("skipped", 0))
    failed = int(summary.get("failed", 0))
    counts_text = f"Found {total_found} | Processed {processed} | Skipped {skipped} | Failed {failed}"

    error_text = str(summary.get("error", "")).strip()
    if error_text:
        return f"Error: {error_text}"

    if bool(summary.get("stopped", False)):
        return f"Stopped | {counts_text}"

    return counts_text


def _write_scan_report(summary: dict[str, Any]) -> str:
    root_folder = str(summary.get("root_folder", "")).strip()
    if not root_folder:
        return "scan report was not written: root folder is unavailable"

    root_path = Path(root_folder)
    if not root_path.exists() or not root_path.is_dir():
        return "scan report was not written: root folder is unavailable"

    report_path = root_path / SCAN_REPORT_FILENAME
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    skipped_files = summary.get("skipped_files", [])
    if not isinstance(skipped_files, list):
        skipped_files = []
    failures = summary.get("failures", [])
    if not isinstance(failures, list):
        failures = []

    lines: list[str] = [
        "GPM Scan Report",
        f"Timestamp: {timestamp}",
        f"Root Folder: {root_folder}",
        f"Preset ID: {summary.get('preset_id', '')}",
        f"Preset Name: {summary.get('preset_name', '')}",
        f"Family: {summary.get('family', '')}",
        f"Overwrite Mode: {summary.get('overwrite_mode', '')}",
        f"Backend: {summary.get('backend', '')}",
        f"Model Name: {summary.get('model_name', '')}",
        f"Total Found: {summary.get('total_found', 0)}",
        f"Processed: {summary.get('processed', 0)}",
        f"Skipped: {summary.get('skipped', 0)}",
        f"Failed: {summary.get('failed', 0)}",
    ]

    if bool(summary.get("stopped", False)):
        lines.append("Stopped: true")
        lines.append(f"Stopped Reason: {summary.get('stopped_reason', '')}")
    else:
        lines.append("Stopped: false")

    lines.append("")
    lines.append("Skipped Files")
    if skipped_files:
        for item in skipped_files:
            if not isinstance(item, dict):
                continue
            lines.append(f"- {item.get('path', '')} | {item.get('reason', '')}")
    else:
        lines.append("- none")

    lines.append("")
    lines.append("Failed Files")
    if failures:
        for item in failures:
            if not isinstance(item, dict):
                continue
            lines.append(f"- {item.get('path', '')} | {item.get('error', '')}")
    else:
        lines.append("- none")

    try:
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as exc:
        return f"scan report write failed: {exc}"

    return ""


class GPMVLMScanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "root_folder": ("STRING", {"default": "", "multiline": False}),
                "preset_id": (_preset_choices(), {"default": "builtin-sdxl"}),
                "overwrite_mode": (
                    [OVERWRITE_SKIP_EXISTING, OVERWRITE_FAMILY],
                    {"default": OVERWRITE_SKIP_EXISTING},
                ),
                "gguf_api_url": (
                    "STRING",
                    {
                        "default": "http://127.0.0.1:1234/v1/chat/completions",
                        "multiline": False,
                    },
                ),
                "gguf_model_name": ("STRING", {"default": "", "multiline": False}),
                "timeout_seconds": ("INT", {"default": 180, "min": 5, "max": 3600, "step": 1}),
                "scan_limit": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "write_scan_report": (WRITE_SCAN_REPORT_MODES, {"default": WRITE_SCAN_REPORT_OFF}),
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
        gguf_api_url: str,
        gguf_model_name: str,
        timeout_seconds: int,
        scan_limit: int,
        write_scan_report: str,
    ):
        store = GPMVLMPresetStore()
        preset = store.get_preset(preset_id)
        if preset is None:
            summary: dict[str, Any] = {
                "ok": False,
                "error": f"preset not found: {preset_id}",
                "total_found": 0,
                "processed": 0,
                "skipped": 0,
                "failed": 0,
                "failures": [],
            }
        elif not str(gguf_model_name).strip():
            summary = {
                "ok": False,
                "error": "gguf_model_name is required for GGUF backend mode",
                "total_found": 0,
                "processed": 0,
                "skipped": 0,
                "failed": 0,
                "failures": [],
            }
        else:
            summary = scan_images_with_preset(
                root_folder=root_folder,
                preset=preset,
                overwrite_mode=overwrite_mode,
                backend_mode=BACKEND_GGUF,
                gguf_api_url=gguf_api_url,
                gguf_model_name=gguf_model_name,
                timeout_seconds=timeout_seconds,
                scan_limit=scan_limit,
            )

        if write_scan_report == WRITE_SCAN_REPORT_ON:
            report_error = _write_scan_report(summary)
            if report_error:
                summary["report_write_error"] = report_error

        status_text = _build_status_text(summary)
        report_error = str(summary.get("report_write_error", "")).strip()
        if report_error:
            status_text = f"{status_text} | Report: {report_error}"

        return (
            json.dumps(summary, indent=2, ensure_ascii=False),
            status_text,
        )
