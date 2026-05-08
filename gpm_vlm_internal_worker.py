from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

if __package__:
    from .gpm_vlm_backend import BACKEND_GGUF, scan_images_with_preset
    from .gpm_vlm_presets import GPMVLMPresetStore, get_preset_generation_settings
    from .gpm_vlm_runtime_base import RUNTIME_MODE_INTERNAL
else:
    _PACKAGE_DIR = Path(__file__).resolve().parent
    _WORKER_PACKAGE_NAME = "gpm_worker_pkg"

    if _WORKER_PACKAGE_NAME not in sys.modules:
        _pkg = types.ModuleType(_WORKER_PACKAGE_NAME)
        _pkg.__path__ = [str(_PACKAGE_DIR)]
        sys.modules[_WORKER_PACKAGE_NAME] = _pkg

    def _load_worker_module(module_basename: str):
        full_name = f"{_WORKER_PACKAGE_NAME}.{module_basename}"
        if full_name in sys.modules:
            return sys.modules[full_name]
        file_path = _PACKAGE_DIR / f"{module_basename}.py"
        spec = importlib.util.spec_from_file_location(full_name, str(file_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to load module: {module_basename}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
        return module

    _backend_mod = _load_worker_module("gpm_vlm_backend")
    _presets_mod = _load_worker_module("gpm_vlm_presets")
    _runtime_base_mod = _load_worker_module("gpm_vlm_runtime_base")
    BACKEND_GGUF = _backend_mod.BACKEND_GGUF
    scan_images_with_preset = _backend_mod.scan_images_with_preset
    GPMVLMPresetStore = _presets_mod.GPMVLMPresetStore
    get_preset_generation_settings = _presets_mod.get_preset_generation_settings
    RUNTIME_MODE_INTERNAL = _runtime_base_mod.RUNTIME_MODE_INTERNAL


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


def _run_worker_scan(request: dict[str, Any]) -> dict[str, Any]:
    preset_id = str(request.get("preset_id", "")).strip()
    try:
        store = GPMVLMPresetStore()
        preset = store.get_preset(preset_id)
        if preset is None:
            return _empty_scan_error(preset_id, f"preset not found: {preset_id}")
        model_name = str(request.get("model_name", "")).strip()
        mmproj_name = str(request.get("mmproj_name", "")).strip()
        model_path_resolved = str(request.get("model_path_resolved", "")).strip()
        mmproj_path_resolved = str(request.get("mmproj_path_resolved", "")).strip()
        if not model_name:
            return _empty_scan_error(preset_id, "model_name is required for internal GGUF mode")
        if not mmproj_name:
            return _empty_scan_error(
                preset_id,
                "mmproj_name is required for internal GGUF mode. Use auto-pair or select manually.",
            )
        if model_path_resolved and not os.path.isfile(model_path_resolved):
            return _empty_scan_error(
                preset_id,
                f"worker resolved model path was not found: {model_path_resolved}",
            )
        if mmproj_path_resolved and not os.path.isfile(mmproj_path_resolved):
            return _empty_scan_error(
                preset_id,
                f"worker resolved mmproj path was not found: {mmproj_path_resolved}",
            )

        preset_temperature, preset_top_p, preset_max_tokens = get_preset_generation_settings(preset)
        raw_temperature = request.get("temperature", None)
        raw_top_p = request.get("top_p", None)
        raw_max_tokens = request.get("max_tokens", None)
        effective_temperature = preset_temperature if raw_temperature is None else float(raw_temperature)
        effective_top_p = preset_top_p if raw_top_p is None else float(raw_top_p)
        effective_max_tokens = preset_max_tokens if raw_max_tokens is None else int(raw_max_tokens)

        return scan_images_with_preset(
            root_folder=str(request.get("root_folder", "")),
            preset=preset,
            overwrite_mode=str(request.get("overwrite_mode", "SKIP_EXISTING")),
            backend_mode=BACKEND_GGUF,
            gguf_model_name="",
            timeout_seconds=int(request.get("timeout_seconds", 180)),
            scan_limit=int(request.get("scan_limit", 0)),
            runtime_mode=RUNTIME_MODE_INTERNAL,
            internal_model_name=model_name,
            internal_mmproj_name=mmproj_name,
            internal_model_path_override=model_path_resolved,
            internal_mmproj_path_override=mmproj_path_resolved,
            internal_n_ctx=int(request.get("n_ctx", 4096)),
            internal_n_gpu_layers=int(request.get("n_gpu_layers", -1)),
            internal_temperature=effective_temperature,
            internal_top_p=effective_top_p,
            internal_max_tokens=effective_max_tokens,
            internal_threads=int(request.get("threads", 0)),
            internal_batch_size=int(request.get("batch_size", 512)),
            internal_keep_model_loaded=True,
            internal_unload_on_complete=True,
            internal_debug_mode=bool(request.get("debug_mode", False)),
        )
    except Exception as exc:
        return _empty_scan_error(preset_id, f"internal scan execution failed: {exc}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GPM internal VLM worker")
    parser.add_argument("--request-json", required=True, help="Path to request JSON file")
    parser.add_argument("--output-json", required=True, help="Path to output summary JSON file")
    args = parser.parse_args(argv)

    request_path = Path(args.request_json).resolve()
    output_path = Path(args.output_json).resolve()
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
    except Exception as exc:
        summary = _empty_scan_error("", f"failed to read request JSON: {exc}")
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return 2

    if not isinstance(request, dict):
        summary = _empty_scan_error("", "request JSON must be an object")
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return 2

    summary = _run_worker_scan(request)
    if not isinstance(summary, dict):
        summary = _empty_scan_error(str(request.get("preset_id", "")), "worker scan returned non-object summary")
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if bool(summary.get("ok", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
