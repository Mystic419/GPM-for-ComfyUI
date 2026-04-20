from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .gpm_vlm_model_discovery import resolve_model_and_mmproj_paths
from .gpm_vlm_runtime_api import GPMGGUFAPIRuntime
from .gpm_vlm_runtime_base import (
    RECOMMENDED_GGUF_MODEL_REPO,
    RUNTIME_MODE_API,
    RUNTIME_MODE_INTERNAL,
    SUPPORTED_RUNTIME_MODES,
    GPMVLMRuntime,
)
from .gpm_vlm_runtime_internal import GPMGGUFInternalRuntime, GPMInternalRuntimeConfig

FAMILY_TO_JSON_FIELDS: dict[str, tuple[str, str]] = {
    "SDXL": ("sdxl_person", "sdxl_scene"),
    "Pony": ("pony_person", "pony_scene"),
    "Natural Language": ("natural_person", "natural_scene"),
}

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

OVERWRITE_SKIP_EXISTING = "SKIP_EXISTING"
OVERWRITE_FAMILY = "OVERWRITE_FAMILY"
OVERWRITE_MODES = {OVERWRITE_SKIP_EXISTING, OVERWRITE_FAMILY}

BACKEND_GGUF = "GGUF"


def _empty_summary(error: str) -> dict[str, Any]:
    return {
        "ok": False,
        "error": error,
        "total_found": 0,
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "skipped_files": [],
        "failures": [],
    }


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _is_execution_interrupted() -> bool:
    try:
        import comfy.model_management as model_management  # type: ignore
    except Exception:
        return False

    check_fn = getattr(model_management, "throw_exception_if_processing_interrupted", None)
    if not callable(check_fn):
        return False

    try:
        check_fn()
        return False
    except Exception as exc:
        if exc.__class__.__name__ == "InterruptProcessingException":
            return True
        return False


def _normalize_root_folder(root_folder: str) -> Path | None:
    if not root_folder or not str(root_folder).strip():
        return None
    root = Path(root_folder).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return None
    return root


def _discover_images(root: Path) -> list[Path]:
    images: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        images.append(path)
    images.sort(key=lambda item: str(item).casefold())
    return images


def _read_sidecar_payload(json_path: Path) -> tuple[dict[str, Any], bool, str]:
    if not json_path.exists() or not json_path.is_file():
        return {}, False, ""

    try:
        with json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return {}, True, f"invalid sidecar JSON: {exc}"

    if not isinstance(payload, dict):
        return {}, True, "invalid sidecar JSON: root must be an object"
    return payload, True, ""


def _family_slot_has_prompt_data(payload: dict[str, Any], family: str) -> bool:
    person_key, scene_key = FAMILY_TO_JSON_FIELDS[family]
    person_text = payload.get(person_key)
    scene_text = payload.get(scene_key)
    return (isinstance(person_text, str) and person_text.strip() != "") or (
        isinstance(scene_text, str) and scene_text.strip() != ""
    )


def _write_sidecar_payload(json_path: Path, payload: dict[str, Any]) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _scan_failure_record(path: Path, error_message: str) -> dict[str, str]:
    return {"path": str(path), "error": error_message}


def _scan_skipped_record(path: Path, reason: str) -> dict[str, str]:
    return {"path": str(path), "reason": reason}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _looks_like_generic_family_living_room_response(raw_response: str) -> bool:
    text = str(raw_response or "").casefold()
    if not text:
        return False
    generic_markers = (
        "family of four",
        "living room",
        "warm living room",
        "cozy living room",
        "portrait of a family",
        "smiling family",
    )
    return any(marker in text for marker in generic_markers)


def _looks_like_outdoor_or_storefront_file_hint(image_path: Path) -> bool:
    hint_text = str(image_path).casefold()
    hint_markers = (
        "store",
        "storefront",
        "shop",
        "cafe",
        "coffee",
        "street",
        "outside",
        "outdoor",
        "building",
        "entrance",
        "city",
    )
    return any(marker in hint_text for marker in hint_markers)


def _is_likely_cross_image_mismatch(raw_response: str, image_path: Path) -> bool:
    return _looks_like_generic_family_living_room_response(raw_response) and _looks_like_outdoor_or_storefront_file_hint(
        image_path
    )


def _build_debug_trace_payload(
    *,
    runtime_trace: dict[str, Any],
    image_path: Path,
    json_path: Path,
    person_prompt: str,
    scene_prompt: str,
) -> dict[str, Any]:
    source_hash = str(runtime_trace.get("source_image_sha256", "")).strip()
    if not source_hash:
        try:
            source_hash = _sha256_file(image_path)
        except Exception:
            source_hash = ""

    return {
        "source_image_filename": str(runtime_trace.get("source_image_filename", "")).strip() or image_path.name,
        "source_image_full_path": str(runtime_trace.get("source_image_full_path", "")).strip() or str(image_path),
        "source_image_sha256": source_hash,
        "output_json_full_path": str(json_path),
        "model_prompt_sent": str(runtime_trace.get("model_prompt_sent", "")).strip(),
        "raw_model_response": str(runtime_trace.get("raw_model_response", "")),
        "parsed_person_prompt": str(runtime_trace.get("parsed_person_prompt", "")).strip() or person_prompt,
        "parsed_scene_prompt": str(runtime_trace.get("parsed_scene_prompt", "")).strip() or scene_prompt,
        "detected_model_family": str(runtime_trace.get("detected_model_family", "")).strip(),
        "selected_chat_handler": str(runtime_trace.get("selected_chat_handler", "")).strip(),
        "family_support_status": str(runtime_trace.get("family_support_status", "")).strip(),
        "support_reason": str(runtime_trace.get("support_reason", "")).strip(),
    }


def _build_sidecar_vlm_scan_meta(
    *,
    family: str,
    preset_id: str,
    backend: str,
    runtime: str,
    model: str,
    status: str,
) -> dict[str, str]:
    return {
        "family": family,
        "preset_id": preset_id,
        "backend": backend,
        "runtime": runtime,
        "model": model,
        "status": status,
        "scanned_at": _now_iso_utc(),
    }


def _build_sidecar_vlm_scan_debug_meta(runtime_summary_metadata: dict[str, Any]) -> dict[str, Any]:
    debug_meta = dict(runtime_summary_metadata)
    # Keep debug payload verbose but avoid duplicate aliases for the same values.
    debug_meta.pop("internal_multimodal_family", None)
    debug_meta.pop("internal_chat_handler", None)
    debug_meta.pop("resolved_model_path", None)
    debug_meta.pop("resolved_mmproj_path", None)
    return debug_meta


def _build_runtime(
    runtime_mode: str,
    gguf_api_url: str,
    gguf_model_name: str,
    timeout_seconds: int,
    internal_model_name: str = "",
    internal_mmproj_name: str = "",
    internal_n_ctx: int = 4096,
    internal_n_gpu_layers: int = -1,
    internal_temperature: float = 0.2,
    internal_top_p: float = 0.95,
    internal_max_tokens: int = 512,
    internal_threads: int = 0,
    internal_batch_size: int = 512,
    internal_keep_model_loaded: bool = False,
    internal_debug_mode: bool = False,
) -> tuple[GPMVLMRuntime | None, str]:
    if runtime_mode == RUNTIME_MODE_API:
        return (
            GPMGGUFAPIRuntime(
                api_url=gguf_api_url,
                model_name=gguf_model_name,
                timeout_seconds=timeout_seconds,
            ),
            "",
        )

    if runtime_mode == RUNTIME_MODE_INTERNAL:
        model_path, mmproj_path, resolve_error = resolve_model_and_mmproj_paths(
            model_name=internal_model_name,
            mmproj_name=internal_mmproj_name,
        )
        if resolve_error:
            return None, resolve_error
        if model_path is None or mmproj_path is None:
            return None, "internal runtime model selection failed"

        runtime_config = GPMInternalRuntimeConfig(
            model_path=str(model_path),
            mmproj_path=str(mmproj_path),
            n_ctx=max(256, int(internal_n_ctx)),
            n_gpu_layers=int(internal_n_gpu_layers),
            temperature=float(internal_temperature),
            top_p=float(internal_top_p),
            max_tokens=max(32, int(internal_max_tokens)),
            threads=int(internal_threads),
            batch_size=max(32, int(internal_batch_size)),
            keep_model_loaded=bool(internal_keep_model_loaded),
            debug_mode=bool(internal_debug_mode),
        )
        return (
            GPMGGUFInternalRuntime(
                model_name=str(internal_model_name).strip() or model_path.name,
                mmproj_name=mmproj_path.name,
                timeout_seconds=timeout_seconds,
                config=runtime_config,
            ),
            "",
        )
    return None, f"unsupported runtime mode: {runtime_mode}"


def scan_images_with_preset(
    root_folder: str,
    preset: dict[str, Any],
    overwrite_mode: str = OVERWRITE_SKIP_EXISTING,
    backend_mode: str = BACKEND_GGUF,
    gguf_api_url: str = "http://127.0.0.1:1234/v1/chat/completions",
    gguf_model_name: str = "",
    timeout_seconds: int = 180,
    scan_limit: int = 0,
    runtime_mode: str = RUNTIME_MODE_API,
    internal_model_name: str = "",
    internal_mmproj_name: str = "",
    internal_n_ctx: int = 4096,
    internal_n_gpu_layers: int = -1,
    internal_temperature: float = 0.2,
    internal_top_p: float = 0.95,
    internal_max_tokens: int = 512,
    internal_threads: int = 0,
    internal_batch_size: int = 512,
    internal_keep_model_loaded: bool = False,
    internal_debug_mode: bool = False,
) -> dict[str, Any]:
    family = str(preset.get("family", "")).strip()
    if family not in FAMILY_TO_JSON_FIELDS:
        return _empty_summary(f"unsupported preset family: {family}")

    normalized_overwrite_mode = overwrite_mode if overwrite_mode in OVERWRITE_MODES else OVERWRITE_SKIP_EXISTING
    if backend_mode != BACKEND_GGUF:
        return _empty_summary(
            f"unsupported backend mode: {backend_mode}. Only GGUF is supported in this phase."
        )

    normalized_runtime_mode = str(runtime_mode).strip().lower() or RUNTIME_MODE_API
    if normalized_runtime_mode not in SUPPORTED_RUNTIME_MODES:
        return _empty_summary(f"unsupported runtime mode: {runtime_mode}")

    normalized_api_model_name = str(gguf_model_name).strip()
    normalized_internal_model_name = str(internal_model_name).strip()
    normalized_internal_mmproj_name = str(internal_mmproj_name).strip()

    if normalized_runtime_mode == RUNTIME_MODE_API and not normalized_api_model_name:
        return _empty_summary("gguf_model_name is required for api runtime mode")
    if normalized_runtime_mode == RUNTIME_MODE_INTERNAL and not normalized_internal_model_name:
        return _empty_summary("internal_model_name is required for internal mode")
    if normalized_runtime_mode == RUNTIME_MODE_INTERNAL and not normalized_internal_mmproj_name:
        return _empty_summary("mmproj_name is required for internal mode")

    requested_model_name = (
        normalized_api_model_name
        if normalized_runtime_mode == RUNTIME_MODE_API
        else normalized_internal_model_name
    )

    root = _normalize_root_folder(root_folder)
    if root is None:
        return _empty_summary("invalid root folder")

    images = _discover_images(root)
    if scan_limit > 0:
        images = images[: int(scan_limit)]

    runtime, runtime_error = _build_runtime(
        runtime_mode=normalized_runtime_mode,
        gguf_api_url=gguf_api_url,
        gguf_model_name=normalized_api_model_name,
        timeout_seconds=timeout_seconds,
        internal_model_name=normalized_internal_model_name,
        internal_mmproj_name=normalized_internal_mmproj_name,
        internal_n_ctx=internal_n_ctx,
        internal_n_gpu_layers=internal_n_gpu_layers,
        internal_temperature=internal_temperature,
        internal_top_p=internal_top_p,
        internal_max_tokens=internal_max_tokens,
        internal_threads=internal_threads,
        internal_batch_size=internal_batch_size,
        internal_keep_model_loaded=internal_keep_model_loaded,
        internal_debug_mode=internal_debug_mode,
    )
    if runtime is None:
        return _empty_summary(runtime_error or "runtime initialization failed")

    started_ok, started_error = runtime.start()
    if not started_ok:
        summary = _empty_summary(started_error or "runtime startup failed")
        if normalized_runtime_mode == RUNTIME_MODE_INTERNAL and bool(internal_debug_mode):
            debug_fn = getattr(runtime, "startup_debug_metadata", None)
            if callable(debug_fn):
                try:
                    raw_debug = debug_fn()
                    if isinstance(raw_debug, dict):
                        summary["internal_startup_debug"] = dict(raw_debug)
                        for key in (
                            "llama_cpp_version",
                            "llama_cpp_import_ok",
                            "llama_cpp_import_error",
                            "backend_probe",
                            "backend_probe_method",
                            "backend_probe_error",
                            "backend_status",
                            "gpu_offload_requested",
                            "gpu_offload_requested_value",
                            "runtime_requested_n_ctx",
                            "runtime_requested_n_batch",
                            "runtime_requested_threads",
                            "llama_constructor_dropped_kwargs",
                            "gpu_offload_hint",
                            "inferred_family",
                            "selected_handler",
                            "detected_model_family",
                            "selected_chat_handler",
                            "family_support_status",
                            "support_reason",
                            "scan_family_approved",
                            "available_internal_families",
                            "approved_internal_scan_families",
                            "resolved_model_path",
                            "resolved_mmproj_path",
                            "model_file_size_bytes",
                            "mmproj_file_size_bytes",
                            "constructor_exception",
                            "keep_model_loaded",
                            "handler_first_attempted",
                            "handler_constructor_kwargs",
                            "llama_constructor_kwargs",
                            "llama_constructor_kwargs_raw",
                            "llama_constructor_kwargs_filtered",
                            "family_resolution_mode",
                            "family_resolved_via_explicit_class_match",
                            "family_resolved_via_fallback_class_preference_order",
                        ):
                            if key in raw_debug:
                                summary[key] = raw_debug.get(key)
                except Exception:
                    pass
        return summary

    runtime_summary_metadata: dict[str, Any] = {}
    runtime_metadata_fn = getattr(runtime, "summary_metadata", None)
    if callable(runtime_metadata_fn):
        try:
            meta = runtime_metadata_fn()
            if isinstance(meta, dict):
                runtime_summary_metadata = dict(meta)
        except Exception:
            runtime_summary_metadata = {}

    effective_model_name = str(runtime_summary_metadata.get("model_name", "")).strip() or requested_model_name

    person_key, scene_key = FAMILY_TO_JSON_FIELDS[family]
    skipped_files: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    processed = 0
    skipped = 0
    failed = 0
    warnings: list[dict[str, str]] = []

    try:
        for image_path in images:
            if _is_execution_interrupted():
                summary = {
                    "ok": True,
                    "error": "",
                    "root_folder": str(root),
                    "preset_id": str(preset.get("id", "")),
                    "preset_name": str(preset.get("name", "")),
                    "family": family,
                    "overwrite_mode": normalized_overwrite_mode,
                    "backend": BACKEND_GGUF,
                    "runtime_mode": normalized_runtime_mode,
                    "model_name": effective_model_name,
                    "recommended_model_repo": RECOMMENDED_GGUF_MODEL_REPO,
                    "total_found": len(images),
                    "processed": processed,
                    "skipped": skipped,
                    "failed": failed,
                    "skipped_files": skipped_files,
                    "failures": failures,
                    "warnings": warnings,
                    "stopped": True,
                    "stopped_reason": "execution interrupted",
                }
                if runtime_summary_metadata:
                    summary.update(runtime_summary_metadata)
                return summary

            json_path = image_path.with_suffix(".json")
            existing_payload, sidecar_existed, read_error = _read_sidecar_payload(json_path)
            if read_error:
                failed += 1
                failures.append(_scan_failure_record(image_path, read_error))
                continue

            if normalized_overwrite_mode == OVERWRITE_SKIP_EXISTING and _family_slot_has_prompt_data(
                existing_payload, family
            ):
                skipped += 1
                skipped_files.append(_scan_skipped_record(image_path, "family slot already populated"))
                continue

            # Explicitly scope and reset per-image request state.
            runtime_trace: dict[str, Any] = {}
            consume_trace_fn = getattr(runtime, "consume_last_scan_debug_trace", None)
            if callable(consume_trace_fn):
                try:
                    consume_trace_fn()
                except Exception:
                    pass

            person_prompt, scene_prompt, backend_error = runtime.generate(image_path, preset)
            if callable(consume_trace_fn):
                try:
                    consumed_trace = consume_trace_fn()
                    if isinstance(consumed_trace, dict):
                        runtime_trace = dict(consumed_trace)
                except Exception:
                    runtime_trace = {}
            if backend_error:
                failed += 1
                failures.append(_scan_failure_record(image_path, backend_error))
                continue

            debug_mode_enabled = bool(runtime_summary_metadata.get("debug_mode", False))
            raw_response_text = str(runtime_trace.get("raw_model_response", ""))
            mismatch_warning = ""
            if normalized_runtime_mode == RUNTIME_MODE_INTERNAL and debug_mode_enabled:
                if runtime_trace:
                    traced_image_path = str(runtime_trace.get("source_image_full_path", "")).strip()
                    if traced_image_path:
                        try:
                            traced_norm = str(Path(traced_image_path).resolve()).casefold()
                            current_norm = str(image_path.resolve()).casefold()
                            if traced_norm != current_norm:
                                mismatch_warning = (
                                    "runtime debug trace image path mismatch; skipped overwrite to avoid stale cross-image data"
                                )
                        except Exception:
                            mismatch_warning = (
                                "runtime debug trace image path mismatch; skipped overwrite to avoid stale cross-image data"
                            )
                if not mismatch_warning and _is_likely_cross_image_mismatch(raw_response_text, image_path):
                    mismatch_warning = (
                        "raw response looked like generic family/living-room text for a likely non-family image hint"
                    )
                if mismatch_warning:
                    warning_record = {"path": str(image_path), "warning": mismatch_warning}
                    warnings.append(warning_record)
                    if sidecar_existed:
                        skipped += 1
                        skipped_files.append(
                            _scan_skipped_record(image_path, f"debug guard: {mismatch_warning}")
                        )
                        continue

            payload = dict(existing_payload)
            payload[person_key] = person_prompt
            payload[scene_key] = scene_prompt

            meta_raw = payload.get("gpm_meta")
            gpm_meta = dict(meta_raw) if isinstance(meta_raw, dict) else {}
            runtime_status = str(runtime_summary_metadata.get("backend_status", "")).strip() or "ok"
            gpm_meta["vlm_scan"] = _build_sidecar_vlm_scan_meta(
                family=family,
                preset_id=str(preset.get("id", "")),
                backend=BACKEND_GGUF,
                runtime=normalized_runtime_mode,
                model=effective_model_name,
                status=runtime_status,
            )
            if normalized_runtime_mode == RUNTIME_MODE_INTERNAL and debug_mode_enabled and runtime_summary_metadata:
                gpm_meta["vlm_scan_debug"] = _build_sidecar_vlm_scan_debug_meta(runtime_summary_metadata)
                gpm_meta["vlm_scan_debug_trace"] = _build_debug_trace_payload(
                    runtime_trace=runtime_trace,
                    image_path=image_path,
                    json_path=json_path,
                    person_prompt=person_prompt,
                    scene_prompt=scene_prompt,
                )
                if mismatch_warning:
                    gpm_meta["vlm_scan_debug_trace"]["warning"] = mismatch_warning
            else:
                gpm_meta.pop("vlm_scan_debug", None)
                gpm_meta.pop("vlm_scan_debug_trace", None)
            payload["gpm_meta"] = gpm_meta

            try:
                _write_sidecar_payload(json_path, payload)
            except OSError as exc:
                failed += 1
                failures.append(_scan_failure_record(image_path, f"write failed: {exc}"))
                continue

            processed += 1
    finally:
        runtime.stop()

    summary = {
        "ok": True,
        "error": "",
        "root_folder": str(root),
        "preset_id": str(preset.get("id", "")),
        "preset_name": str(preset.get("name", "")),
        "family": family,
        "overwrite_mode": normalized_overwrite_mode,
        "backend": BACKEND_GGUF,
        "runtime_mode": normalized_runtime_mode,
        "model_name": effective_model_name,
        "recommended_model_repo": RECOMMENDED_GGUF_MODEL_REPO,
        "total_found": len(images),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "skipped_files": skipped_files,
        "failures": failures,
        "warnings": warnings,
    }
    if runtime_summary_metadata:
        summary.update(runtime_summary_metadata)
    return summary
