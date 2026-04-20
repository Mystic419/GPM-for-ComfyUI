from __future__ import annotations

import json
import platform as py_platform
import sys
from pathlib import Path
from typing import Any
import inspect

from .gpm_vlm_internal_multimodal import (
    FAMILY_QWEN_VL,
    detect_chat_handler_support,
    infer_internal_multimodal_family,
    resolve_internal_chat_handler,
)
from .gpm_vlm_model_discovery import (
    discover_gguf_model_choices,
    discover_mmproj_choices,
    resolve_model_and_mmproj_paths,
)
from .gpm_vlm_runtime_internal import _gpu_offload_requested, _json_safe_debug_value, _probe_llama_cpp_backend, _summarize_backend_status


def _model_choices() -> list[str]:
    return discover_gguf_model_choices()


def _mmproj_choices() -> list[str]:
    return discover_mmproj_choices()


def _family_label(family: str) -> str:
    if family == "qwen_vl":
        return "Qwen/Gliese"
    if family == "llava":
        return "LLaVA"
    return family or "Unknown"


def _inspect_llama_chat_format() -> tuple[bool, str, list[str], list[str], str]:
    try:
        import llama_cpp  # type: ignore
    except Exception as exc:
        return False, "", [], [], str(exc)

    version = str(getattr(llama_cpp, "__version__", "") or "")
    try:
        from llama_cpp import llama_chat_format  # type: ignore
    except Exception as exc:
        return True, version, [], [], str(exc)

    all_attr_names = sorted(dir(llama_chat_format))
    relevant_attrs = [
        name
        for name in all_attr_names
        if (
            "ChatHandler" in name
            and any(token in name for token in ("Llava", "Qwen", "VL", "Vision", "Multi"))
        )
    ]

    support = detect_chat_handler_support()
    supported_classes: list[str] = []
    for classes in support.family_to_classes.values():
        supported_classes.extend(classes)
    supported_classes = sorted(set(supported_classes), key=str.casefold)

    return True, version, relevant_attrs, supported_classes, ""


def _filter_kwargs_for_callable(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(fn)
    except Exception:
        return dict(kwargs)

    params = list(signature.parameters.values())
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params):
        return dict(kwargs)

    allowed: set[str] = set()
    for param in params:
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            allowed.add(param.name)
    return {key: value for key, value in kwargs.items() if key in allowed}


class GPMVLMInternalDiagnostics:
    @classmethod
    def INPUT_TYPES(cls):
        model_choices = _model_choices()
        mmproj_choices = _mmproj_choices()
        return {
            "required": {
                "model_name": (model_choices, {"default": model_choices[0]}),
                "mmproj_name": (mmproj_choices, {"default": mmproj_choices[0]}),
                "n_ctx": ("INT", {"default": 4096, "min": 256, "max": 32768, "step": 256}),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 200, "step": 1}),
                "n_batch": ("INT", {"default": 512, "min": 32, "max": 8192, "step": 32}),
                "threads": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("summary_json", "status_text")
    FUNCTION = "run"
    CATEGORY = "GPM"

    def run(
        self,
        model_name: str,
        mmproj_name: str,
        n_ctx: int,
        n_gpu_layers: int,
        n_batch: int,
        threads: int,
    ):
        support = detect_chat_handler_support()
        import_ok, llama_version, relevant_attrs, supported_classes, import_error = _inspect_llama_chat_format()
        backend_probe_payload = _probe_llama_cpp_backend()
        available_families = list(support.available_families()) if not support.import_error else []

        model_path, mmproj_path, resolve_error = resolve_model_and_mmproj_paths(
            model_name=str(model_name).strip(),
            mmproj_name=str(mmproj_name).strip(),
        )

        resolved_model_path = str(model_path.resolve()) if model_path is not None else ""
        resolved_mmproj_path = str(mmproj_path.resolve()) if mmproj_path is not None else ""
        model_exists = bool(model_path and model_path.exists() and model_path.is_file())
        mmproj_exists = bool(mmproj_path and mmproj_path.exists() and mmproj_path.is_file())

        infer_source = resolved_model_path or str(model_name).strip()
        selected_family, family_infer_error = infer_internal_multimodal_family(
            infer_source,
            mmproj_path=resolved_mmproj_path,
        )
        family_supported = bool(selected_family and selected_family in support.family_to_classes)
        selected_handler = ""
        selected_handler_error = ""
        handler_debug: dict[str, Any] = {}
        handler_obj: Any = None
        if (
            not family_infer_error
            and family_supported
            and selected_family
            and resolved_mmproj_path
            and mmproj_exists
        ):
            try:
                handler_obj, selected_handler, selected_handler_error = resolve_internal_chat_handler(
                    family=selected_family,
                    mmproj_path=resolved_mmproj_path,
                    image_max_tokens=(4096 if selected_family == FAMILY_QWEN_VL else None),
                    debug_info=handler_debug,
                )
            except Exception as exc:
                selected_handler_error = str(exc)

        requested_n_ctx = max(256, int(n_ctx))
        requested_n_batch = max(32, int(n_batch))
        requested_threads = int(threads)
        requested_threads_or_none: int | None = requested_threads if requested_threads > 0 else None
        requested_kwargs: dict[str, Any] = {
            "model_path": resolved_model_path or str(model_name).strip(),
            "chat_handler": handler_obj,
            "n_ctx": requested_n_ctx,
            "n_gpu_layers": int(n_gpu_layers),
            "n_batch": requested_n_batch,
            "verbose": False,
        }
        if requested_threads_or_none is not None:
            requested_kwargs["n_threads"] = requested_threads_or_none
        if selected_family == FAMILY_QWEN_VL:
            requested_kwargs["swa_full"] = True
            requested_kwargs["top_k"] = 0
            requested_kwargs["pool_size"] = 4194304
            requested_kwargs["image_min_tokens"] = 1024
            requested_kwargs["image_max_tokens"] = 4096

        filtered_kwargs: dict[str, Any] = dict(requested_kwargs)
        dropped_kwargs: list[str] = []
        llama_ctor_error = ""
        if import_ok:
            try:
                from llama_cpp import Llama  # type: ignore

                filtered_kwargs = _filter_kwargs_for_callable(getattr(Llama, "__init__", Llama), requested_kwargs)
                dropped_kwargs = sorted([key for key in requested_kwargs.keys() if key not in filtered_kwargs])
            except Exception as exc:
                llama_ctor_error = str(exc)

        gpu_requested, gpu_requested_value = _gpu_offload_requested(int(n_gpu_layers))
        backend_status = _summarize_backend_status(backend_probe_payload, gpu_requested)
        gpu_offload_hint = ""
        if gpu_requested and backend_status == "CPU-only llama_cpp build detected":
            gpu_offload_hint = (
                f"n_gpu_layers={gpu_requested_value} requests GPU offload, but backend probe indicates CPU-only."
            )

        if not import_ok:
            status_text = "llama_cpp import failed"
        elif backend_status:
            status_text = backend_status
        elif selected_family and not family_supported:
            status_text = f"{_family_label(selected_family)} family not supported by installed llama_cpp build"
        elif family_infer_error:
            status_text = "internal model family could not be inferred from model_name"
        elif model_exists and mmproj_exists and family_supported:
            status_text = "model/mmproj paths resolved; family support detected"
        elif resolve_error:
            status_text = "model/mmproj path resolution failed"
        else:
            status_text = "diagnostics completed"

        payload: dict[str, Any] = {
            "python_version": sys.version.split()[0],
            "platform": py_platform.platform(),
            "llama_cpp_import_ok": import_ok,
            "llama_cpp_import_error": import_error or support.import_error,
            "llama_cpp_version": llama_version,
            "backend_probe": backend_probe_payload.get("backend_probe", {}),
            "backend_probe_method": str(backend_probe_payload.get("backend_probe_method", "") or ""),
            "backend_probe_error": str(backend_probe_payload.get("backend_probe_error", "") or ""),
            "llama_chat_format_relevant_attributes": relevant_attrs,
            "llama_chat_format_supported_handler_classes": supported_classes,
            "detected_available_internal_families": available_families,
            "selected_model_family": selected_family,
            "selected_model_family_label": _family_label(selected_family),
            "selected_family_infer_error": family_infer_error,
            "selected_family_supported": family_supported,
            "selected_chat_handler": selected_handler,
            "selected_chat_handler_error": selected_handler_error,
            "handler_debug": _json_safe_debug_value(handler_debug),
            "model_name": str(model_name).strip(),
            "mmproj_name": str(mmproj_name).strip(),
            "resolved_model_path": resolved_model_path,
            "resolved_mmproj_path": resolved_mmproj_path,
            "model_path_exists": model_exists,
            "mmproj_path_exists": mmproj_exists,
            "gpu_offload_requested": bool(gpu_requested),
            "gpu_offload_requested_value": int(gpu_requested_value),
            "runtime_requested_n_ctx": int(requested_n_ctx),
            "runtime_requested_n_batch": int(requested_n_batch),
            "runtime_requested_threads": int(requested_threads),
            "llama_constructor_requested_kwargs": _json_safe_debug_value(requested_kwargs),
            "llama_constructor_filtered_kwargs": _json_safe_debug_value(filtered_kwargs),
            "llama_constructor_dropped_kwargs": dropped_kwargs,
            "llama_constructor_filter_error": llama_ctor_error,
            "gpu_offload_hint": gpu_offload_hint,
            "path_resolution_error": resolve_error,
            "backend_status": backend_status,
            "status": status_text,
        }
        return json.dumps(payload, indent=2, ensure_ascii=False), status_text
