from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any
import inspect

from .gpm_vlm_internal_multimodal import (
    FAMILY_QWEN_VL,
    approved_internal_family_labels_text,
    assess_internal_model_family_support,
    available_internal_families_text,
    infer_internal_multimodal_family,
    infer_internal_multimodal_family_from_path_hint,
    resolve_internal_chat_handler,
)
from .gpm_vlm_runtime_api import (
    _build_user_prompt,
    _extract_json_object,
    _image_to_data_url,
    _normalize_model_prompts,
    _sanitize_person_prompt_if_environment_only,
)
from .gpm_vlm_runtime_base import GPMVLMRuntime, RUNTIME_MODE_INTERNAL


_GPU_BACKEND_KEYS = ("cuda", "vulkan", "metal", "hip", "sycl", "opencl")


def _json_safe_debug_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe_debug_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe_debug_value(item) for key, item in value.items()}
    return repr(value)


def _probe_call_bool(module: Any, fn_name: str) -> tuple[bool | None, str]:
    fn = getattr(module, fn_name, None)
    if not callable(fn):
        return None, ""
    try:
        return bool(fn()), f"{fn_name}()"
    except Exception as exc:
        return None, f"{fn_name}() failed: {exc}"


def _probe_llama_cpp_backend() -> dict[str, Any]:
    probe: dict[str, Any] = {
        "llama_cpp_import_ok": False,
        "llama_cpp_import_error": "",
        "llama_cpp_version": "",
        "backend_probe": {},
        "backend_probe_method": (
            "llama_cpp callable probes + optional llama_print_system_info + module attribute token scan"
        ),
        "backend_probe_error": "",
    }
    try:
        import llama_cpp  # type: ignore
    except Exception as exc:
        probe["llama_cpp_import_error"] = str(exc)
        probe["backend_probe_error"] = str(exc)
        return probe

    probe["llama_cpp_import_ok"] = True
    probe["llama_cpp_version"] = str(getattr(llama_cpp, "__version__", "") or "")

    capability_specs = {
        "cuda": ("llama_supports_cuda", "llama_supports_cublas"),
        "vulkan": ("llama_supports_vulkan",),
        "metal": ("llama_supports_metal",),
        "hip": ("llama_supports_hipblas", "llama_supports_hip"),
        "sycl": ("llama_supports_sycl",),
        "opencl": ("llama_supports_clblast", "llama_supports_opencl"),
        "gpu_offload": ("llama_supports_gpu_offload",),
    }
    capabilities: dict[str, bool | None] = {}
    capability_probe_sources: dict[str, str] = {}
    capability_probe_errors: dict[str, str] = {}
    for capability_name, probe_functions in capability_specs.items():
        found_value: bool | None = None
        source_text = ""
        for fn_name in probe_functions:
            value, source = _probe_call_bool(llama_cpp, fn_name)
            if source and "failed:" in source:
                capability_probe_errors[capability_name] = source
                continue
            if source:
                found_value = value
                source_text = source
                break
        capabilities[capability_name] = found_value
        if source_text:
            capability_probe_sources[capability_name] = source_text

    system_info_text = ""
    system_info_fn = getattr(llama_cpp, "llama_print_system_info", None)
    if callable(system_info_fn):
        try:
            raw_info = system_info_fn()
            if isinstance(raw_info, bytes):
                system_info_text = raw_info.decode("utf-8", errors="replace")
            else:
                system_info_text = str(raw_info)
            system_info_text = " ".join(system_info_text.split())
        except Exception as exc:
            capability_probe_errors["llama_print_system_info"] = str(exc)

    dir_names = sorted(dir(llama_cpp))
    attribute_hits: dict[str, list[str]] = {}
    for backend_key in _GPU_BACKEND_KEYS:
        hits = [name for name in dir_names if backend_key in name.casefold()]
        if hits:
            attribute_hits[backend_key] = hits[:12]

    inferred_from_system_info: dict[str, bool] = {}
    lower_system_info = system_info_text.casefold()
    for backend_key in _GPU_BACKEND_KEYS:
        if lower_system_info and backend_key in lower_system_info:
            inferred_from_system_info[backend_key] = True
            if capabilities.get(backend_key) is None:
                capabilities[backend_key] = True
                capability_probe_sources[backend_key] = "llama_print_system_info token match"

    known_gpu_values = [capabilities.get(name) for name in _GPU_BACKEND_KEYS]
    known_gpu_true = any(value is True for value in known_gpu_values)
    known_gpu_explicit = [value for value in known_gpu_values if value is not None]
    cpu_only_confident = bool(known_gpu_explicit) and not known_gpu_true
    if capabilities.get("gpu_offload") is False and not known_gpu_true:
        cpu_only_confident = True

    probe["backend_probe"] = {
        "capabilities": capabilities,
        "capability_probe_sources": capability_probe_sources,
        "capability_probe_errors": capability_probe_errors,
        "system_info_sample": system_info_text[:500],
        "system_info_inferred_backends": inferred_from_system_info,
        "module_attribute_hits": attribute_hits,
        "gpu_backend_any_true": known_gpu_true,
        "cpu_only_confident": cpu_only_confident,
    }
    return probe


def _gpu_offload_requested(n_gpu_layers: int) -> tuple[bool, int]:
    value = int(n_gpu_layers)
    return (value == -1 or value > 0), value


def _summarize_backend_status(probe: dict[str, Any], gpu_offload_requested: bool) -> str:
    import_ok = bool(probe.get("llama_cpp_import_ok"))
    if not import_ok:
        return "llama_cpp import failed"

    backend_probe = probe.get("backend_probe")
    capabilities = backend_probe.get("capabilities", {}) if isinstance(backend_probe, dict) else {}
    cuda_capable = capabilities.get("cuda")
    any_gpu_capable = any(capabilities.get(key) is True for key in _GPU_BACKEND_KEYS)
    cpu_only_confident = bool(
        backend_probe.get("cpu_only_confident", False) if isinstance(backend_probe, dict) else False
    )

    if cuda_capable is True:
        return "CUDA-capable llama_cpp build detected"
    if gpu_offload_requested and cpu_only_confident:
        return "CPU-only llama_cpp build detected"
    if gpu_offload_requested and not any_gpu_capable:
        return "GPU offload requested but backend capability could not be confirmed"
    if cpu_only_confident:
        return "CPU-only llama_cpp build detected"
    if any_gpu_capable:
        return "GPU-capable llama_cpp build detected"
    return "Backend capability unknown / not exposed by current build"


@dataclass(frozen=True)
class GPMInternalRuntimeConfig:
    model_path: str
    mmproj_path: str
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 512
    threads: int = 0
    batch_size: int = 512
    image_min_tokens: int = 1024
    image_max_tokens: int = 4096
    top_k: int = 0
    pool_size: int = 4194304
    keep_model_loaded: bool = False
    debug_mode: bool = False


class GPMGGUFInternalRuntime(GPMVLMRuntime):
    runtime_mode = RUNTIME_MODE_INTERNAL

    _cache_lock: Lock = Lock()
    _cached_signature: tuple[Any, ...] | None = None
    _cached_llm: Any = None

    def __init__(self, model_name: str, mmproj_name: str, timeout_seconds: int, config: GPMInternalRuntimeConfig):
        self.model_name = str(model_name).strip()
        self.mmproj_name = str(mmproj_name).strip()
        self.timeout_seconds = max(1, int(timeout_seconds))
        self.config = config
        self._llm: Any = None
        self._chat_handler_name = ""
        self._model_family = ""
        self._detected_model_family = ""
        self._family_support_status = ""
        self._family_support_reason = ""
        self._scan_family_approved = False
        self._available_families = ""
        self._approved_scan_families = ""
        self._startup_debug: dict[str, Any] = {}
        self._last_scan_debug_trace: dict[str, Any] = {}

    def _source_image_sha256(self, image_path: Path) -> str:
        digest = hashlib.sha256()
        with image_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def _build_multimodal_messages(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_data_url: str,
    ) -> tuple[list[dict[str, Any]], str]:
        if self._model_family == FAMILY_QWEN_VL:
            image_payload: Any = {"url": image_data_url}
            image_payload_mode = "image_url_object"
        else:
            image_payload = image_data_url
            image_payload_mode = "image_url_string"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": image_payload},
                ],
            },
        ]
        return messages, image_payload_mode

    def _diagnostic_prefix(
        self,
        *,
        stage: str,
        resolved_model_path: str,
        resolved_mmproj_path: str,
        inferred_family: str = "",
        selected_handler: str = "",
    ) -> str:
        return (
            f"internal startup stage={stage} | "
            f"inferred_family={inferred_family or self._model_family or '<unknown>'} | "
            f"selected_handler={selected_handler or self._chat_handler_name or '<unresolved>'} | "
            f"available_internal_families={self._available_families or 'none'} | "
            f"resolved_model_path={resolved_model_path} | "
            f"resolved_mmproj_path={resolved_mmproj_path}"
        )

    def _signature(self) -> tuple[Any, ...]:
        return (
            Path(self.config.model_path).resolve().as_posix(),
            Path(self.config.mmproj_path).resolve().as_posix(),
            int(self.config.n_ctx),
            int(self.config.n_gpu_layers),
            int(self.config.threads),
            int(self.config.batch_size),
        )

    def _file_size_bytes(self, path: Path) -> int:
        try:
            if path.exists() and path.is_file():
                return int(path.stat().st_size)
        except Exception:
            return 0
        return 0

    def _llama_cpp_version(self) -> str:
        return str(_probe_llama_cpp_backend().get("llama_cpp_version", "") or "")

    def _filter_kwargs_for_callable(self, fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
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

    def _load_llama(self) -> tuple[Any | None, str]:
        model_path = Path(self.config.model_path).expanduser()
        mmproj_path = Path(self.config.mmproj_path).expanduser()
        self._startup_debug = {
            "llama_cpp_version": self._llama_cpp_version(),
            "llama_cpp_import_ok": False,
            "llama_cpp_import_error": "",
            "backend_probe": {},
            "backend_probe_method": "",
            "backend_probe_error": "",
            "backend_status": "",
            "inferred_family": "",
            "selected_handler": "",
            "detected_model_family": "",
            "selected_chat_handler": "",
            "family_support_status": "",
            "support_reason": "",
            "scan_family_approved": False,
            "available_internal_families": "",
            "approved_internal_scan_families": "",
            "resolved_model_path": str(model_path),
            "resolved_mmproj_path": str(mmproj_path),
            "model_file_size_bytes": self._file_size_bytes(model_path),
            "mmproj_file_size_bytes": self._file_size_bytes(mmproj_path),
            "constructor_exception": "",
            "keep_model_loaded": bool(self.config.keep_model_loaded),
            "handler_first_attempted": "",
            "handler_constructor_kwargs": {},
            "llama_constructor_kwargs": {},
            "llama_constructor_kwargs_raw": {},
            "llama_constructor_kwargs_filtered": {},
            "llama_constructor_dropped_kwargs": [],
            "gpu_offload_requested": False,
            "gpu_offload_requested_value": int(self.config.n_gpu_layers),
            "runtime_requested_n_ctx": 0,
            "runtime_requested_n_batch": 0,
            "runtime_requested_threads": 0,
            "gpu_offload_hint": "",
            "family_resolution_mode": "",
            "family_resolved_via_explicit_class_match": False,
            "family_resolved_via_fallback_class_preference_order": False,
        }
        backend_probe = _probe_llama_cpp_backend()
        self._startup_debug["llama_cpp_version"] = str(backend_probe.get("llama_cpp_version", "") or "")
        self._startup_debug["llama_cpp_import_ok"] = bool(backend_probe.get("llama_cpp_import_ok", False))
        self._startup_debug["llama_cpp_import_error"] = str(backend_probe.get("llama_cpp_import_error", "") or "")
        self._startup_debug["backend_probe"] = dict(backend_probe.get("backend_probe", {}))
        self._startup_debug["backend_probe_method"] = str(backend_probe.get("backend_probe_method", "") or "")
        self._startup_debug["backend_probe_error"] = str(backend_probe.get("backend_probe_error", "") or "")
        try:
            resolved_model_path = str(model_path.resolve())
            resolved_mmproj_path = str(mmproj_path.resolve())
        except Exception as exc:
            self._startup_debug["constructor_exception"] = str(exc)
            return None, (
                "startup_failure_phase=before_model_construction | "
                f"internal startup stage=path_resolution | exception={exc}"
            )
        self._available_families = available_internal_families_text()
        self._approved_scan_families = approved_internal_family_labels_text()
        self._startup_debug["available_internal_families"] = self._available_families
        self._startup_debug["approved_internal_scan_families"] = self._approved_scan_families
        self._startup_debug["resolved_model_path"] = resolved_model_path
        self._startup_debug["resolved_mmproj_path"] = resolved_mmproj_path

        if not model_path.exists() or not model_path.is_file():
            return (
                None,
                "startup_failure_phase=before_model_construction | "
                f"{self._diagnostic_prefix(stage='path_resolution', resolved_model_path=resolved_model_path, resolved_mmproj_path=resolved_mmproj_path)} | "
                f"error=internal model file was not found: {model_path}",
            )
        if not mmproj_path.exists() or not mmproj_path.is_file():
            return (
                None,
                "startup_failure_phase=before_model_construction | "
                f"{self._diagnostic_prefix(stage='path_resolution', resolved_model_path=resolved_model_path, resolved_mmproj_path=resolved_mmproj_path)} | "
                f"error=internal mmproj file was not found: {mmproj_path}",
            )

        try:
            inferred_family, infer_error = infer_internal_multimodal_family(
                str(model_path),
                mmproj_path=str(mmproj_path),
            )
        except Exception as exc:
            return (
                None,
                "startup_failure_phase=before_model_construction | "
                f"{self._diagnostic_prefix(stage='family_inference', resolved_model_path=resolved_model_path, resolved_mmproj_path=resolved_mmproj_path)} | "
                f"exception={exc}",
            )
        if infer_error:
            return (
                None,
                "startup_failure_phase=before_model_construction | "
                f"{self._diagnostic_prefix(stage='family_inference', resolved_model_path=resolved_model_path, resolved_mmproj_path=resolved_mmproj_path)} | "
                f"error={infer_error}",
            )
        self._model_family = inferred_family
        self._startup_debug["inferred_family"] = inferred_family
        support_decision = assess_internal_model_family_support(
            model_path=str(model_path),
            mmproj_path=str(mmproj_path),
            inferred_runtime_family=inferred_family,
        )
        self._detected_model_family = support_decision.detected_model_family
        self._family_support_status = support_decision.family_support_status
        self._family_support_reason = support_decision.support_reason
        self._scan_family_approved = bool(support_decision.scan_approved)
        self._startup_debug["detected_model_family"] = self._detected_model_family
        self._startup_debug["family_support_status"] = self._family_support_status
        self._startup_debug["support_reason"] = self._family_support_reason
        self._startup_debug["scan_family_approved"] = self._scan_family_approved
        self._startup_debug["selected_chat_handler"] = self._chat_handler_name
        if not self._scan_family_approved:
            detected_label = support_decision.detected_model_family_label or self._detected_model_family or "unknown"
            support_state = self._family_support_status or "unsupported"
            reason = self._family_support_reason or (
                "Internal scan correctness is currently verified for Qwen2.5-VL only. "
                "This model may load, but its multimodal input/output format is not yet validated in GPM."
            )
            return (
                None,
                f"{reason} detected_internal_model_family={detected_label} ({support_state}).",
            )
        mmproj_family_hint = infer_internal_multimodal_family_from_path_hint(str(mmproj_path))
        if mmproj_family_hint and mmproj_family_hint != inferred_family:
            return (
                None,
                "startup_failure_phase=before_model_construction | "
                f"{self._diagnostic_prefix(stage='family_validation', resolved_model_path=resolved_model_path, resolved_mmproj_path=resolved_mmproj_path, inferred_family=inferred_family)} | "
                f"error=model/mmproj may be mismatched (model family inferred as '{inferred_family}', mmproj hint inferred as '{mmproj_family_hint}'). "
                "Choose matching model and mmproj files from the same multimodal family.",
            )

        handler_debug: dict[str, Any] = {}
        try:
            chat_handler, chat_handler_name, handler_error = resolve_internal_chat_handler(
                family=inferred_family,
                mmproj_path=str(mmproj_path),
                image_max_tokens=(
                    int(self.config.image_max_tokens)
                    if inferred_family == FAMILY_QWEN_VL
                    else None
                ),
                debug_info=handler_debug if self.config.debug_mode else None,
            )
        except Exception as exc:
            return (
                None,
                "startup_failure_phase=before_model_construction | "
                f"{self._diagnostic_prefix(stage='handler_detection_or_init', resolved_model_path=resolved_model_path, resolved_mmproj_path=resolved_mmproj_path, inferred_family=inferred_family)} | "
                f"exception={exc}",
            )
        if chat_handler is None:
            if self.config.debug_mode and handler_debug:
                self._startup_debug["handler_first_attempted"] = str(handler_debug.get("handler_first_attempted", ""))
                self._startup_debug["handler_constructor_kwargs"] = dict(
                    handler_debug.get("handler_constructor_kwargs", {})
                )
                mode = str(handler_debug.get("handler_selection_mode", ""))
                self._startup_debug["family_resolution_mode"] = mode
                self._startup_debug["family_resolved_via_explicit_class_match"] = mode == "explicit_class_match"
                self._startup_debug["family_resolved_via_fallback_class_preference_order"] = (
                    mode == "fallback_class_preference_order"
                )
            return (
                None,
                "startup_failure_phase=before_model_construction | "
                f"{self._diagnostic_prefix(stage='handler_detection_or_init', resolved_model_path=resolved_model_path, resolved_mmproj_path=resolved_mmproj_path, inferred_family=inferred_family)} | "
                f"error={handler_error}",
            )
        self._chat_handler_name = chat_handler_name
        self._startup_debug["selected_handler"] = self._chat_handler_name
        self._startup_debug["selected_chat_handler"] = self._chat_handler_name
        if self.config.debug_mode and handler_debug:
            self._startup_debug["handler_first_attempted"] = str(handler_debug.get("handler_first_attempted", ""))
            self._startup_debug["handler_constructor_kwargs"] = dict(handler_debug.get("handler_constructor_kwargs", {}))
            mode = str(handler_debug.get("handler_selection_mode", ""))
            self._startup_debug["family_resolution_mode"] = mode
            self._startup_debug["family_resolved_via_explicit_class_match"] = mode == "explicit_class_match"
            self._startup_debug["family_resolved_via_fallback_class_preference_order"] = (
                mode == "fallback_class_preference_order"
            )

        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as exc:
            return (
                None,
                "startup_failure_phase=before_model_construction | "
                f"{self._diagnostic_prefix(stage='llama_cpp_import', resolved_model_path=resolved_model_path, resolved_mmproj_path=resolved_mmproj_path, inferred_family=inferred_family, selected_handler=self._chat_handler_name)} | "
                f"error=internal runtime requires llama-cpp-python with vision support (failed import: {exc}). "
                "Install/update llama-cpp-python for your platform.",
            )

        n_threads = int(self.config.threads)
        if n_threads <= 0:
            n_threads = None  # type: ignore[assignment]

        requested_kwargs: dict[str, Any] = {
            "model_path": str(model_path),
            "chat_handler": chat_handler,
            "n_ctx": max(256, int(self.config.n_ctx)),
            "n_gpu_layers": int(self.config.n_gpu_layers),
            "n_batch": max(32, int(self.config.batch_size)),
            "verbose": False,
        }
        if n_threads is not None:
            requested_kwargs["n_threads"] = n_threads

        if inferred_family == FAMILY_QWEN_VL:
            requested_kwargs["swa_full"] = True
            requested_kwargs["top_k"] = int(self.config.top_k)
            requested_kwargs["pool_size"] = int(self.config.pool_size)
            requested_kwargs["image_min_tokens"] = max(0, int(self.config.image_min_tokens))
            requested_kwargs["image_max_tokens"] = max(0, int(self.config.image_max_tokens))

        gpu_offload_requested, gpu_offload_requested_value = _gpu_offload_requested(int(self.config.n_gpu_layers))
        self._startup_debug["gpu_offload_requested"] = bool(gpu_offload_requested)
        self._startup_debug["gpu_offload_requested_value"] = int(gpu_offload_requested_value)
        self._startup_debug["runtime_requested_n_ctx"] = int(requested_kwargs["n_ctx"])
        self._startup_debug["runtime_requested_n_batch"] = int(requested_kwargs["n_batch"])
        self._startup_debug["runtime_requested_threads"] = (
            int(requested_kwargs["n_threads"]) if "n_threads" in requested_kwargs else 0
        )
        backend_status = _summarize_backend_status(backend_probe, gpu_offload_requested)
        self._startup_debug["backend_status"] = backend_status

        filtered_kwargs = self._filter_kwargs_for_callable(getattr(Llama, "__init__", Llama), requested_kwargs)
        dropped_llama_kwargs = sorted([key for key in requested_kwargs.keys() if key not in filtered_kwargs])
        if self.config.debug_mode:
            self._startup_debug["llama_constructor_kwargs_raw"] = dict(requested_kwargs)
            self._startup_debug["llama_constructor_kwargs_filtered"] = dict(filtered_kwargs)
        else:
            self._startup_debug["llama_constructor_kwargs_raw"] = {}
            self._startup_debug["llama_constructor_kwargs_filtered"] = {}
        self._startup_debug["llama_constructor_dropped_kwargs"] = list(dropped_llama_kwargs)
        self._startup_debug["llama_constructor_kwargs"] = {
            "model_path": str(model_path),
            "chat_handler_class": self._chat_handler_name,
            "n_ctx": requested_kwargs["n_ctx"],
            "n_gpu_layers": requested_kwargs["n_gpu_layers"],
            "n_batch": requested_kwargs["n_batch"],
            "verbose": requested_kwargs["verbose"],
            "n_threads": requested_kwargs.get("n_threads", None),
            "requested_kwargs": _json_safe_debug_value(dict(requested_kwargs)),
            "filtered_kwargs": _json_safe_debug_value(dict(filtered_kwargs)),
            "dropped_kwargs": dropped_llama_kwargs,
        }
        if self.config.debug_mode and gpu_offload_requested and backend_status == "CPU-only llama_cpp build detected":
            self._startup_debug["gpu_offload_hint"] = (
                f"n_gpu_layers={gpu_offload_requested_value} requests GPU offload, "
                "but backend probe indicates a CPU-only llama_cpp build."
            )
        else:
            self._startup_debug["gpu_offload_hint"] = ""

        constructed_llm: Any | None = None
        try:
            constructed_llm = Llama(**filtered_kwargs)
        except Exception as exc:
            # Some llama-cpp-python builds emit a secondary __del__/sampler AttributeError
            # after constructor failure. The primary model-load exception below is the real cause.
            self._llm = None
            chat_handler = None
            constructed_llm = None
            self._startup_debug["constructor_exception"] = str(exc)
            environment_hint = ""
            if inferred_family == FAMILY_QWEN_VL:
                environment_hint = (
                    " Qwen-VL GGUF often requires a vision-capable llama-cpp-python build "
                    "with Qwen-VL support in the bundled llama.cpp backend (not just chat handler symbols)."
                )
            return (
                None,
                "startup_failure_phase=during_model_construction | "
                f"{self._diagnostic_prefix(stage='llama_construction', resolved_model_path=resolved_model_path, resolved_mmproj_path=resolved_mmproj_path, inferred_family=self._model_family, selected_handler=self._chat_handler_name)} | "
                "error=llama-cpp-python recognized the family/handler, but failed to load the selected GGUF model file. "
                "This usually indicates model/build compatibility or a bad/corrupt GGUF, not a node wiring issue. "
                f"{environment_hint}"
                f"constructor_exception={exc}",
            )
        return constructed_llm, ""

    def start(self) -> tuple[bool, str]:
        signature = self._signature()
        if self.config.keep_model_loaded:
            with self._cache_lock:
                if self.__class__._cached_signature == signature and self.__class__._cached_llm is not None:
                    self._llm = self.__class__._cached_llm
                    return True, ""

        llm, error = self._load_llama()
        if llm is None:
            return False, error

        self._llm = llm
        if self.config.keep_model_loaded:
            with self._cache_lock:
                self.__class__._cached_signature = signature
                self.__class__._cached_llm = llm
        return True, ""

    def stop(self) -> None:
        if self.config.keep_model_loaded:
            return

        current = self._llm
        self._llm = None
        with self._cache_lock:
            if self.__class__._cached_llm is not None and self.__class__._cached_llm is current:
                self.__class__._cached_signature = None
                self.__class__._cached_llm = None

    def generate(self, image_path: Path, preset: dict[str, Any]) -> tuple[str, str, str]:
        if self._llm is None:
            return "", "", "internal runtime model is not loaded"

        # Force per-image request state to be rebuilt every call.
        self._last_scan_debug_trace = {}
        source_image_full_path = str(image_path.resolve())
        source_image_filename = image_path.name
        source_image_sha256 = ""
        system_prompt = str(preset.get("system_prompt", "")).strip()
        user_prompt = _build_user_prompt(preset)
        image_payload_mode = ""
        raw_response_text = ""
        parsed_person_prompt = ""
        parsed_scene_prompt = ""
        trace_support = {
            "detected_model_family": self._detected_model_family,
            "selected_chat_handler": self._chat_handler_name,
            "family_support_status": self._family_support_status,
            "support_reason": self._family_support_reason,
        }

        try:
            image_data_url = _image_to_data_url(image_path)
            source_image_sha256 = self._source_image_sha256(image_path)
        except Exception as exc:
            self._last_scan_debug_trace = {
                "source_image_filename": source_image_filename,
                "source_image_full_path": source_image_full_path,
                "source_image_sha256": source_image_sha256,
                "model_prompt_sent": user_prompt,
                "system_prompt_sent": system_prompt,
                "raw_model_response": "",
                "parsed_person_prompt": "",
                "parsed_scene_prompt": "",
                **trace_support,
                "error": f"image encode error: {exc}",
            }
            return "", "", f"image encode error: {exc}"

        messages, image_payload_mode = self._build_multimodal_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_data_url=image_data_url,
        )

        try:
            completion_kwargs: dict[str, Any] = {
                "messages": messages,
                "temperature": float(self.config.temperature),
                "top_p": float(self.config.top_p),
                "max_tokens": max(32, int(self.config.max_tokens)),
            }
            if self._model_family == FAMILY_QWEN_VL:
                completion_kwargs["stop"] = ["<|im_end|>", "<|im_start|>"]
            response = self._llm.create_chat_completion(**completion_kwargs)
        except Exception as exc:
            self._last_scan_debug_trace = {
                "source_image_filename": source_image_filename,
                "source_image_full_path": source_image_full_path,
                "source_image_sha256": source_image_sha256,
                "model_prompt_sent": user_prompt,
                "system_prompt_sent": system_prompt,
                "raw_model_response": "",
                "parsed_person_prompt": "",
                "parsed_scene_prompt": "",
                "request_image_payload_mode": image_payload_mode,
                **trace_support,
                "error": f"internal runtime inference failed: {exc}",
            }
            return "", "", f"internal runtime inference failed: {exc}"

        try:
            content = response["choices"][0]["message"]["content"]
        except Exception:
            self._last_scan_debug_trace = {
                "source_image_filename": source_image_filename,
                "source_image_full_path": source_image_full_path,
                "source_image_sha256": source_image_sha256,
                "model_prompt_sent": user_prompt,
                "system_prompt_sent": system_prompt,
                "raw_model_response": "",
                "parsed_person_prompt": "",
                "parsed_scene_prompt": "",
                "request_image_payload_mode": image_payload_mode,
                **trace_support,
                "error": "internal runtime response format was not recognized",
            }
            return "", "", "internal runtime response format was not recognized"

        if isinstance(content, list):
            content = "\n".join(
                str(part.get("text", "")).strip()
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ).strip()
        raw_response_text = str(content)

        output_payload = _extract_json_object(raw_response_text)
        if output_payload is None:
            self._last_scan_debug_trace = {
                "source_image_filename": source_image_filename,
                "source_image_full_path": source_image_full_path,
                "source_image_sha256": source_image_sha256,
                "model_prompt_sent": user_prompt,
                "system_prompt_sent": system_prompt,
                "raw_model_response": raw_response_text,
                "parsed_person_prompt": "",
                "parsed_scene_prompt": "",
                "request_image_payload_mode": image_payload_mode,
                **trace_support,
                "error": f"internal runtime did not return strict JSON: {raw_response_text[:600]}",
            }
            return "", "", f"internal runtime did not return strict JSON: {raw_response_text[:600]}"

        family = str(preset.get("family", "SDXL"))
        person_prompt, scene_prompt = _normalize_model_prompts(output_payload, family)
        person_prompt = _sanitize_person_prompt_if_environment_only(person_prompt)
        parsed_person_prompt = person_prompt
        parsed_scene_prompt = scene_prompt
        self._last_scan_debug_trace = {
            "source_image_filename": source_image_filename,
            "source_image_full_path": source_image_full_path,
            "source_image_sha256": source_image_sha256,
            "model_prompt_sent": user_prompt,
            "system_prompt_sent": system_prompt,
            "raw_model_response": raw_response_text,
            "parsed_person_prompt": parsed_person_prompt,
            "parsed_scene_prompt": parsed_scene_prompt,
            "request_image_payload_mode": image_payload_mode,
            **trace_support,
            "error": "",
        }
        return person_prompt, scene_prompt, ""

    def consume_last_scan_debug_trace(self) -> dict[str, Any]:
        trace = dict(self._last_scan_debug_trace) if isinstance(self._last_scan_debug_trace, dict) else {}
        self._last_scan_debug_trace = {}
        return trace

    def startup_debug_metadata(self) -> dict[str, Any]:
        safe_payload = {key: _json_safe_debug_value(value) for key, value in self._startup_debug.items()}
        return dict(safe_payload)

    def summary_metadata(self) -> dict[str, Any]:
        resolved_model = Path(self.config.model_path).expanduser().resolve()
        resolved_mmproj = Path(self.config.mmproj_path).expanduser().resolve()
        debug_enabled = bool(self.config.debug_mode)
        return {
            "runtime_mode": RUNTIME_MODE_INTERNAL,
            "model_name": self.model_name or resolved_model.name,
            "mmproj_name": self.mmproj_name,
            "model_path_resolved": str(resolved_model),
            "mmproj_path_resolved": str(resolved_mmproj),
            "internal_multimodal_family": self._model_family,
            "internal_chat_handler": self._chat_handler_name,
            "detected_model_family": self._detected_model_family,
            "selected_chat_handler": self._chat_handler_name,
            "family_support_status": self._family_support_status,
            "support_reason": self._family_support_reason,
            "scan_family_approved": bool(self._scan_family_approved),
            "selected_internal_family": self._model_family,
            "selected_internal_handler": self._chat_handler_name,
            "resolved_model_path": str(resolved_model),
            "resolved_mmproj_path": str(resolved_mmproj),
            "llama_cpp_version": str(self._startup_debug.get("llama_cpp_version", "") or ""),
            "llama_cpp_import_ok": bool(self._startup_debug.get("llama_cpp_import_ok", False)),
            "llama_cpp_import_error": str(self._startup_debug.get("llama_cpp_import_error", "") or ""),
            "backend_probe": (
                _json_safe_debug_value(self._startup_debug.get("backend_probe", {})) if debug_enabled else {}
            ),
            "backend_probe_method": str(self._startup_debug.get("backend_probe_method", "") or ""),
            "backend_probe_error": str(self._startup_debug.get("backend_probe_error", "") or ""),
            "backend_status": str(self._startup_debug.get("backend_status", "") or ""),
            "gpu_offload_requested": bool(self._startup_debug.get("gpu_offload_requested", False)),
            "gpu_offload_requested_value": int(self._startup_debug.get("gpu_offload_requested_value", int(self.config.n_gpu_layers))),
            "runtime_requested_n_ctx": int(self._startup_debug.get("runtime_requested_n_ctx", int(self.config.n_ctx))),
            "runtime_requested_n_batch": int(
                self._startup_debug.get("runtime_requested_n_batch", int(self.config.batch_size))
            ),
            "runtime_requested_threads": int(
                self._startup_debug.get("runtime_requested_threads", int(self.config.threads))
            ),
            "llama_constructor_dropped_kwargs": (
                list(self._startup_debug.get("llama_constructor_dropped_kwargs", [])) if debug_enabled else []
            ),
            "gpu_offload_hint": str(self._startup_debug.get("gpu_offload_hint", "") or "") if debug_enabled else "",
            "available_internal_families": self._available_families,
            "approved_internal_scan_families": self._approved_scan_families,
            "keep_model_loaded": bool(self.config.keep_model_loaded),
            "debug_mode": bool(self.config.debug_mode),
            "n_ctx": int(self.config.n_ctx),
            "n_gpu_layers": int(self.config.n_gpu_layers),
            "threads": int(self.config.threads),
            "batch_size": int(self.config.batch_size),
            "image_min_tokens": int(self.config.image_min_tokens),
            "image_max_tokens": int(self.config.image_max_tokens),
            "top_k": int(self.config.top_k),
            "pool_size": int(self.config.pool_size),
            "temperature": float(self.config.temperature),
            "top_p": float(self.config.top_p),
            "max_tokens": int(self.config.max_tokens),
        }
