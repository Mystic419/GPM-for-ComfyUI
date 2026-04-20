from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FAMILY_LLAVA = "llava"
FAMILY_QWEN_VL = "qwen_vl"

DETECTED_FAMILY_LLAVA = "llava"
DETECTED_FAMILY_QWEN25_VL = "qwen2_5_vl"
DETECTED_FAMILY_QWEN3_VL = "qwen3_vl"
DETECTED_FAMILY_GLIESE_QWEN35_VL = "gliese_qwen3_5_vl"
DETECTED_FAMILY_QWEN_VL_UNVERIFIED = "qwen_vl_unverified"
DETECTED_FAMILY_UNKNOWN = "unknown"

SUPPORT_STATUS_SUPPORTED = "supported"
SUPPORT_STATUS_EXPERIMENTAL = "experimental"
SUPPORT_STATUS_UNSUPPORTED = "unsupported"

_DETECTED_FAMILY_LABELS: dict[str, str] = {
    DETECTED_FAMILY_LLAVA: "LLaVA",
    DETECTED_FAMILY_QWEN25_VL: "Qwen2.5-VL",
    DETECTED_FAMILY_QWEN3_VL: "Qwen3-VL",
    DETECTED_FAMILY_GLIESE_QWEN35_VL: "Gliese/Qwen3.5-VL",
    DETECTED_FAMILY_QWEN_VL_UNVERIFIED: "Qwen-VL (unverified variant)",
    DETECTED_FAMILY_UNKNOWN: "unknown",
}

_INTERNAL_FAMILY_SUPPORT_MAP: dict[str, dict[str, Any]] = {
    DETECTED_FAMILY_QWEN25_VL: {
        "status": SUPPORT_STATUS_SUPPORTED,
        "scan_approved": True,
        "reason": "Internal scan correctness is verified for Qwen2.5-VL.",
    },
    DETECTED_FAMILY_QWEN3_VL: {
        "status": SUPPORT_STATUS_EXPERIMENTAL,
        "scan_approved": False,
        "reason": (
            "Internal scan correctness is currently verified for Qwen2.5-VL only. "
            "This model may load, but its multimodal input/output format is not yet validated in GPM."
        ),
    },
    DETECTED_FAMILY_GLIESE_QWEN35_VL: {
        "status": SUPPORT_STATUS_EXPERIMENTAL,
        "scan_approved": False,
        "reason": (
            "Internal scan correctness is currently verified for Qwen2.5-VL only. "
            "This model may load, but its multimodal input/output format is not yet validated in GPM."
        ),
    },
    DETECTED_FAMILY_QWEN_VL_UNVERIFIED: {
        "status": SUPPORT_STATUS_EXPERIMENTAL,
        "scan_approved": False,
        "reason": (
            "Internal scan correctness is currently verified for Qwen2.5-VL only. "
            "This Qwen-VL variant is not yet validated in GPM."
        ),
    },
    DETECTED_FAMILY_LLAVA: {
        "status": SUPPORT_STATUS_UNSUPPORTED,
        "scan_approved": False,
        "reason": (
            "Internal scan correctness is currently verified for Qwen2.5-VL only. "
            "LLaVA-family internal scanning is not yet validated in GPM."
        ),
    },
    DETECTED_FAMILY_UNKNOWN: {
        "status": SUPPORT_STATUS_UNSUPPORTED,
        "scan_approved": False,
        "reason": (
            "Internal scan correctness is currently verified for Qwen2.5-VL only. "
            "This internal multimodal family is not recognized as validated in GPM."
        ),
    },
}


@dataclass(frozen=True)
class GPMInternalFamilySupportDecision:
    detected_model_family: str
    detected_model_family_label: str
    normalized_runtime_family: str
    family_support_status: str
    support_reason: str
    scan_approved: bool


def _normalize_pairing_name(raw: str) -> str:
    text = str(raw).casefold().replace("mmproj", "")
    text = re.sub(r"\.gguf$", "", text)
    text = re.sub(r"[-_.]?q\d+(_[a-z0-9]+)?$", "", text)
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def _mmproj_matches_model(model_path: str, mmproj_path: str) -> bool:
    model_key = _normalize_pairing_name(Path(str(model_path)).stem)
    mmproj_key = _normalize_pairing_name(Path(str(mmproj_path)).stem)
    if not model_key or not mmproj_key:
        return False
    if model_key == mmproj_key and len(model_key) >= 8:
        return True
    if len(model_key) >= 12 and mmproj_key in model_key:
        return True
    if len(mmproj_key) >= 12 and model_key in mmproj_key:
        return True
    return False


def _looks_like_qwen_multimodal_name(name: str) -> tuple[bool, bool]:
    text = str(name).casefold()
    has_qwen = any(token in text for token in ("qwen", "qwen3.5", "qwen35", "qwen3"))
    has_vl_marker = re.search(r"(^|[^a-z0-9])vl([^a-z0-9]|$)", text) is not None
    has_captioning = "captioning" in text or "nsfw-captioning" in text or "vision-captioning" in text
    has_strong_multimodal_token = any(
        token in text
        for token in (
            "qwen-vl",
            "qwen2-vl",
            "qwen2.5-vl",
            "qwen25-vl",
            "qwen3-vl",
            "gliese",
        )
    )
    strong_multimodal_name = has_strong_multimodal_token or (has_qwen and (has_captioning or has_vl_marker))
    return has_qwen, strong_multimodal_name


def _path_hint_text(model_path: str, mmproj_path: str = "") -> str:
    model_text = Path(str(model_path)).name.casefold()
    mmproj_text = Path(str(mmproj_path)).name.casefold() if mmproj_path else ""
    return f"{model_text} {mmproj_text}".strip()


def detect_internal_model_family(model_path: str, mmproj_path: str = "", inferred_runtime_family: str = "") -> str:
    hint_text = _path_hint_text(model_path, mmproj_path)
    if "gliese" in hint_text:
        return DETECTED_FAMILY_GLIESE_QWEN35_VL
    if any(token in hint_text for token in ("qwen3.5", "qwen35", "qwen3-vl", "qwen3_vl", "qwen3 vl")):
        return DETECTED_FAMILY_QWEN3_VL
    if any(
        token in hint_text
        for token in (
            "qwen2.5-vl",
            "qwen2_5-vl",
            "qwen2.5_vl",
            "qwen25-vl",
            "qwen25_vl",
            "qwen2.5vl",
            "qwen25vl",
            "qwen2.5 vl",
            "qwen25 vl",
        )
    ):
        return DETECTED_FAMILY_QWEN25_VL
    if any(token in hint_text for token in ("llava", "bakllava")):
        return DETECTED_FAMILY_LLAVA
    has_qwen_lineage, has_qwen_multimodal_name = _looks_like_qwen_multimodal_name(hint_text)
    if has_qwen_lineage and has_qwen_multimodal_name:
        return DETECTED_FAMILY_QWEN_VL_UNVERIFIED
    if inferred_runtime_family == FAMILY_QWEN_VL:
        return DETECTED_FAMILY_QWEN_VL_UNVERIFIED
    if inferred_runtime_family == FAMILY_LLAVA:
        return DETECTED_FAMILY_LLAVA
    return DETECTED_FAMILY_UNKNOWN


def assess_internal_model_family_support(
    model_path: str,
    mmproj_path: str = "",
    inferred_runtime_family: str = "",
) -> GPMInternalFamilySupportDecision:
    detected_family = detect_internal_model_family(
        model_path=model_path,
        mmproj_path=mmproj_path,
        inferred_runtime_family=inferred_runtime_family,
    )
    spec = _INTERNAL_FAMILY_SUPPORT_MAP.get(detected_family, _INTERNAL_FAMILY_SUPPORT_MAP[DETECTED_FAMILY_UNKNOWN])
    return GPMInternalFamilySupportDecision(
        detected_model_family=detected_family,
        detected_model_family_label=_DETECTED_FAMILY_LABELS.get(detected_family, detected_family),
        normalized_runtime_family=str(inferred_runtime_family or ""),
        family_support_status=str(spec.get("status", SUPPORT_STATUS_UNSUPPORTED)),
        support_reason=str(spec.get("reason", "")).strip(),
        scan_approved=bool(spec.get("scan_approved", False)),
    )


def approved_internal_family_labels_text() -> str:
    labels: list[str] = []
    for family_key, spec in _INTERNAL_FAMILY_SUPPORT_MAP.items():
        if bool(spec.get("scan_approved", False)):
            labels.append(_DETECTED_FAMILY_LABELS.get(family_key, family_key))
    labels = sorted(set(labels))
    return ", ".join(labels)


@dataclass(frozen=True)
class GPMChatHandlerSupport:
    family_to_classes: dict[str, tuple[str, ...]]
    import_error: str = ""

    def available_families(self) -> tuple[str, ...]:
        return tuple(sorted([key for key, value in self.family_to_classes.items() if value]))


def infer_internal_multimodal_family(model_path: str, mmproj_path: str = "") -> tuple[str, str]:
    name = Path(str(model_path)).name.casefold()
    llava_tokens = ("llava", "bakllava")
    qwen_tokens = ("qwen-vl", "qwen2-vl", "qwen2.5-vl", "qwen25-vl", "qwen3-vl", "gliese")
    likely_text_only_tokens = (
        "instruct",
        "chat",
        "coder",
        "mistral",
        "llama",
        "phi",
        "gemma",
        "deepseek",
        "mixtral",
    )

    if any(token in name for token in llava_tokens):
        return FAMILY_LLAVA, ""
    if any(token in name for token in qwen_tokens):
        return FAMILY_QWEN_VL, ""
    has_qwen_lineage, has_qwen_multimodal_name = _looks_like_qwen_multimodal_name(name)
    has_matching_mmproj_evidence = _mmproj_matches_model(model_path, mmproj_path) if mmproj_path else False
    if has_qwen_lineage and (has_qwen_multimodal_name or has_matching_mmproj_evidence):
        return FAMILY_QWEN_VL, ""
    if has_qwen_lineage:
        if not has_qwen_multimodal_name and not has_matching_mmproj_evidence:
            return "", (
                "selected GGUF may be a Qwen-family model, but no multimodal naming evidence was found "
                "and no matching mmproj evidence was found. choose a qwen-vl/gliese/captioning-style GGUF "
                "or use its matching mmproj file."
            )
        if not has_qwen_multimodal_name:
            return "", (
                "selected GGUF may be a Qwen-family model, but no multimodal naming evidence was found. "
                "choose a qwen-vl/gliese/captioning-style GGUF for the internal VLM scanner."
            )
        return "", (
            "selected GGUF may be a Qwen-family model, but no matching mmproj evidence was found. "
            "select the matching mmproj file for this model."
        )

    if any(token in name for token in likely_text_only_tokens):
        return "", (
            "selected GGUF appears to be a text-only model, not a vision model. "
            "choose a llava-style or qwen-vl/gliese-style GGUF for the internal VLM scanner."
        )

    return "", (
        "selected GGUF does not look like a supported internal vision model. "
        "choose a llava-style or qwen-vl/gliese-style GGUF for the internal VLM scanner."
    )


def infer_internal_multimodal_family_from_path_hint(path_text: str) -> str:
    text = str(path_text).casefold()
    llava_tokens = ("llava", "bakllava")
    qwen_tokens = ("qwen-vl", "qwen2-vl", "qwen2.5-vl", "qwen25-vl", "qwen3-vl", "gliese")
    if any(token in text for token in llava_tokens):
        return FAMILY_LLAVA
    if any(token in text for token in qwen_tokens):
        return FAMILY_QWEN_VL
    _, has_qwen_multimodal_name = _looks_like_qwen_multimodal_name(text)
    if has_qwen_multimodal_name:
        return FAMILY_QWEN_VL
    return ""


def detect_chat_handler_support() -> GPMChatHandlerSupport:
    try:
        from llama_cpp import llama_chat_format  # type: ignore
    except Exception as exc:
        return GPMChatHandlerSupport(family_to_classes={}, import_error=str(exc))

    family_candidates: dict[str, tuple[str, ...]] = {
        FAMILY_LLAVA: ("Llava16ChatHandler", "Llava15ChatHandler"),
        # Qwen/Gliese support depends on the installed llama-cpp-python build.
        # We only declare support if one of these handlers is actually importable.
        FAMILY_QWEN_VL: (
            "Qwen3VLChatHandler",
            "Qwen25VLChatHandler",
            "Qwen2_5VLChatHandler",
            "Qwen2VLChatHandler",
            "QwenVLChatHandler",
        ),
    }

    found: dict[str, tuple[str, ...]] = {}
    for family, class_names in family_candidates.items():
        present = [name for name in class_names if hasattr(llama_chat_format, name)]
        if present:
            found[family] = tuple(present)
    return GPMChatHandlerSupport(family_to_classes=found, import_error="")


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


def _base_handler_kwargs(
    family: str,
    mmproj_text: str,
    image_max_tokens: int | None,
) -> dict[str, Any]:
    if family == FAMILY_QWEN_VL:
        kwargs: dict[str, Any] = {
            "clip_model_path": mmproj_text,
            "force_reasoning": False,
            "verbose": False,
        }
        if image_max_tokens is not None and int(image_max_tokens) > 0:
            kwargs["image_max_tokens"] = int(image_max_tokens)
        return kwargs

    return {"clip_model_path": mmproj_text}


def _build_handler_kwargs(
    handler_cls: type[Any],
    family: str,
    mmproj_path: Path,
    image_max_tokens: int | None,
) -> tuple[dict[str, Any], str]:
    mmproj_text = str(mmproj_path)
    base_kwargs = _base_handler_kwargs(family, mmproj_text, image_max_tokens)
    filtered = _filter_kwargs_for_callable(getattr(handler_cls, "__init__", handler_cls), base_kwargs)
    if not filtered:
        return {}, (
            f"{handler_cls.__name__} does not expose a compatible mmproj parameter; "
            "cannot wire mmproj safely for this model family."
        )
    if family != FAMILY_QWEN_VL and "clip_model_path" not in filtered:
        return {}, (
            f"{handler_cls.__name__} does not expose clip_model_path; "
            "cannot wire mmproj safely for this model family."
        )
    if "clip_model_path" not in filtered:
        return {}, (
            f"{handler_cls.__name__} init kwargs do not include clip_model_path after filtering; "
            "cannot wire mmproj safely for this model family."
        )
    return filtered, ""


def _build_handler_instance(
    handler_cls: type[Any],
    family: str,
    mmproj_path: Path,
    image_max_tokens: int | None,
) -> tuple[Any | None, str]:
    kwargs, kwargs_error = _build_handler_kwargs(handler_cls, family, mmproj_path, image_max_tokens)
    if kwargs_error:
        return None, kwargs_error
    try:
        return handler_cls(**kwargs), ""
    except Exception as exc:
        return None, f"{handler_cls.__name__} init failed: {exc}"


def _build_handler_instance_with_debug(
    handler_cls: type[Any],
    family: str,
    mmproj_path: Path,
    image_max_tokens: int | None,
) -> tuple[Any | None, str, dict[str, Any]]:
    # Keep debug payload concise and JSON-safe.
    kwargs, kwargs_error = _build_handler_kwargs(handler_cls, family, mmproj_path, image_max_tokens)
    if kwargs_error:
        return None, kwargs_error, {}
    try:
        return handler_cls(**kwargs), "", {
            "handler_constructor_kwargs": kwargs,
        }
    except Exception as exc:
        return None, f"{handler_cls.__name__} init failed: {exc}", {
            "handler_constructor_kwargs": kwargs,
        }


def resolve_internal_chat_handler(
    family: str,
    mmproj_path: str,
    image_max_tokens: int | None = None,
    debug_info: dict[str, Any] | None = None,
) -> tuple[Any | None, str, str]:
    support = detect_chat_handler_support()
    if support.import_error:
        return None, "", (
            "internal runtime requires llama-cpp-python with vision support "
            f"(failed import: {support.import_error}). Install/update llama-cpp-python for your platform."
        )

    if family not in {FAMILY_LLAVA, FAMILY_QWEN_VL}:
        return None, "", f"unsupported internal multimodal model family: {family}"

    class_names = support.family_to_classes.get(family, ())
    if debug_info is not None:
        debug_info["handler_candidate_order"] = list(class_names)
        debug_info["handler_first_attempted"] = class_names[0] if class_names else ""
    if not class_names:
        supported = ", ".join(support.available_families()) or "none"
        return None, "", (
            f"installed llama-cpp-python build does not support internal family '{family}'. "
            f"Available internal multimodal families in this environment: {supported}"
        )

    mmproj = Path(str(mmproj_path)).expanduser()
    if not mmproj.exists() or not mmproj.is_file():
        return None, "", f"internal mmproj file was not found: {mmproj}"

    from llama_cpp import llama_chat_format  # type: ignore

    errors: list[str] = []
    selected_index = -1
    for class_name in class_names:
        handler_cls = getattr(llama_chat_format, class_name, None)
        if handler_cls is None:
            continue
        if debug_info is None:
            handler, error = _build_handler_instance(
                handler_cls=handler_cls,
                family=family,
                mmproj_path=mmproj,
                image_max_tokens=image_max_tokens,
            )
            handler_debug: dict[str, Any] = {}
        else:
            handler, error, handler_debug = _build_handler_instance_with_debug(
                handler_cls=handler_cls,
                family=family,
                mmproj_path=mmproj,
                image_max_tokens=image_max_tokens,
            )
            if "handler_first_attempted" not in debug_info or not str(debug_info.get("handler_first_attempted", "")):
                debug_info["handler_first_attempted"] = class_name
            debug_info["handler_last_attempted"] = class_name
            if handler_debug:
                debug_info.update(handler_debug)
        if handler is not None:
            if class_name in class_names:
                selected_index = class_names.index(class_name)
            if debug_info is not None:
                debug_info["handler_selected"] = class_name
                debug_info["handler_selection_mode"] = (
                    "explicit_class_match" if selected_index == 0 else "fallback_class_preference_order"
                )
            return handler, class_name, ""
        errors.append(error)
        if debug_info is not None:
            debug_info["handler_last_error"] = error

    detail = "; ".join([item for item in errors if item]) or "no compatible handler class could be initialized"
    return None, "", (
        f"installed llama-cpp-python build exposes family '{family}' handlers but initialization failed: {detail}"
    )


def available_internal_families_text() -> str:
    support = detect_chat_handler_support()
    if support.import_error:
        return ""
    return ", ".join(support.available_families())
