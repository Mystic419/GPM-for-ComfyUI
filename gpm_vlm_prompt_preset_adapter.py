from __future__ import annotations

from copy import deepcopy
from typing import Any

from .gpm_vlm_prompt_presets import get_all_presets, get_last_status, get_preset_by_id

_FAMILY_TO_PRESET_FAMILY = {
    "SDXL": "sdxl",
    "Pony": "pony",
    "Natural Language": "natural",
}

_PRESET_FAMILY_TO_RUNTIME = {value: key for key, value in _FAMILY_TO_PRESET_FAMILY.items()}

_DEFAULT_PRESET_BY_FAMILY = {
    "sdxl": "builtin-sdxl",
    "pony": "builtin-pony",
    "natural": "builtin-natural-language",
}


def _normalize_family_for_lookup(family: str | None) -> str:
    raw = str(family or "").strip()
    if raw in _DEFAULT_PRESET_BY_FAMILY:
        return raw
    return _FAMILY_TO_PRESET_FAMILY.get(raw, "sdxl")


def get_default_preset_id_for_family(family: str | None) -> str:
    return _DEFAULT_PRESET_BY_FAMILY[_normalize_family_for_lookup(family)]


def _coerce_runtime_preset(raw: dict[str, Any]) -> dict[str, Any]:
    preset_family = str(raw.get("family", "")).strip().lower()
    runtime_family = _PRESET_FAMILY_TO_RUNTIME.get(preset_family, "SDXL")
    return {
        "id": str(raw.get("id", "")).strip(),
        "name": str(raw.get("name", "")).strip() or "Preset",
        "family": runtime_family,
        "system_prompt": str(raw.get("system_prompt", "")).strip(),
        "ban_list": list(raw.get("ban_list", [])) if isinstance(raw.get("ban_list", []), list) else [],
        "use_ban_list": bool(raw.get("use_ban_list", False)),
    }


def _append_ban_list_instruction(system_prompt: str, preset: dict[str, Any]) -> str:
    base = str(system_prompt or "").strip()
    use_ban_list = bool(preset.get("use_ban_list", False))
    raw_ban_list = preset.get("ban_list", [])
    if not use_ban_list or not isinstance(raw_ban_list, list):
        return base
    terms = [str(item).strip() for item in raw_ban_list if str(item).strip()]
    if not terms:
        return base
    instruction = (
        "Avoid using these terms unless they are explicitly visible and necessary: "
        + ", ".join(terms)
        + "."
    )
    return f"{base}\n\n{instruction}".strip() if base else instruction


def list_vlm_prompt_preset_options(family: str | None = None) -> list[dict[str, str]]:
    requested_family = _normalize_family_for_lookup(family) if family else ""
    options: list[dict[str, str]] = []
    for preset in get_all_presets():
        if not isinstance(preset, dict):
            continue
        preset_id = str(preset.get("id", "")).strip()
        if not preset_id:
            continue
        preset_family = str(preset.get("family", "")).strip().lower()
        if requested_family and preset_family != requested_family:
            continue
        options.append(
            {
                "id": preset_id,
                "name": str(preset.get("name", "")).strip() or preset_id,
                "editable": "true" if bool(preset.get("editable", False)) else "false",
            }
        )
    return options


def get_vlm_prompt_preset(preset_id: str, fallback_family: str | None = None) -> tuple[dict[str, Any], str]:
    requested_id = str(preset_id or "").strip()
    fallback_id = get_default_preset_id_for_family(fallback_family)
    status_messages: list[str] = []

    preset_raw: dict[str, Any] | None = None
    if requested_id:
        candidate = get_preset_by_id(requested_id)
        if isinstance(candidate, dict):
            preset_raw = candidate
        else:
            status_messages.append(f"warning: preset '{requested_id}' not found; using '{fallback_id}'")
    if preset_raw is None:
        candidate = get_preset_by_id(fallback_id)
        if isinstance(candidate, dict):
            preset_raw = candidate
        else:
            # Built-ins should always exist; final hard fallback guards runtime.
            preset_raw = {
                "id": fallback_id,
                "name": fallback_id,
                "family": _normalize_family_for_lookup(fallback_family),
                "system_prompt": "",
                "use_ban_list": False,
                "ban_list": [],
            }
            status_messages.append(
                f"warning: fallback preset '{fallback_id}' was unavailable; using empty system prompt"
            )

    runtime_preset = _coerce_runtime_preset(preset_raw)
    runtime_preset["system_prompt"] = _append_ban_list_instruction(
        runtime_preset.get("system_prompt", ""),
        preset_raw,
    )

    backend_status = str(get_last_status() or "").strip()
    if backend_status:
        status_messages.append(backend_status)
    return deepcopy(runtime_preset), " | ".join(part for part in status_messages if part)
