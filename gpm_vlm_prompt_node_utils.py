from __future__ import annotations

from typing import Any

_FAMILY_NORMALIZATION = {
    "sdxl": "sdxl",
    "pony": "pony",
    "natural": "natural",
    "SDXL": "sdxl",
    "Pony": "pony",
    "Natural Language": "natural",
}


def normalize_family(value: Any) -> str:
    raw = str(value or "").strip()
    return _FAMILY_NORMALIZATION.get(raw, raw.lower() or "sdxl")


def normalize_editable(raw_preset: dict[str, Any]) -> bool:
    if "editable" in raw_preset:
        return bool(raw_preset.get("editable", True))
    return not bool(raw_preset.get("is_builtin", False))


def normalize_ban_list(raw_value: Any) -> list[str]:
    if not isinstance(raw_value, list):
        return []
    cleaned: list[str] = []
    for item in raw_value:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text:
            cleaned.append(text)
    return cleaned


def safe_version(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def normalize_preset_payload(raw_preset: dict[str, Any], *, status: str = "", ok: bool = True) -> dict[str, Any]:
    return {
        "ok": bool(ok),
        "status": str(status or "").strip(),
        "id": str(raw_preset.get("id", "")).strip(),
        "name": str(raw_preset.get("name", "")).strip(),
        "family": normalize_family(raw_preset.get("family", "")),
        "editable": normalize_editable(raw_preset),
        "system_prompt": str(raw_preset.get("system_prompt", "")).strip(),
        "use_ban_list": bool(raw_preset.get("use_ban_list", True)),
        "ban_list": normalize_ban_list(raw_preset.get("ban_list", [])),
        "version": safe_version(raw_preset.get("version", 1)),
    }
