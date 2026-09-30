from __future__ import annotations

import json
import os
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

ALLOWED_FAMILIES = {"sdxl", "pony", "natural"}
_LAST_STATUS = ""

_NATURAL_SKIN_GUIDANCE = (
    "When skin is visible, describe natural, realistic details only when supported by the image: "
    "skin texture, pores, subtle tonal variation, freckles, blemishes, blush, natural highlights, or shadows. "
    "Do not use plastic, waxy, porcelain, airbrushed, doll-like, or overly smooth skin language unless that "
    "stylization is clearly visible. "
)

_VISIBLE_ANATOMY_GUIDANCE = (
    "For clearly adult subjects, describe visible anatomy accurately and plainly when relevant, including breasts, "
    "nipples or areolae, vulva or vaginal opening, penis, scrotum, buttocks, and body hair. Keep those details in "
    "person_prompt, never scene_prompt. Do not euphemize, omit, censor, or invent anatomy. Never infer age, sex, "
    "gender, identity, or anatomy hidden by pose, clothing, framing, or obstruction. If adult status is unclear, "
    "use only neutral, non-sexual visible body and clothing descriptions. "
)


def _set_status(message: str) -> None:
    global _LAST_STATUS
    _LAST_STATUS = str(message or "").strip()


def get_last_status() -> str:
    return _LAST_STATUS


def _builtin_presets() -> list[dict[str, Any]]:
    return [
        {
            "id": "builtin-sdxl",
            "name": "SDXL",
            "family": "sdxl",
            "editable": False,
            "system_prompt": (
                "You are a prompt extraction assistant for SDXL workflows. "
                "Return strict JSON only with exactly two string keys: person_prompt and scene_prompt. "
                "If there is no clear visible main person, person_prompt must be an empty string. "
                "If there is no clear scene/background, scene_prompt must be an empty string. "
                "Use concise comma-separated prompt fragments (not prose), describe only visible details, "
                "and keep person traits in person_prompt while keeping environment details in scene_prompt. "
                + _NATURAL_SKIN_GUIDANCE
                + _VISIBLE_ANATOMY_GUIDANCE
                + "Avoid quality/rating/source tags and avoid negative prompts."
            ),
            "use_ban_list": True,
            "ban_list": [],
            "version": 1,
        },
        {
            "id": "builtin-pony",
            "name": "Pony",
            "family": "pony",
            "editable": False,
            "system_prompt": (
                "You are a prompt extraction assistant for Pony-style prompting. "
                "Return strict JSON only with exactly two string keys: person_prompt and scene_prompt. "
                "Use concise, lowercase, comma-separated visual tags (not prose). "
                "Keep character/body/clothing/hair/expression tags in person_prompt and "
                "background/environment/lighting/prop tags in scene_prompt. "
                + _NATURAL_SKIN_GUIDANCE
                + _VISIBLE_ANATOMY_GUIDANCE
                + "Describe only visible content and avoid quality/rating/source tags and negative prompts."
            ),
            "use_ban_list": True,
            "ban_list": [],
            "version": 1,
        },
        {
            "id": "builtin-natural-language",
            "name": "Natural Language",
            "family": "natural",
            "editable": False,
            "system_prompt": (
                "You are a prompt extraction assistant for natural-language prompting. "
                "Return strict JSON only with exactly two string keys: person_prompt and scene_prompt. "
                "Use short descriptive caption-style clauses (not tag soup), describe only visible details, "
                "keep subject traits in person_prompt and environment details in scene_prompt, and avoid "
                + _NATURAL_SKIN_GUIDANCE
                + _VISIBLE_ANATOMY_GUIDANCE
                + "quality/rating/source tags and negative prompts."
            ),
            "use_ban_list": True,
            "ban_list": [],
            "version": 1,
        },
    ]


def _default_payload() -> dict[str, Any]:
    return {"version": 1, "presets": []}


def _resolve_user_presets_path() -> Path:
    try:
        import folder_paths  # type: ignore

        user_dir = getattr(folder_paths, "user_directory", None)
        if user_dir:
            return Path(str(user_dir)).expanduser().resolve() / "default" / "GPM" / "vlm_prompt_presets.json"
    except Exception:
        pass
    fallback_root = Path(os.getenv("GPM_USER_DATA_DIR", "")).expanduser() if os.getenv("GPM_USER_DATA_DIR") else None
    if fallback_root:
        return fallback_root.resolve() / "vlm_prompt_presets.json"
    return Path(__file__).resolve().parent / "gpm_vlm_prompt_presets.user.json"


def _read_user_payload() -> dict[str, Any]:
    path = _resolve_user_presets_path()
    if not path.exists():
        _set_status("")
        return _default_payload()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _set_status(f"warning: user preset JSON is invalid; using built-ins only ({exc})")
        return _default_payload()
    if not isinstance(payload, dict):
        _set_status("warning: user preset JSON root is not an object; using built-ins only")
        return _default_payload()
    _set_status("")
    return payload


def _normalize_ban_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text:
            cleaned.append(text)
    return cleaned


def _normalize_user_preset(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    preset = dict(raw)
    preset["id"] = str(preset.get("id", "")).strip()
    preset["name"] = str(preset.get("name", "")).strip()
    preset["family"] = str(preset.get("family", "")).strip().lower()
    preset["editable"] = bool(preset.get("editable", True))
    preset["system_prompt"] = str(preset.get("system_prompt", "")).strip()
    preset["use_ban_list"] = bool(preset.get("use_ban_list", True))
    preset["ban_list"] = _normalize_ban_list(preset.get("ban_list", []))
    try:
        preset["version"] = int(preset.get("version", 1))
    except (TypeError, ValueError):
        preset["version"] = 1
    if not preset["id"] or not preset["name"]:
        return None
    if preset["family"] not in ALLOWED_FAMILIES:
        return None
    return preset


def _write_user_payload(payload: dict[str, Any]) -> None:
    path = _resolve_user_presets_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(path)


def get_builtin_presets() -> list[dict[str, Any]]:
    return deepcopy(_builtin_presets())


def get_user_presets() -> list[dict[str, Any]]:
    payload = _read_user_payload()
    raw_presets = payload.get("presets", [])
    if not isinstance(raw_presets, list):
        _set_status("warning: user preset JSON has invalid presets list; using built-ins only")
        return []
    seen: set[str] = set()
    builtins = {item["id"] for item in _builtin_presets()}
    cleaned: list[dict[str, Any]] = []
    for raw in raw_presets:
        normalized = _normalize_user_preset(raw)
        if normalized is None:
            continue
        preset_id = str(normalized.get("id", ""))
        if preset_id in seen or preset_id in builtins:
            continue
        normalized["editable"] = True
        seen.add(preset_id)
        cleaned.append(normalized)
    return deepcopy(cleaned)


def get_all_presets() -> list[dict[str, Any]]:
    return get_builtin_presets() + get_user_presets()


def get_preset_by_id(preset_id: str) -> dict[str, Any] | None:
    key = str(preset_id or "").strip()
    if not key:
        return None
    for preset in get_all_presets():
        if str(preset.get("id", "")) == key:
            return deepcopy(preset)
    return None


def validate_preset(
    preset_dict: dict[str, Any],
    *,
    allow_existing_user_id: bool = False,
) -> tuple[bool, str]:
    if not isinstance(preset_dict, dict):
        return False, "preset must be an object"
    required = {"id", "name", "family", "editable", "system_prompt", "use_ban_list", "ban_list", "version"}
    missing = sorted(item for item in required if item not in preset_dict)
    if missing:
        return False, f"missing required fields: {', '.join(missing)}"
    preset_id = str(preset_dict.get("id", "")).strip()
    if not preset_id:
        return False, "id is required"
    family = str(preset_dict.get("family", "")).strip().lower()
    if family not in ALLOWED_FAMILIES:
        return False, "family must be one of: sdxl, pony, natural"
    ban_list = preset_dict.get("ban_list")
    if not isinstance(ban_list, list):
        return False, "ban_list must be a list[str]"
    if any(not isinstance(item, str) for item in ban_list):
        return False, "ban_list must be a list[str]"
    duplicate_count = sum(1 for item in get_all_presets() if str(item.get("id", "")) == preset_id)
    if duplicate_count > 0:
        existing = get_preset_by_id(preset_id)
        if existing and bool(existing.get("editable", False)):
            if allow_existing_user_id:
                return True, ""
            return False, f"duplicate preset id: {preset_id}"
        if existing and not bool(existing.get("editable", True)):
            return False, f"cannot overwrite built-in preset id: {preset_id}"
    return True, ""


def save_user_preset(preset_dict: dict[str, Any]) -> dict[str, Any]:
    preset = dict(preset_dict)
    preset["id"] = str(preset.get("id", "")).strip() or f"user-{uuid.uuid4().hex}"
    preset["family"] = str(preset.get("family", "")).strip().lower()
    preset["editable"] = True
    preset["use_ban_list"] = bool(preset.get("use_ban_list", True))
    raw_ban_list = preset.get("ban_list", [])
    if not isinstance(raw_ban_list, list):
        raise ValueError("ban_list must be a list[str]")
    if any(not isinstance(item, str) for item in raw_ban_list):
        raise ValueError("ban_list must be a list[str]")
    preset["ban_list"] = _normalize_ban_list(raw_ban_list)
    try:
        preset["version"] = int(preset.get("version", 1))
    except (TypeError, ValueError):
        preset["version"] = 1

    builtin_ids = {item["id"] for item in _builtin_presets()}
    if preset["id"] in builtin_ids:
        raise ValueError(f"cannot overwrite built-in preset: {preset['id']}")

    valid, error = validate_preset(preset, allow_existing_user_id=True)
    if not valid:
        raise ValueError(error)

    payload = _read_user_payload()
    raw_presets = payload.get("presets", [])
    if not isinstance(raw_presets, list):
        raw_presets = []

    kept: list[dict[str, Any]] = []
    replaced = False
    for raw in raw_presets:
        normalized = _normalize_user_preset(raw)
        if normalized is None:
            continue
        if normalized["id"] == preset["id"]:
            kept.append(dict(preset))
            replaced = True
        else:
            kept.append(normalized)
    if not replaced:
        kept.append(dict(preset))
    payload = {"version": 1, "presets": kept}
    _write_user_payload(payload)
    return deepcopy(preset)


def clone_preset(source_preset_id: str, new_name: str | None = None) -> dict[str, Any]:
    source = get_preset_by_id(source_preset_id)
    if source is None:
        raise ValueError(f"preset not found: {source_preset_id}")
    clone = dict(source)
    clone["id"] = f"user-{uuid.uuid4().hex}"
    clone["editable"] = True
    clone["name"] = str(new_name or f"{source.get('name', 'Preset')} Copy").strip() or "Preset Copy"
    return save_user_preset(clone)


def delete_user_preset(preset_id: str) -> bool:
    key = str(preset_id or "").strip()
    if not key:
        return False
    builtin_ids = {item["id"] for item in _builtin_presets()}
    if key in builtin_ids:
        raise ValueError(f"cannot delete built-in preset: {key}")

    payload = _read_user_payload()
    raw_presets = payload.get("presets", [])
    if not isinstance(raw_presets, list):
        return False

    kept: list[dict[str, Any]] = []
    deleted = False
    for raw in raw_presets:
        normalized = _normalize_user_preset(raw)
        if normalized is None:
            continue
        if normalized["id"] == key:
            deleted = True
            continue
        kept.append(normalized)
    if not deleted:
        return False

    _write_user_payload({"version": 1, "presets": kept})
    return True
