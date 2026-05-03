from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .gpm_vlm_runtime_base import RECOMMENDED_GGUF_MODEL_REPO

PRESETS_FILE_PATH = Path(__file__).with_name("gpm_vlm_presets.json")

FAMILY_SDXL = "SDXL"
FAMILY_PONY = "Pony"
FAMILY_NATURAL = "Natural Language"
ALLOWED_FAMILIES = (FAMILY_SDXL, FAMILY_PONY, FAMILY_NATURAL)
USER_PRESET_IDS_BY_FAMILY = {
    FAMILY_SDXL: "sdxl_user",
    FAMILY_PONY: "pony_user",
    FAMILY_NATURAL: "natural_user",
}

GENERATION_TEMPERATURE_DEFAULT = 0.2
GENERATION_TOP_P_DEFAULT = 0.95
GENERATION_MAX_TOKENS_DEFAULT = 512


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_ban_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return []


def _default_builtin_presets() -> list[dict[str, Any]]:
    now = _now_iso_utc()
    return [
        {
            "id": "builtin-sdxl",
            "name": "SDXL",
            "family": FAMILY_SDXL,
            "system_prompt": (
                "You are a prompt extraction assistant for SDXL workflows using Qwen2.5-VL. "
                "Return strict JSON only with exactly two string keys: person_prompt and scene_prompt. "
                "If there is no clear visible main person/character/subject, person_prompt must be an empty string. "
                "If there is no clear scene/background/environment, scene_prompt must be an empty string. "
                "Describe only what is actually visible in the input image. Do not guess hidden details, unseen body parts, identity, age, profession, or story context. "
                "person_prompt: describe only a visible main person/character/subject with concrete visible details such as apparent gender presentation, hairstyle, clothing, accessories, pose, facial expression, camera distance, and view angle. "
                "Do not include environment, room, architecture, street, furniture, decor, lighting, or background details in person_prompt. "
                "scene_prompt: describe only environment, background, lighting, weather, time-of-day cues, composition, depth, and camera framing; do not include person traits when no clear person is present. "
                "Keep both outputs moderately detailed, comma-separated, generation-friendly, modular, and reusable. "
                "Prefer 12-40 descriptive tags/phrases per field when useful instead of very short outputs. "
                "Avoid full sentences. Avoid quality tags. Avoid negative prompts. "
                "Avoid clutter props unless essential to the subject or scene."
            ),
            "ban_list": [],
            "temperature": GENERATION_TEMPERATURE_DEFAULT,
            "top_p": GENERATION_TOP_P_DEFAULT,
            "max_tokens": GENERATION_MAX_TOKENS_DEFAULT,
            "validated_model_name": RECOMMENDED_GGUF_MODEL_REPO,
            "is_builtin": True,
            "created_at": now,
            "updated_at": now,
        },
        {
            "id": "builtin-pony",
            "name": "Pony",
            "family": FAMILY_PONY,
            "system_prompt": (
                "You are a prompt extraction assistant for Pony-style prompting using Qwen2.5-VL. "
                "Return strict JSON only with exactly two string keys: person_prompt and scene_prompt. "
                "If there is no clear visible main person/character/subject, person_prompt must be an empty string. "
                "If there is no clear scene/background/environment, scene_prompt must be an empty string. "
                "Describe only what is actually visible in the input image. Do not invent lore, character names, personality, or unseen details. "
                "Use Pony-friendly descriptive tags/phrases suitable for reuse. "
                "person_prompt: focus on a visible character's traits, outfit details, hair, accessories, pose, expression, and camera framing only. "
                "Do not include environment, room, architecture, street, furniture, decor, lighting, or background details in person_prompt. "
                "scene_prompt: focus on setting, background, lighting, and composition only; do not include person traits when no clear person is present. "
                "Keep outputs moderately detailed and reusable. Prefer rich tag groups over minimal one-liners. Avoid full sentences. "
                "Prefer 12-40 descriptive tags/phrases per field when useful. "
                "Avoid quality tags. Avoid negative prompts. Avoid clutter props unless essential."
            ),
            "ban_list": [],
            "temperature": GENERATION_TEMPERATURE_DEFAULT,
            "top_p": GENERATION_TOP_P_DEFAULT,
            "max_tokens": GENERATION_MAX_TOKENS_DEFAULT,
            "validated_model_name": RECOMMENDED_GGUF_MODEL_REPO,
            "is_builtin": True,
            "created_at": now,
            "updated_at": now,
        },
        {
            "id": "builtin-natural-language",
            "name": "Natural Language",
            "family": FAMILY_NATURAL,
            "system_prompt": (
                "You are a prompt extraction assistant for natural-language prompting using Qwen2.5-VL. "
                "Return strict JSON only with exactly two string keys: person_prompt and scene_prompt. "
                "If there is no clear visible main person/character/subject, person_prompt must be an empty string. "
                "If there is no clear scene/background/environment, scene_prompt must be an empty string. "
                "Describe only what is actually visible in the image. Do not hallucinate identity, biography, intentions, or unseen events. "
                "Use natural-language prompt fragments instead of dense tag soup. "
                "person_prompt: describe only a visible subject/person with concrete visible attributes, clothing, pose, expression, and framing. "
                "Do not include environment, room, architecture, street, furniture, decor, lighting, or background details in person_prompt. "
                "scene_prompt: describe only setting, background, lighting, and composition; do not include person traits when no clear person is present. "
                "Keep outputs moderately detailed and reusable, using short phrase clauses rather than long prose. "
                "Prefer 2-6 compact phrase clauses per field when useful. "
                "Avoid full narrative sentences where possible. "
                "Avoid quality tags. Avoid negative prompts. Avoid clutter props unless essential."
            ),
            "ban_list": [],
            "temperature": GENERATION_TEMPERATURE_DEFAULT,
            "top_p": GENERATION_TOP_P_DEFAULT,
            "max_tokens": GENERATION_MAX_TOKENS_DEFAULT,
            "validated_model_name": RECOMMENDED_GGUF_MODEL_REPO,
            "is_builtin": True,
            "created_at": now,
            "updated_at": now,
        },
    ]


def _default_payload() -> dict[str, Any]:
    return {"version": 1, "presets": _merge_with_required_builtins(_default_builtin_presets())}


def _normalize_preset(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None

    preset_id = str(raw.get("id", "")).strip()
    name = str(raw.get("name", "")).strip()
    family = str(raw.get("family", "")).strip()
    if not preset_id or not name or family not in ALLOWED_FAMILIES:
        return None

    created_at = str(raw.get("created_at", "")).strip() or _now_iso_utc()
    updated_at = str(raw.get("updated_at", "")).strip() or _now_iso_utc()
    extras = {str(k): v for k, v in raw.items() if str(k) not in {
        "id",
        "name",
        "family",
        "system_prompt",
        "ban_list",
        "validated_model_name",
        "is_builtin",
        "created_at",
        "updated_at",
        "temperature",
        "top_p",
        "max_tokens",
    }}
    temperature, top_p, max_tokens = get_preset_generation_settings(raw)
    return {
        "id": preset_id,
        "name": name,
        "family": family,
        "system_prompt": str(raw.get("system_prompt", "")).strip(),
        "ban_list": _normalize_ban_list(raw.get("ban_list", [])),
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "validated_model_name": str(raw.get("validated_model_name", "")).strip(),
        "is_builtin": bool(raw.get("is_builtin", False)),
        "created_at": created_at,
        "updated_at": updated_at,
        **extras,
    }


def _clamp_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed < minimum:
        return minimum
    if parsed > maximum:
        return maximum
    return parsed


def _clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if parsed < minimum:
        return minimum
    if parsed > maximum:
        return maximum
    return parsed


def get_preset_generation_settings(preset: dict[str, Any] | None) -> tuple[float, float, int]:
    data = preset if isinstance(preset, dict) else {}
    temperature = _clamp_float(
        data.get("temperature", GENERATION_TEMPERATURE_DEFAULT),
        GENERATION_TEMPERATURE_DEFAULT,
        0.0,
        2.0,
    )
    top_p = _clamp_float(
        data.get("top_p", GENERATION_TOP_P_DEFAULT),
        GENERATION_TOP_P_DEFAULT,
        0.0,
        1.0,
    )
    max_tokens = _clamp_int(
        data.get("max_tokens", GENERATION_MAX_TOKENS_DEFAULT),
        GENERATION_MAX_TOKENS_DEFAULT,
        32,
        2048,
    )
    return temperature, top_p, max_tokens


def _merge_with_required_builtins(presets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {preset["id"]: preset for preset in presets}
    merged: list[dict[str, Any]] = []
    for builtin in _default_builtin_presets():
        existing = by_id.pop(builtin["id"], None)
        if existing is None:
            merged.append(builtin)
            continue
        existing["is_builtin"] = True
        existing["family"] = builtin["family"]
        existing["name"] = builtin["name"]
        existing["system_prompt"] = builtin["system_prompt"]
        existing["ban_list"] = list(builtin["ban_list"])
        existing["temperature"] = builtin["temperature"]
        existing["top_p"] = builtin["top_p"]
        existing["max_tokens"] = builtin["max_tokens"]
        existing["validated_model_name"] = str(builtin.get("validated_model_name", "")).strip()
        merged.append(existing)

    for preset in by_id.values():
        merged.append(preset)

    by_id_merged = {str(item.get("id", "")): item for item in merged if isinstance(item, dict)}
    for family, user_id in USER_PRESET_IDS_BY_FAMILY.items():
        if user_id in by_id_merged:
            continue
        builtin_id = {
            FAMILY_SDXL: "builtin-sdxl",
            FAMILY_PONY: "builtin-pony",
            FAMILY_NATURAL: "builtin-natural-language",
        }[family]
        source = by_id_merged.get(builtin_id)
        if not isinstance(source, dict):
            continue
        now = _now_iso_utc()
        new_user = {
            "id": user_id,
            "name": f"{family} User",
            "family": family,
            "system_prompt": str(source.get("system_prompt", "")).strip(),
            "ban_list": _normalize_ban_list(source.get("ban_list", [])),
            "temperature": _clamp_float(source.get("temperature"), GENERATION_TEMPERATURE_DEFAULT, 0.0, 2.0),
            "top_p": _clamp_float(source.get("top_p"), GENERATION_TOP_P_DEFAULT, 0.0, 1.0),
            "max_tokens": _clamp_int(source.get("max_tokens"), GENERATION_MAX_TOKENS_DEFAULT, 32, 2048),
            "validated_model_name": str(source.get("validated_model_name", "")).strip(),
            "is_builtin": False,
            "created_at": now,
            "updated_at": now,
        }
        merged.append(new_user)
    return merged


class GPMVLMPresetStore:
    def __init__(self, presets_path: Path | None = None):
        self.presets_path = presets_path or PRESETS_FILE_PATH

    def _write_payload(self, payload: dict[str, Any]) -> None:
        self.presets_path.parent.mkdir(parents=True, exist_ok=True)
        with self.presets_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def _load_payload(self) -> dict[str, Any]:
        if not self.presets_path.exists():
            payload = _default_payload()
            self._write_payload(payload)
            return payload

        try:
            with self.presets_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            payload = _default_payload()
            self._write_payload(payload)
            return payload

        if not isinstance(payload, dict):
            payload = _default_payload()
            self._write_payload(payload)
            return payload

        raw_presets = payload.get("presets", [])
        if not isinstance(raw_presets, list):
            raw_presets = []

        seen_ids: set[str] = set()
        normalized: list[dict[str, Any]] = []
        for raw in raw_presets:
            preset = _normalize_preset(raw)
            if preset is None or preset["id"] in seen_ids:
                continue
            seen_ids.add(preset["id"])
            normalized.append(preset)

        repaired = {"version": 1, "presets": _merge_with_required_builtins(normalized)}
        if repaired != payload:
            self._write_payload(repaired)
        return repaired

    def list_presets(self) -> list[dict[str, Any]]:
        payload = self._load_payload()
        presets = payload.get("presets", [])
        if not isinstance(presets, list):
            return []
        return [dict(item) for item in presets if isinstance(item, dict)]

    def get_preset(self, preset_id: str) -> dict[str, Any] | None:
        key = str(preset_id).strip()
        if not key:
            return None
        for preset in self.list_presets():
            if preset.get("id") == key:
                return preset
        return None

    def create_user_preset(
        self,
        name: str,
        family: str,
        system_prompt: str,
        ban_list: list[str] | str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        normalized_family = str(family).strip()
        if normalized_family not in ALLOWED_FAMILIES:
            raise ValueError("family must be one of SDXL, Pony, Natural Language")

        cleaned_name = str(name).strip()
        if not cleaned_name:
            raise ValueError("name is required")

        payload = self._load_payload()
        presets = payload.get("presets", [])
        if not isinstance(presets, list):
            presets = []

        final_temperature = _clamp_float(
            temperature if temperature is not None else GENERATION_TEMPERATURE_DEFAULT,
            GENERATION_TEMPERATURE_DEFAULT,
            0.0,
            2.0,
        )
        final_top_p = _clamp_float(
            top_p if top_p is not None else GENERATION_TOP_P_DEFAULT,
            GENERATION_TOP_P_DEFAULT,
            0.0,
            1.0,
        )
        final_max_tokens = _clamp_int(
            max_tokens if max_tokens is not None else GENERATION_MAX_TOKENS_DEFAULT,
            GENERATION_MAX_TOKENS_DEFAULT,
            32,
            2048,
        )

        now = _now_iso_utc()
        new_preset = {
            "id": f"user-{uuid.uuid4().hex}",
            "name": cleaned_name,
            "family": normalized_family,
            "system_prompt": str(system_prompt or "").strip(),
            "ban_list": _normalize_ban_list(ban_list),
            "temperature": final_temperature,
            "top_p": final_top_p,
            "max_tokens": final_max_tokens,
            "validated_model_name": "",
            "is_builtin": False,
            "created_at": now,
            "updated_at": now,
        }
        presets.append(new_preset)
        payload["presets"] = presets
        self._write_payload(payload)
        return new_preset

    def duplicate_preset(self, preset_id: str, new_name: str | None = None) -> dict[str, Any]:
        source = self.get_preset(preset_id)
        if source is None:
            raise ValueError("preset not found")

        duplicate_name = str(new_name or f"{source.get('name', 'Preset')} Copy").strip()
        if not duplicate_name:
            duplicate_name = f"{source.get('name', 'Preset')} Copy"

        return self.create_user_preset(
            duplicate_name,
            str(source.get("family", FAMILY_SDXL)),
            str(source.get("system_prompt", "")),
            _normalize_ban_list(source.get("ban_list", [])),
            _clamp_float(source.get("temperature"), GENERATION_TEMPERATURE_DEFAULT, 0.0, 2.0),
            _clamp_float(source.get("top_p"), GENERATION_TOP_P_DEFAULT, 0.0, 1.0),
            _clamp_int(source.get("max_tokens"), GENERATION_MAX_TOKENS_DEFAULT, 32, 2048),
        )

    def update_user_preset(
        self,
        preset_id: str,
        name: str | None = None,
        family: str | None = None,
        system_prompt: str | None = None,
        ban_list: list[str] | str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        key = str(preset_id).strip()
        if not key:
            raise ValueError("preset id is required")

        payload = self._load_payload()
        presets = payload.get("presets", [])
        if not isinstance(presets, list):
            raise ValueError("invalid preset storage")

        for preset in presets:
            if not isinstance(preset, dict) or preset.get("id") != key:
                continue
            if bool(preset.get("is_builtin", False)):
                raise ValueError("builtin presets are read-only")

            if name is not None:
                cleaned_name = str(name).strip()
                if not cleaned_name:
                    raise ValueError("name cannot be empty")
                preset["name"] = cleaned_name
            if family is not None:
                normalized_family = str(family).strip()
                if normalized_family not in ALLOWED_FAMILIES:
                    raise ValueError("family must be one of SDXL, Pony, Natural Language")
                preset["family"] = normalized_family
            if system_prompt is not None:
                preset["system_prompt"] = str(system_prompt).strip()
            if ban_list is not None:
                preset["ban_list"] = _normalize_ban_list(ban_list)
            if temperature is not None:
                preset["temperature"] = _clamp_float(temperature, GENERATION_TEMPERATURE_DEFAULT, 0.0, 2.0)
            if top_p is not None:
                preset["top_p"] = _clamp_float(top_p, GENERATION_TOP_P_DEFAULT, 0.0, 1.0)
            if max_tokens is not None:
                preset["max_tokens"] = _clamp_int(max_tokens, GENERATION_MAX_TOKENS_DEFAULT, 32, 2048)
            if "validated_model_name" not in preset:
                preset["validated_model_name"] = ""

            preset["updated_at"] = _now_iso_utc()
            payload["presets"] = presets
            self._write_payload(payload)
            return dict(preset)

        raise ValueError("preset not found")

    def delete_user_preset(self, preset_id: str) -> bool:
        key = str(preset_id).strip()
        if not key:
            return False

        payload = self._load_payload()
        presets = payload.get("presets", [])
        if not isinstance(presets, list):
            return False

        kept: list[dict[str, Any]] = []
        deleted = False
        for preset in presets:
            if not isinstance(preset, dict):
                continue
            if preset.get("id") != key:
                kept.append(preset)
                continue
            if bool(preset.get("is_builtin", False)):
                kept.append(preset)
                continue
            deleted = True

        if deleted:
            payload["presets"] = kept
            self._write_payload(payload)
        return deleted
