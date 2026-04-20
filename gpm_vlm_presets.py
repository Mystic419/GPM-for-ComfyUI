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
                "You are a prompt extraction assistant for SDXL workflows. "
                "Return strict JSON only with exactly two string keys: person_prompt and scene_prompt. "
                "If there is no clear visible main person/character/subject, person_prompt must be an empty string. "
                "If there is no clear scene/background/environment, scene_prompt must be an empty string. "
                "person_prompt: describe only a visible main person/character/subject. "
                "Do not include environment, room, architecture, street, furniture, decor, lighting, or background details in person_prompt. "
                "scene_prompt: describe only environment, background, lighting, and composition; do not include person traits when no clear person is present. "
                "Keep both outputs concise, comma-separated, generation-friendly, modular, and reusable. "
                "Avoid full sentences. Avoid quality tags. Avoid negative prompts. "
                "Avoid clutter props unless essential to the subject or scene."
            ),
            "ban_list": [],
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
                "You are a prompt extraction assistant for Pony-style prompting. "
                "Return strict JSON only with exactly two string keys: person_prompt and scene_prompt. "
                "If there is no clear visible main person/character/subject, person_prompt must be an empty string. "
                "If there is no clear scene/background/environment, scene_prompt must be an empty string. "
                "Use short descriptive tags suitable for Pony workflows. "
                "person_prompt: focus on a visible character's traits, outfit, pose, and expression only. "
                "Do not include environment, room, architecture, street, furniture, decor, lighting, or background details in person_prompt. "
                "scene_prompt: focus on setting, background, lighting, and composition only; do not include person traits when no clear person is present. "
                "Keep outputs concise and reusable. Avoid full sentences. "
                "Avoid quality tags. Avoid negative prompts. Avoid clutter props unless essential."
            ),
            "ban_list": [],
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
                "You are a prompt extraction assistant for natural-language prompting. "
                "Return strict JSON only with exactly two string keys: person_prompt and scene_prompt. "
                "If there is no clear visible main person/character/subject, person_prompt must be an empty string. "
                "If there is no clear scene/background/environment, scene_prompt must be an empty string. "
                "Use short natural-language phrases instead of tag soup. "
                "person_prompt: describe only a visible subject/person. "
                "Do not include environment, room, architecture, street, furniture, decor, lighting, or background details in person_prompt. "
                "scene_prompt: describe only setting, background, lighting, and composition; do not include person traits when no clear person is present. "
                "Keep outputs compact and reusable. Avoid full sentences where possible. "
                "Avoid quality tags. Avoid negative prompts. Avoid clutter props unless essential."
            ),
            "ban_list": [],
            "validated_model_name": RECOMMENDED_GGUF_MODEL_REPO,
            "is_builtin": True,
            "created_at": now,
            "updated_at": now,
        },
    ]


def _default_payload() -> dict[str, Any]:
    return {"version": 1, "presets": _default_builtin_presets()}


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
    return {
        "id": preset_id,
        "name": name,
        "family": family,
        "system_prompt": str(raw.get("system_prompt", "")).strip(),
        "ban_list": _normalize_ban_list(raw.get("ban_list", [])),
        "validated_model_name": str(raw.get("validated_model_name", "")).strip(),
        "is_builtin": bool(raw.get("is_builtin", False)),
        "created_at": created_at,
        "updated_at": updated_at,
    }


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
        existing["validated_model_name"] = str(builtin.get("validated_model_name", "")).strip()
        merged.append(existing)

    for preset in by_id.values():
        merged.append(preset)
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

        now = _now_iso_utc()
        new_preset = {
            "id": f"user-{uuid.uuid4().hex}",
            "name": cleaned_name,
            "family": normalized_family,
            "system_prompt": str(system_prompt or "").strip(),
            "ban_list": _normalize_ban_list(ban_list),
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
        )

    def update_user_preset(
        self,
        preset_id: str,
        name: str | None = None,
        family: str | None = None,
        system_prompt: str | None = None,
        ban_list: list[str] | str | None = None,
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
