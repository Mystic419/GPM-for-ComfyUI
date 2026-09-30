from __future__ import annotations

import json

from .gpm_vlm_prompt_node_utils import normalize_preset_payload
from .gpm_vlm_prompt_choice_utils import preset_choice_labels, preset_id_from_choice
from .gpm_vlm_prompt_presets import get_preset_by_id


def _preset_id_choices() -> list[str]:
    return preset_choice_labels()


class GPMVLMPromptLoaderNode:
    @classmethod
    def INPUT_TYPES(cls):
        preset_ids = _preset_id_choices()
        return {"required": {"selected_preset": (preset_ids, {"default": preset_ids[0]})}}

    RETURN_TYPES = ("GPM_VLM_PROMPT_PRESET", "STRING", "STRING")
    RETURN_NAMES = ("preset_bundle", "status_text", "preset_json")
    FUNCTION = "load"
    CATEGORY = "GPM / VLM"

    def load(self, selected_preset: str):
        preset_id = preset_id_from_choice(selected_preset)
        preset = get_preset_by_id(preset_id)
        if not isinstance(preset, dict):
            status = f"Preset not found: '{preset_id}'."
            bundle = {
                "ok": False,
                "status": status,
                "id": preset_id,
                "name": "",
                "family": "sdxl",
                "editable": False,
                "system_prompt": "",
                "use_ban_list": True,
                "ban_list": [],
                "version": 1,
            }
            return bundle, status, json.dumps(bundle, indent=2, ensure_ascii=False)

        status = f"Loaded preset '{preset_id}'."
        bundle = normalize_preset_payload(preset, status=status, ok=True)
        return bundle, status, json.dumps(bundle, indent=2, ensure_ascii=False)
