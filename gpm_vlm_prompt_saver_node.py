from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .gpm_vlm_prompt_node_utils import normalize_preset_payload
from .gpm_vlm_prompt_choice_utils import preset_choice_labels
from .gpm_vlm_prompt_presets import delete_user_preset, save_user_preset

STATE_FILE_PATH = Path(__file__).with_name("gpm_vlm_prompt_saver_state.json")


def _safe_json_string(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def _ban_list_lines_to_list(raw_text: str) -> list[str]:
    items: list[str] = []
    for line in str(raw_text or "").splitlines():
        term = line.strip()
        if term:
            items.append(term)
    return items


def _normalized_node_id(unique_id: Any) -> str:
    if isinstance(unique_id, (int, float)):
        if isinstance(unique_id, float) and not unique_id.is_integer():
            return "__default__"
        return str(int(unique_id))
    if isinstance(unique_id, str):
        cleaned = unique_id.strip()
        return cleaned if cleaned else "__default__"
    return "__default__"


def _load_state_map() -> dict[str, dict[str, Any]]:
    if not STATE_FILE_PATH.exists() or not STATE_FILE_PATH.is_file():
        return {}
    try:
        payload = json.loads(STATE_FILE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    state_map: dict[str, dict[str, Any]] = {}
    for node_id, node_state in payload.items():
        if isinstance(node_id, str) and isinstance(node_state, dict):
            state_map[node_id] = dict(node_state)
    return state_map


def _save_state_map(state_map: dict[str, dict[str, Any]]) -> None:
    STATE_FILE_PATH.write_text(_safe_json_string(state_map), encoding="utf-8")


def _get_state(unique_id: Any) -> dict[str, Any]:
    node_id = _normalized_node_id(unique_id)
    return dict(_load_state_map().get(node_id, {}))


def _set_state(unique_id: Any, state: dict[str, Any]) -> None:
    node_id = _normalized_node_id(unique_id)
    state_map = _load_state_map()
    state_map[node_id] = dict(state)
    _save_state_map(state_map)


def _clear_state(unique_id: Any) -> None:
    node_id = _normalized_node_id(unique_id)
    state_map = _load_state_map()
    if node_id in state_map:
        del state_map[node_id]
        _save_state_map(state_map)


def _state_from_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "loaded_source_id": str(bundle.get("id", "")).strip(),
        "loaded_source_name": str(bundle.get("name", "")).strip(),
        "loaded_source_family": str(bundle.get("family", "sdxl")).strip() or "sdxl",
        "loaded_source_editable": bool(bundle.get("editable", False)),
        "loaded_source_system_prompt": str(bundle.get("system_prompt", "")).strip(),
        "loaded_source_use_ban_list": bool(bundle.get("use_ban_list", False)),
        "loaded_source_ban_list": list(bundle.get("ban_list", []))
        if isinstance(bundle.get("ban_list", []), list)
        else [],
        "loaded_source_version": int(bundle.get("version", 1) or 1),
        "loaded_source_json": _safe_json_string(bundle),
    }


def load_source_into_saver_state(unique_id: Any, source_preset: Any) -> dict[str, Any]:
    """Normalize a preset bundle and make it this Saver node's explicit source.

    The frontend Load button uses this same helper through a small read-only
    preset endpoint.  Keeping the state write here makes that path equivalent
    to the existing queued ``LOAD FROM SOURCE`` action.
    """
    if not isinstance(source_preset, dict):
        return {
            "ok": False,
            "status": "Cannot load source: source_preset is missing.",
            "error": "source_missing",
        }

    bundle = normalize_preset_payload(
        source_preset,
        status=str(source_preset.get("status", "")).strip(),
        ok=bool(source_preset.get("ok", True)),
    )
    if not bundle["ok"]:
        return bundle

    _set_state(unique_id, _state_from_bundle(bundle))
    return bundle


class GPMVLMPromptSaverNode:
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        # This selector is intentionally separate from the Loader noodle.  It
        # powers the node-local Load button, which is a frontend proof of
        # concept and does not queue or execute the graph.
        load_preset_ids = preset_choice_labels()
        return {
            "required": {
                "action": (
                    [
                        "LOAD FROM SOURCE",
                        "SAVE AS NEW USER PRESET",
                        "UPDATE EXISTING USER PRESET",
                        "DELETE USER PRESET",
                    ],
                    {"default": "LOAD FROM SOURCE"},
                ),
                "load_preset": (load_preset_ids, {"default": load_preset_ids[0]}),
                "preset_name": ("STRING", {"default": ""}),
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "use_ban_list": (["KEEP", "OFF", "ON"], {"default": "KEEP"}),
                "ban_list": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                # Retains Loader -> Saver compatibility while allowing the
                # node-local Load button to establish the source on its own.
                "source_preset": ("GPM_VLM_PROMPT_PRESET", {"forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("status_text", "saved_preset_id", "preset_json")
    FUNCTION = "run"
    CATEGORY = "GPM / VLM"

    def run(
        self,
        action: str,
        source_preset: Any = None,
        preset_name: str = "",
        system_prompt: str = "",
        use_ban_list: str = "KEEP",
        ban_list: str = "",
        unique_id: Any = "",
        load_preset: str = "",
    ):
        state = _get_state(unique_id)
        action_key = str(action or "").strip()
        source_ok = isinstance(source_preset, dict) and bool(source_preset.get("ok", True))
        source_id = str(source_preset.get("id", "")).strip() if isinstance(source_preset, dict) else ""
        print(f"[GPM Prompt Saver backend] action={action_key} source_ok={source_ok} source_id={source_id}")
        name_input = str(preset_name or "").strip()
        prompt_input = str(system_prompt or "").strip()
        use_ban_choice = str(use_ban_list or "").strip().upper() or "KEEP"
        terms = _ban_list_lines_to_list(ban_list)

        def _out(status: str, saved_id: str, payload: Any, ui: dict[str, Any] | None = None):
            if ui:
                return {
                    "ui": {key: [value] for key, value in ui.items()},
                    "result": (status, saved_id, _safe_json_string(payload)),
                }
            return status, saved_id, _safe_json_string(payload)

        try:
            if action_key == "LOAD FROM SOURCE":
                if not isinstance(source_preset, dict):
                    status = "Cannot load source: source_preset is missing."
                    payload = {"ok": False, "error": "source_missing"}
                    return _out(status, "", payload)
                bundle = load_source_into_saver_state(unique_id, source_preset)
                if not bundle["ok"]:
                    status = str(bundle.get("status", "")).strip() or "Cannot load source: bundle is not ok."
                    return _out(status, "", bundle)
                status = (
                    f"Loaded source '{bundle['id']}'. UI payload emitted. "
                    "Edit Saver fields, then run a save/update/delete action."
                )
                payload = {
                    "ok": True,
                    "action": "LOAD FROM SOURCE",
                    "loaded_source": bundle,
                    "loaded_id": bundle["id"],
                    "loaded_name": bundle["name"],
                    "loaded_system_prompt": bundle["system_prompt"],
                    "loaded_prompt_length": len(bundle["system_prompt"]),
                    "loaded_use_ban_list": bool(bundle["use_ban_list"]),
                    "loaded_ban_list": list(bundle["ban_list"]),
                    "loaded_ban_list_count": len(bundle["ban_list"]),
                    "ui_payload_emitted": True,
                }
                print(f"[GPM Prompt Saver backend] LOAD emitted ui payload for {bundle['id']}")
                return _out(
                    status,
                    bundle["id"],
                    payload,
                    ui={
                        "gpm_vlm_prompt_saver_loaded_name": bundle["name"],
                        "gpm_vlm_prompt_saver_loaded_system_prompt": bundle["system_prompt"],
                        "gpm_vlm_prompt_saver_loaded_use_ban_list": "ON" if bundle["use_ban_list"] else "OFF",
                        "gpm_vlm_prompt_saver_loaded_ban_list": "\n".join(bundle["ban_list"]),
                        "gpm_vlm_prompt_saver_loaded_id": bundle["id"],
                    },
                )

            loaded_id = str(state.get("loaded_source_id", "")).strip()
            loaded_name = str(state.get("loaded_source_name", "")).strip()
            loaded_family = str(state.get("loaded_source_family", "sdxl")).strip() or "sdxl"
            loaded_editable = bool(state.get("loaded_source_editable", False))
            loaded_prompt = str(state.get("loaded_source_system_prompt", "")).strip()
            loaded_use_ban = bool(state.get("loaded_source_use_ban_list", False))
            loaded_ban = list(state.get("loaded_source_ban_list", [])) if isinstance(state.get("loaded_source_ban_list", []), list) else []
            loaded_version = int(state.get("loaded_source_version", 1) or 1)

            if action_key == "SAVE AS NEW USER PRESET":
                print(f"[GPM Prompt Saver backend] SAVE/UPDATE/DELETE action executed: {action_key}")
                resolved_name = name_input or (f"{loaded_name} Copy" if loaded_name else "User Preset")
                resolved_prompt = prompt_input or loaded_prompt
                if not resolved_prompt:
                    return _out(
                        "Cannot save preset: system_prompt is blank and no loaded source prompt is available.",
                        "",
                        {"ok": False, "error": "blank_system_prompt"},
                    )
                if use_ban_choice == "ON":
                    resolved_use_ban = True
                elif use_ban_choice == "OFF":
                    resolved_use_ban = False
                else:
                    resolved_use_ban = loaded_use_ban
                resolved_ban = terms if terms else loaded_ban
                draft = {
                    "id": "",
                    "name": resolved_name,
                    "family": loaded_family,
                    "editable": True,
                    "system_prompt": resolved_prompt,
                    "use_ban_list": resolved_use_ban,
                    "ban_list": resolved_ban,
                    "version": loaded_version or 1,
                }
                saved = save_user_preset(draft)
                saved_id = str(saved.get("id", "")).strip()
                if loaded_id:
                    status = f"Saved new user preset '{saved_id}' from loaded source '{loaded_id}'."
                else:
                    status = (
                        f"Saved new user preset '{saved_id}' with default family 'sdxl' "
                        "(no loaded source was found)."
                    )
                return _out(status, saved_id, saved)

            if action_key == "UPDATE EXISTING USER PRESET":
                print(f"[GPM Prompt Saver backend] SAVE/UPDATE/DELETE action executed: {action_key}")
                if not loaded_id:
                    return _out(
                        "Cannot update preset: no source has been loaded. Run LOAD FROM SOURCE first.",
                        "",
                        {"ok": False, "error": "missing_loaded_source"},
                    )
                if not loaded_editable:
                    return _out(
                        f"Cannot update built-in preset '{loaded_id}'.",
                        loaded_id,
                        {"ok": False, "error": "builtin_read_only", "preset_id": loaded_id},
                    )
                resolved_name = name_input or loaded_name or "User Preset"
                resolved_prompt = prompt_input or loaded_prompt
                if use_ban_choice == "ON":
                    resolved_use_ban = True
                elif use_ban_choice == "OFF":
                    resolved_use_ban = False
                else:
                    resolved_use_ban = loaded_use_ban
                resolved_ban = terms if terms else loaded_ban
                draft = {
                    "id": loaded_id,
                    "name": resolved_name,
                    "family": loaded_family,
                    "editable": True,
                    "system_prompt": resolved_prompt,
                    "use_ban_list": resolved_use_ban,
                    "ban_list": resolved_ban,
                    "version": loaded_version or 1,
                }
                saved = save_user_preset(draft)
                return _out(f"Updated user preset '{loaded_id}'.", loaded_id, saved)

            if action_key == "DELETE USER PRESET":
                print(f"[GPM Prompt Saver backend] SAVE/UPDATE/DELETE action executed: {action_key}")
                if not loaded_id:
                    return _out(
                        "Cannot delete preset: no source has been loaded. Run LOAD FROM SOURCE first.",
                        "",
                        {"ok": False, "deleted": False, "error": "missing_loaded_source"},
                    )
                if not loaded_editable:
                    return _out(
                        f"Cannot delete built-in preset '{loaded_id}'.",
                        loaded_id,
                        {"ok": False, "deleted": False, "error": "builtin_read_only", "preset_id": loaded_id},
                    )
                deleted = delete_user_preset(loaded_id)
                if deleted:
                    _clear_state(unique_id)
                payload = {"ok": bool(deleted), "deleted": bool(deleted), "preset_id": loaded_id}
                status = f"Deleted user preset '{loaded_id}'." if deleted else f"User preset '{loaded_id}' was not found."
                return _out(status, loaded_id, payload)

            return _out(
                f"Unknown action '{action_key}'.",
                "",
                {"ok": False, "error": "unknown_action", "action": action_key},
            )
        except Exception as exc:
            return _out(f"Action failed: {exc}", "", {"ok": False, "error": str(exc)})
