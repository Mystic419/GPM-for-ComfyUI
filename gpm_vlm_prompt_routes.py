"""Small HTTP endpoints used by the prompt Saver's node-local Load button."""

from __future__ import annotations

import json

from aiohttp import web
from server import PromptServer

from .gpm_vlm_prompt_node_utils import normalize_preset_payload
from .gpm_vlm_prompt_presets import get_preset_by_id
from .gpm_vlm_prompt_saver_node import load_source_into_saver_state


def _load_preset(preset_id: object) -> dict:
    selected_id = str(preset_id or "").strip()
    preset = get_preset_by_id(selected_id)
    if not isinstance(preset, dict):
        return {"ok": False, "error": "preset_not_found", "preset_id": selected_id}
    return normalize_preset_payload(preset, status=f"Loaded preset '{selected_id}'.", ok=True)


@PromptServer.instance.routes.post("/gpm/vlm/prompt-saver/load")
async def gpm_vlm_prompt_saver_load(request: web.Request):
    """Load a selected preset into one Saver's server-side source state.

    This is deliberately read-only with respect to preset storage: it only
    records which source the user chose for the existing Saver workflow.
    """
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    node_id = payload.get("node_id", "")
    bundle = _load_preset(payload.get("preset_id", ""))
    if not bundle.get("ok"):
        return web.json_response(bundle, status=404)

    loaded = load_source_into_saver_state(node_id, bundle)
    if not loaded.get("ok"):
        return web.json_response(loaded, status=400)
    return web.json_response({"ok": True, "preset": loaded})
