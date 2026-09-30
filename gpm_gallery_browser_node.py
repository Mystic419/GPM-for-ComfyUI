from __future__ import annotations

import json
import secrets
from pathlib import Path
from typing import Any

import torch
from aiohttp import web
from PIL import Image
from server import PromptServer

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
STATE_FILE_PATH = Path(__file__).with_name("gpm_gallery_state.json")
PROMPT_PROFILE_FIELDS: dict[str, tuple[str, str]] = {
    "SDXL": ("sdxl_person", "sdxl_scene"),
    "Pony": ("pony_person", "pony_scene"),
    "Natural Language": ("natural_person", "natural_scene"),
}
SELECTION_MANUAL = "MANUAL"
SELECTION_SEQUENTIAL = "SEQUENTIAL"
SELECTION_RANDOM = "RANDOM"
SELECTION_MODES = {SELECTION_MANUAL, SELECTION_SEQUENTIAL, SELECTION_RANDOM}

# Saved workflows from the first gallery-browser release used these values.
# Keep accepting them so upgrading the node does not silently change its mode.
LEGACY_SELECTION_MODES = {"OFF": SELECTION_MANUAL, "ON": SELECTION_RANDOM}


def _normalize_relpath(relpath: str) -> str:
    value = (relpath or "").replace("\\", "/").strip()
    if value in {"", "."}:
        return ""

    parts = [part for part in value.split("/") if part not in {"", "."}]
    cleaned: list[str] = []
    for part in parts:
        if part == "..":
            if cleaned:
                cleaned.pop()
            continue
        cleaned.append(part)
    return "/".join(cleaned)


def _resolve_root(root_folder: str) -> Path | None:
    if not root_folder or not root_folder.strip():
        return None

    root = Path(root_folder).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return None
    return root


def _safe_join_under_root(root: Path, relpath: str) -> Path | None:
    rel = _normalize_relpath(relpath)
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _validated_visible_rows(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = 3
    return max(1, min(6, parsed))


def _empty_state() -> dict[str, Any]:
    return {
        "root_folder": "",
        "current_subfolder": "",
        "selected_image_rel": "",
        "visible_rows": 3,
        "randomize_mode": SELECTION_MANUAL,
        "random_order": [],
        "random_index": 0,
    }


def _validated_state(
    root_folder: str,
    current_subfolder: str,
    selected_image_rel: str,
    visible_rows: Any,
    randomize_mode: Any = SELECTION_MANUAL,
    random_order: Any = None,
    random_index: Any = 0,
) -> dict[str, Any]:
    root = _resolve_root(root_folder)
    if root is None:
        print(
            "[GPM][state][validate] root invalid -> empty state",
            {
                "root_folder": root_folder,
                "current_subfolder": current_subfolder,
                "selected_image_rel": selected_image_rel,
                "visible_rows": visible_rows,
            },
        )
        state = _empty_state()
        state["visible_rows"] = _validated_visible_rows(visible_rows)
        return state

    current = _safe_join_under_root(root, current_subfolder)
    if current is None or not current.exists() or not current.is_dir():
        print(
            "[GPM][state][validate] current_subfolder invalid -> reset to root",
            {
                "root_folder": str(root),
                "current_subfolder": current_subfolder,
            },
        )
        current = root

    current_rel = ""
    if current != root:
        current_rel = current.relative_to(root).as_posix()

    selected_rel = _normalize_relpath(selected_image_rel)
    if selected_rel:
        selected_abs = _safe_join_under_root(root, selected_rel)
        if (
            selected_abs is None
            or not selected_abs.exists()
            or not selected_abs.is_file()
            or selected_abs.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS
        ):
            print(
                "[GPM][state][validate] selected image invalid -> clear",
                {
                    "root_folder": str(root),
                    "selected_image_rel": selected_image_rel,
                },
            )
            selected_rel = ""
        else:
            selected_parent_rel = ""
            if selected_abs.parent != root:
                selected_parent_rel = selected_abs.parent.relative_to(root).as_posix()
            if selected_parent_rel != current_rel:
                print(
                    "[GPM][state][validate] selected image outside current_subfolder -> clear",
                    {
                        "current_subfolder": current_rel,
                        "selected_image_rel": selected_rel,
                    },
                )
                selected_rel = ""

    normalized_order: list[str] = []
    if isinstance(random_order, list):
        for value in random_order:
            if not isinstance(value, str):
                continue
            rel_path = _normalize_relpath(value)
            if rel_path and rel_path not in normalized_order:
                normalized_order.append(rel_path)
    try:
        normalized_index = max(0, int(random_index))
    except (TypeError, ValueError):
        normalized_index = 0

    return {
        "root_folder": str(root),
        "current_subfolder": current_rel,
        "selected_image_rel": selected_rel,
        "visible_rows": _validated_visible_rows(visible_rows),
        "randomize_mode": _normalized_randomize_mode(randomize_mode),
        "random_order": normalized_order,
        "random_index": normalized_index,
    }


def _normalized_node_id(node_id: Any) -> str:
    if isinstance(node_id, (int, float)):
        if isinstance(node_id, float) and not node_id.is_integer():
            return ""
        return str(int(node_id))

    if isinstance(node_id, str):
        value = node_id.strip()
        return value

    return ""


def _load_state_map() -> dict[str, dict[str, Any]]:
    if not STATE_FILE_PATH.exists() or not STATE_FILE_PATH.is_file():
        return {}

    try:
        with STATE_FILE_PATH.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError):
        print("[GPM][state][load] failed to read/parse state file", {"path": str(STATE_FILE_PATH)})
        return {}

    if not isinstance(payload, dict):
        return {}

    state_map: dict[str, dict[str, Any]] = {}
    for raw_node_id, raw_state in payload.items():
        node_id = _normalized_node_id(raw_node_id)
        if not node_id or not isinstance(raw_state, dict):
            continue
        state_map[node_id] = _validated_state(
            str(raw_state.get("root_folder", "")),
            str(raw_state.get("current_subfolder", "")),
            str(raw_state.get("selected_image_rel", "")),
            raw_state.get("visible_rows", 3),
            raw_state.get("randomize_mode", SELECTION_MANUAL),
            raw_state.get("random_order", []),
            raw_state.get("random_index", 0),
        )
    return state_map


def _load_persisted_state(node_id: Any) -> dict[str, Any]:
    normalized_node_id = _normalized_node_id(node_id)
    if not normalized_node_id:
        print("[GPM][state][get] missing/invalid node_id -> empty state", {"node_id": node_id})
        return _empty_state()

    state_map = _load_state_map()
    state = state_map.get(normalized_node_id)
    if isinstance(state, dict):
        return state

    print(
        "[GPM][state][get] no state found -> empty state",
        {"node_id": normalized_node_id},
    )
    return _empty_state()


def _save_persisted_state(
    node_id: Any,
    root_folder: str,
    current_subfolder: str,
    selected_image_rel: str,
    visible_rows: Any,
    randomize_mode: Any = SELECTION_MANUAL,
    random_order: list[str] | None = None,
    random_index: int | None = None,
) -> dict[str, Any]:
    normalized_node_id = _normalized_node_id(node_id)
    normalized_mode = _normalized_randomize_mode(randomize_mode)
    state_map = _load_state_map()
    existing = state_map.get(normalized_node_id, _empty_state()) if normalized_node_id else _empty_state()
    context_changed = (
        str(existing.get("root_folder", "")) != str(_resolve_root(root_folder) or "")
        or str(existing.get("current_subfolder", "")) != _normalize_relpath(current_subfolder)
        or str(existing.get("randomize_mode", SELECTION_MANUAL)) != normalized_mode
    )
    if random_order is None:
        random_order = [] if context_changed else existing.get("random_order", [])
    if random_index is None:
        random_index = 0 if context_changed else existing.get("random_index", 0)
    state = _validated_state(
        root_folder,
        current_subfolder,
        selected_image_rel,
        visible_rows,
        normalized_mode,
        random_order,
        random_index,
    )
    if not normalized_node_id:
        print(
            "[GPM][state][post] missing/invalid node_id -> skip save",
            {"node_id": node_id, "state": state},
        )
        return state

    state_map[normalized_node_id] = state

    try:
        with STATE_FILE_PATH.open("w", encoding="utf-8") as file:
            json.dump(state_map, file, indent=2)
    except OSError:
        print("[GPM][state][save] failed to write state file", {"path": str(STATE_FILE_PATH)})
        pass
    return state


def _list_folder(root_folder: str, current_subfolder: str) -> dict[str, Any]:
    root = _resolve_root(root_folder)
    if root is None:
        return {
            "ok": False,
            "error": "Invalid root folder.",
            "root_folder": root_folder or "",
            "current_subfolder": "",
            "can_go_back": False,
            "items": [],
        }

    current_path = _safe_join_under_root(root, current_subfolder)
    if current_path is None or not current_path.exists() or not current_path.is_dir():
        current_path = root

    current_rel = ""
    if current_path != root:
        current_rel = current_path.relative_to(root).as_posix()

    folder_items: list[dict[str, str]] = []
    image_items: list[dict[str, str]] = []

    for child in current_path.iterdir():
        name = child.name
        rel_path = (Path(current_rel) / name).as_posix() if current_rel else name

        if child.is_dir():
            folder_items.append(
                {
                    "type": "folder",
                    "name": name,
                    "rel_path": rel_path,
                }
            )
            continue

        if child.is_file() and child.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            image_items.append(
                {
                    "type": "image",
                    "name": name,
                    "rel_path": rel_path,
                }
            )

    folder_items = sorted(folder_items, key=lambda item: item["name"].casefold())
    image_items = sorted(image_items, key=lambda item: item["name"].casefold())

    return {
        "ok": True,
        "error": "",
        "root_folder": str(root),
        "current_subfolder": current_rel,
        "can_go_back": current_path != root,
        "items": folder_items + image_items,
    }


def _empty_profile_prompt_payload() -> dict[str, str]:
    payload: dict[str, str] = {}
    for person_key, scene_key in PROMPT_PROFILE_FIELDS.values():
        payload[person_key] = ""
        payload[scene_key] = ""
    return payload


def _load_sibling_prompts_payload(image_abs_path: Path | None) -> dict[str, str]:
    if image_abs_path is None:
        return _empty_profile_prompt_payload()

    json_path = image_abs_path.with_suffix(".json")
    if not json_path.exists() or not json_path.is_file():
        return _empty_profile_prompt_payload()

    try:
        with json_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError):
        return _empty_profile_prompt_payload()

    if not isinstance(payload, dict):
        return _empty_profile_prompt_payload()

    prompts = _empty_profile_prompt_payload()
    for key in prompts:
        value = payload.get(key, "")
        prompts[key] = value if isinstance(value, str) else ""
    return prompts

def _load_sibling_prompts(image_abs_path: Path | None) -> tuple[str, str]:
    payload = _load_sibling_prompts_payload(image_abs_path)
    return payload.get("sdxl_person", ""), payload.get("sdxl_scene", "")


def _save_sibling_prompts_for_profile(
    node_id: str,
    click_token: str,
    root_folder: str,
    image_rel_path: str,
    prompt_profile: str,
    person_prompt: str,
    scene_prompt: str,
) -> dict[str, Any]:
    normalized_profile = _normalized_prompt_profile(prompt_profile)
    image_path = _resolve_image_path(root_folder, image_rel_path)
    if image_path is None:
        return {"ok": False, "error": "No selected image to save."}

    json_path = image_path.with_suffix(".json")
    existed = json_path.exists() and json_path.is_file()
    payload: dict[str, Any] = {}

    if existed:
        try:
            with json_path.open("r", encoding="utf-8") as file:
                loaded_payload = json.load(file)
        except (OSError, json.JSONDecodeError):
            return {"ok": False, "error": "Existing JSON is invalid and could not be updated."}
        if not isinstance(loaded_payload, dict):
            return {"ok": False, "error": "Existing JSON is not an object and could not be updated."}
        payload = loaded_payload

    person_key, scene_key = PROMPT_PROFILE_FIELDS[normalized_profile]
    payload[person_key] = person_prompt if isinstance(person_prompt, str) else ""
    payload[scene_key] = scene_prompt if isinstance(scene_prompt, str) else ""

    try:
        with json_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
    except OSError:
        return {"ok": False, "error": "Failed to write sibling JSON file."}

    print(
        "[GPM][save_json] saved prompts",
        {
            "click_token": click_token,
            "node_id": node_id,
            "selected_image_rel": _normalize_relpath(image_rel_path),
            "json_path": str(json_path),
            "prompt_profile": normalized_profile,
            "created": not existed,
            "updated": existed,
        },
    )
    return {
        "ok": True,
        "click_token": click_token,
        "node_id": node_id,
        "created": not existed,
        "updated": existed,
        "json_path": str(json_path),
        "prompt_profile": normalized_profile,
    }


def _normalized_prompt_profile(value: Any) -> str:
    text = value if isinstance(value, str) else ""
    return text if text in PROMPT_PROFILE_FIELDS else "SDXL"


def _normalized_randomize_mode(value: Any) -> str:
    text = value.strip().upper() if isinstance(value, str) else ""
    if text in LEGACY_SELECTION_MODES:
        return LEGACY_SELECTION_MODES[text]
    return text if text in SELECTION_MODES else SELECTION_MANUAL


def _visible_image_rels(items: list[dict[str, str]]) -> list[str]:
    return [
        item["rel_path"]
        for item in items
        if item.get("type") == "image" and isinstance(item.get("rel_path"), str) and item["rel_path"]
    ]


def _choose_sequential_image_rel(image_rels: list[str], selected_rel: str) -> str:
    if not image_rels:
        return ""
    try:
        return image_rels[(image_rels.index(selected_rel) + 1) % len(image_rels)]
    except ValueError:
        return image_rels[0]


def _choose_random_image_rel(
    image_rels: list[str], random_order: Any, random_index: Any, previous_rel: str
) -> tuple[str, list[str], int]:
    """Return the next random item without repeats until this folder's pool is exhausted."""
    if not image_rels:
        return "", [], 0

    available = set(image_rels)
    order = [rel for rel in random_order if isinstance(rel, str) and rel in available] if isinstance(random_order, list) else []
    try:
        index = max(0, int(random_index))
    except (TypeError, ValueError):
        index = 0

    if set(order) != available or len(order) != len(image_rels) or index >= len(order):
        order = list(image_rels)
        secrets.SystemRandom().shuffle(order)
        if len(order) > 1 and order[0] == previous_rel:
            order[0], order[1] = order[1], order[0]
        index = 0
        print("[GPM][random] started fresh non-repeating cycle", {"image_count": len(order)})

    selected_rel = order[index]
    return selected_rel, order, index + 1


def _resolve_image_path(root_folder: str, image_rel_path: str) -> Path | None:
    root = _resolve_root(root_folder)
    if root is None:
        return None

    image_abs = _safe_join_under_root(root, image_rel_path)
    if image_abs is None:
        return None
    if not image_abs.exists() or not image_abs.is_file():
        return None
    if image_abs.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        return None
    return image_abs


def _empty_image_tensor() -> torch.Tensor:
    return torch.zeros((1, 64, 64, 3), dtype=torch.float32)


def _load_image_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    storage = torch.ByteStorage.from_buffer(image.tobytes())
    tensor = torch.ByteTensor(storage).view(height, width, 3).to(dtype=torch.float32) / 255.0
    return tensor.unsqueeze(0)


class GPMGalleryBrowser:
    @classmethod
    def INPUT_TYPES(cls):
        persisted = _empty_state()
        return {
            "required": {
                "root_folder": (
                    "STRING",
                    {
                        "default": persisted.get("root_folder", ""),
                        "multiline": False,
                    },
                ),
                "current_subfolder": (
                    "STRING",
                    {
                        "default": persisted.get("current_subfolder", ""),
                        "multiline": False,
                    },
                ),
                "selected_image_rel": (
                    "STRING",
                    {
                        "default": persisted.get("selected_image_rel", ""),
                        "multiline": False,
                    },
                ),
                "visible_rows": (
                    "INT",
                    {
                        "default": _validated_visible_rows(persisted.get("visible_rows", 3)),
                        "min": 1,
                        "max": 6,
                        "step": 1,
                    },
                ),
                "person_prompt_text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
                "scene_prompt_text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
                "prompt_profile": (
                    "STRING",
                    {
                        "default": "SDXL",
                        "multiline": False,
                    },
                ),
                "randomize_mode": (
                    "STRING",
                    {
                        "default": SELECTION_MANUAL,
                        "multiline": False,
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def IS_CHANGED(
        cls,
        root_folder: str,
        current_subfolder: str,
        selected_image_rel: str,
        visible_rows: int,
        person_prompt_text: str,
        scene_prompt_text: str,
        prompt_profile: str,
        randomize_mode: str,
        unique_id: str = "",
    ):
        mode = _normalized_randomize_mode(randomize_mode)
        if mode in {SELECTION_SEQUENTIAL, SELECTION_RANDOM}:
            print("[GPM][is_changed] force rerun", {"mode": mode})
            return float("nan")

        fingerprint = (
            str(root_folder),
            str(current_subfolder),
            str(selected_image_rel),
            int(_validated_visible_rows(visible_rows)),
            str(person_prompt_text) if isinstance(person_prompt_text, str) else "",
            str(scene_prompt_text) if isinstance(scene_prompt_text, str) else "",
            _normalized_prompt_profile(prompt_profile),
            mode,
        )
        print("[GPM][is_changed] normal cache", {"mode": mode})
        return fingerprint

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "person_prompt", "scene_prompt")
    FUNCTION = "execute"
    CATEGORY = "GPM"

    def execute(
        self,
        root_folder: str,
        current_subfolder: str,
        selected_image_rel: str,
        visible_rows: int,
        person_prompt_text: str,
        scene_prompt_text: str,
        prompt_profile: str,
        randomize_mode: str,
        unique_id: str = "",
    ):
        image_tensor = _empty_image_tensor()
        person_prompt = person_prompt_text if isinstance(person_prompt_text, str) else ""
        scene_prompt = scene_prompt_text if isinstance(scene_prompt_text, str) else ""
        selected_rel = _normalize_relpath(selected_image_rel)
        normalized_prompt_profile = _normalized_prompt_profile(prompt_profile)
        normalized_randomize_mode = _normalized_randomize_mode(randomize_mode)
        print(
            "[GPM][execute] start",
            {
                "mode": normalized_randomize_mode,
                "current_subfolder": current_subfolder,
                "selected_image_rel": selected_rel,
            },
        )

        listing = _list_folder(root_folder, current_subfolder)
        listing_root_folder = str(listing.get("root_folder", ""))
        listing_current_subfolder = str(listing.get("current_subfolder", ""))
        listing_items = listing.get("items", [])
        if not isinstance(listing_items, list):
            listing_items = []

        persisted = _load_persisted_state(unique_id)
        random_order: list[str] = []
        random_index = 0
        same_folder = (
            persisted.get("root_folder") == listing_root_folder
            and persisted.get("current_subfolder") == listing_current_subfolder
            and persisted.get("randomize_mode") == normalized_randomize_mode
        )
        if same_folder:
            random_order = persisted.get("random_order", [])
            random_index = persisted.get("random_index", 0)

        # Queued executions use the workflow's original widget values.  Continue
        # from the per-node saved selection instead, so automation advances even
        # when the browser UI has not had a chance to write widget values back.
        previous_selected_rel = selected_rel
        if same_folder and isinstance(persisted.get("selected_image_rel"), str):
            previous_selected_rel = persisted["selected_image_rel"]

        if listing.get("ok") and normalized_randomize_mode != SELECTION_MANUAL:
            image_rels = _visible_image_rels(listing_items)
            if normalized_randomize_mode == SELECTION_SEQUENTIAL:
                selected_rel = _choose_sequential_image_rel(image_rels, previous_selected_rel)
                print("[GPM][selection] sequential", {"selected_image_rel": selected_rel})
            else:
                selected_rel, random_order, random_index = _choose_random_image_rel(
                    image_rels, random_order, random_index, previous_selected_rel
                )
                print("[GPM][selection] random", {"selected_image_rel": selected_rel})

            if selected_rel:
                prompt_payload = _load_sibling_prompts_payload(_resolve_image_path(listing_root_folder, selected_rel))
                profile_keys = PROMPT_PROFILE_FIELDS[normalized_prompt_profile]
                person_prompt = prompt_payload.get(profile_keys[0], "")
                scene_prompt = prompt_payload.get(profile_keys[1], "")
        else:
            print("[GPM][selection] manual", {"current_subfolder": listing_current_subfolder})

        if listing.get("ok"):
            _save_persisted_state(
                unique_id,
                listing_root_folder,
                listing_current_subfolder,
                selected_rel,
                visible_rows,
                normalized_randomize_mode,
                random_order,
                random_index,
            )

        image_path = _resolve_image_path(listing_root_folder or root_folder, selected_rel)
        if image_path is not None:
            try:
                image_tensor = _load_image_tensor(image_path)
            except Exception:
                image_tensor = _empty_image_tensor()

        return {
            "ui": {
                "gpm_root_folder": [listing_root_folder],
                "gpm_current_subfolder": [listing_current_subfolder],
                "gpm_items": [listing_items],
                "gpm_error": [listing.get("error", "")],
                "gpm_person_prompt": [person_prompt],
                "gpm_scene_prompt": [scene_prompt],
                "gpm_visible_rows": [_validated_visible_rows(visible_rows)],
                "gpm_selected_image_rel": [selected_rel],
                "gpm_prompt_profile": [normalized_prompt_profile],
                "gpm_randomize_mode": [normalized_randomize_mode],
            },
            "result": (image_tensor, person_prompt, scene_prompt),
        }


@PromptServer.instance.routes.get("/gpm/gallery/list")
async def gpm_gallery_list(request: web.Request):
    root_folder = request.rel_url.query.get("root_folder", "")
    current_subfolder = request.rel_url.query.get("current_subfolder", "")
    payload = _list_folder(root_folder, current_subfolder)
    return web.json_response(payload)


@PromptServer.instance.routes.get("/gpm/gallery/prompts")
async def gpm_gallery_prompts(request: web.Request):
    root_folder = request.rel_url.query.get("root_folder", "")
    image_rel_path = request.rel_url.query.get("image_rel_path", "")

    image_path = _resolve_image_path(root_folder, image_rel_path)
    prompts = _load_sibling_prompts_payload(image_path)

    response_payload: dict[str, Any] = {"ok": image_path is not None}
    response_payload.update(prompts)
    return web.json_response(response_payload)


@PromptServer.instance.routes.get("/gpm/gallery/image")
async def gpm_gallery_image(request: web.Request):
    root_folder = request.rel_url.query.get("root_folder", "")
    image_rel_path = request.rel_url.query.get("image_rel_path", "")

    image_path = _resolve_image_path(root_folder, image_rel_path)
    if image_path is None:
        return web.Response(status=404, text="Image not found")

    return web.FileResponse(image_path)


@PromptServer.instance.routes.get("/gpm/gallery/state")
async def gpm_gallery_state_get(request: web.Request):
    node_id = request.rel_url.query.get("node_id", "")
    return web.json_response(_load_persisted_state(node_id))


@PromptServer.instance.routes.post("/gpm/gallery/state")
async def gpm_gallery_state_post(request: web.Request):
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        payload = {}

    if not isinstance(payload, dict):
        payload = {}

    saved = _save_persisted_state(
        payload.get("node_id", ""),
        str(payload.get("root_folder", "")),
        str(payload.get("current_subfolder", "")),
        str(payload.get("selected_image_rel", "")),
        payload.get("visible_rows", 3),
        payload.get("randomize_mode", SELECTION_MANUAL),
    )
    return web.json_response(saved)


@PromptServer.instance.routes.post("/gpm/gallery/save_prompts")
async def gpm_gallery_save_prompts(request: web.Request):
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        payload = {}

    if not isinstance(payload, dict):
        payload = {}

    click_token = str(payload.get("click_token", ""))
    node_id = _normalized_node_id(payload.get("node_id", ""))
    if not node_id:
        print(
            "[GPM][save_json] skip save: missing/invalid node_id",
            {"click_token": click_token, "node_id": payload.get("node_id", "")},
        )
        return web.json_response({"ok": False, "error": "Missing node context for save.", "click_token": click_token, "node_id": ""})

    save_root_folder = str(payload.get("root_folder", ""))
    save_image_rel_path = str(payload.get("image_rel_path", ""))
    prompt_profile = str(payload.get("prompt_profile", "SDXL"))

    print(
        "[GPM][save_json] request",
        {
            "click_token": click_token,
            "node_id": node_id,
            "image_rel_path": _normalize_relpath(save_image_rel_path),
            "prompt_profile": _normalized_prompt_profile(prompt_profile),
        },
    )

    response_payload = _save_sibling_prompts_for_profile(
        node_id,
        click_token,
        save_root_folder,
        save_image_rel_path,
        prompt_profile,
        str(payload.get("person_prompt", "")),
        str(payload.get("scene_prompt", "")),
    )
    print(
        "[GPM][save_json] response",
        {
            "click_token": click_token,
            "node_id": node_id,
            "image_rel_path": _normalize_relpath(save_image_rel_path),
            "prompt_profile": _normalized_prompt_profile(prompt_profile),
            "json_path": response_payload.get("json_path", ""),
            "created": bool(response_payload.get("created", False)),
            "updated": bool(response_payload.get("updated", False)),
        },
    )
    return web.json_response(response_payload)








