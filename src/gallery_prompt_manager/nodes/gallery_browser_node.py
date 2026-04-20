from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from gallery_prompt_manager.core.gallery_browser_core import (
    enter_folder,
    get_selected_image_path,
    go_back,
    list_folder,
    load_sibling_prompts,
)


def _empty_image_tensor(width: int = 64, height: int = 64) -> torch.Tensor:
    return torch.zeros((1, height, width, 3), dtype=torch.float32)


def _load_image_tensor(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array)[None,]


class GPMGalleryBrowser:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "root_folder": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Root/start folder. Browsing cannot move above this folder.",
                    },
                ),
                "current_subfolder": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Subfolder under root_folder. Leave blank for root.",
                    },
                ),
                "action": (
                    ["refresh", "enter_folder", "back", "select_image"],
                    {"default": "refresh"},
                ),
                "entry_name": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Folder or image name in the current folder.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "person_prompt", "scene_prompt", "selected_image_path")
    FUNCTION = "run"
    CATEGORY = "GPM"

    def run(self, root_folder: str, current_subfolder: str, action: str, entry_name: str):
        image_tensor = _empty_image_tensor()
        person_prompt = ""
        scene_prompt = ""
        selected_image_path = ""

        try:
            if action == "back":
                current_subfolder = go_back(root_folder, current_subfolder)
            elif action == "enter_folder":
                current_subfolder = enter_folder(root_folder, current_subfolder, entry_name)

            listing = list_folder(root_folder, current_subfolder)
            status = "Ready"

            if action == "select_image":
                selected_path = get_selected_image_path(root_folder, listing.current_folder_rel, entry_name)
                if selected_path is not None:
                    image_tensor = _load_image_tensor(selected_path)
                    person_prompt, scene_prompt = load_sibling_prompts(selected_path)
                    selected_image_path = str(selected_path)
                    status = "Image selected"
                else:
                    status = "Image not found in current folder"

            ui = {
                "status": [status],
                "current_subfolder": [listing.current_folder_rel or "."],
                "entries": [listing.listing_text],
            }
        except Exception as exc:  # Keep prototype resilient for missing paths or bad input.
            ui = {
                "status": [f"Error: {exc}"],
                "current_subfolder": ["."],
                "entries": [""],
            }

        return {
            "ui": ui,
            "result": (image_tensor, person_prompt, scene_prompt, selected_image_path),
        }


NODE_CLASS_MAPPINGS = {
    "GPM Gallery Browser": GPMGalleryBrowser,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPM Gallery Browser": "GPM Gallery Browser",
}
