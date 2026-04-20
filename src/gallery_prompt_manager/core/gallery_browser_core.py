from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class FolderListing:
    current_folder_abs: Path
    current_folder_rel: str
    can_go_back: bool
    folders: list[str]
    images: list[str]

    @property
    def listing_text(self) -> str:
        lines: list[str] = []
        for folder_name in self.folders:
            lines.append(f"[DIR] {folder_name}")
        for image_name in self.images:
            lines.append(f"[IMG] {image_name}")
        return "\n".join(lines)


def _normalize_relpath(relpath: str) -> str:
    clean = (relpath or "").strip().replace("\\", "/")
    if clean in {"", "."}:
        return ""
    return clean.strip("/")


def _safe_under_root(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def _sorted_names(paths: Iterable[Path]) -> list[str]:
    return sorted((p.name for p in paths), key=lambda name: name.casefold())


def ensure_root_folder(root_folder: str) -> Path:
    if not root_folder or not root_folder.strip():
        raise ValueError("root_folder is empty")

    root = Path(root_folder).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"root_folder does not exist or is not a folder: {root}")
    return root


def resolve_current_folder(root_folder: str, current_subfolder: str) -> Path:
    root = ensure_root_folder(root_folder)
    rel = _normalize_relpath(current_subfolder)
    candidate = (root / rel).resolve()
    if not candidate.exists() or not candidate.is_dir() or not _safe_under_root(root, candidate):
        return root
    return candidate


def relpath_from_root(root: Path, folder: Path) -> str:
    try:
        rel = folder.relative_to(root).as_posix()
    except ValueError:
        return ""
    return "" if rel == "." else rel


def list_folder(root_folder: str, current_subfolder: str) -> FolderListing:
    root = ensure_root_folder(root_folder)
    current = resolve_current_folder(root_folder, current_subfolder)

    children = list(current.iterdir())
    folders = [p for p in children if p.is_dir()]
    images = [p for p in children if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS]

    current_rel = relpath_from_root(root, current)

    return FolderListing(
        current_folder_abs=current,
        current_folder_rel=current_rel,
        can_go_back=current != root,
        folders=_sorted_names(folders),
        images=_sorted_names(images),
    )


def enter_folder(root_folder: str, current_subfolder: str, folder_name: str) -> str:
    folder_name = (folder_name or "").strip()
    if not folder_name:
        return list_folder(root_folder, current_subfolder).current_folder_rel

    current = resolve_current_folder(root_folder, current_subfolder)
    target = (current / folder_name).resolve()
    root = ensure_root_folder(root_folder)

    if target.exists() and target.is_dir() and _safe_under_root(root, target):
        return relpath_from_root(root, target)

    return relpath_from_root(root, current)


def go_back(root_folder: str, current_subfolder: str) -> str:
    root = ensure_root_folder(root_folder)
    current = resolve_current_folder(root_folder, current_subfolder)

    if current == root:
        return ""

    parent = current.parent.resolve()
    if not _safe_under_root(root, parent):
        return ""

    return relpath_from_root(root, parent)


def get_selected_image_path(root_folder: str, current_subfolder: str, image_name: str) -> Path | None:
    image_name = (image_name or "").strip()
    if not image_name:
        return None

    current = resolve_current_folder(root_folder, current_subfolder)
    root = ensure_root_folder(root_folder)
    target = (current / image_name).resolve()

    if not _safe_under_root(root, target):
        return None
    if not target.exists() or not target.is_file():
        return None
    if target.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        return None
    return target


def load_sibling_prompts(image_path: Path | None) -> tuple[str, str]:
    if image_path is None:
        return "", ""

    json_path = image_path.with_suffix(".json")
    if not json_path.exists() or not json_path.is_file():
        return "", ""

    try:
        with json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return "", ""

    if not isinstance(payload, dict):
        return "", ""

    person_prompt = payload.get("sdxl_person", "")
    scene_prompt = payload.get("sdxl_scene", "")

    if not isinstance(person_prompt, str):
        person_prompt = ""
    if not isinstance(scene_prompt, str):
        scene_prompt = ""

    return person_prompt, scene_prompt

