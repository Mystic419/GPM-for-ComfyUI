"""Stable preset-ID resolution with readable ComfyUI dropdown labels."""

from __future__ import annotations

from typing import Any

from .gpm_vlm_prompt_presets import get_all_presets, get_preset_by_id


def preset_choice_pairs() -> list[tuple[str, str]]:
    raw_pairs: list[tuple[str, str]] = []
    for preset in get_all_presets():
        if not isinstance(preset, dict):
            continue
        preset_id = str(preset.get("id", "")).strip()
        if not preset_id:
            continue
        name = str(preset.get("name", "")).strip() or preset_id
        raw_pairs.append((preset_id, name))

    label_counts: dict[str, int] = {}
    for _, name in raw_pairs:
        label_counts[name.casefold()] = label_counts.get(name.casefold(), 0) + 1

    used_labels: set[str] = set()
    resolved: list[tuple[str, str]] = []
    for preset_id, name in raw_pairs:
        label = name
        if label_counts[label.casefold()] > 1 or label in used_labels:
            label = f"{name} [{preset_id}]"
        used_labels.add(label)
        resolved.append((preset_id, label))
    return resolved or [("builtin-sdxl", "SDXL")]


def preset_choice_labels() -> list[str]:
    return [label for _, label in preset_choice_pairs()]


def preset_id_from_choice(value: Any) -> str:
    selected = str(value or "").strip()
    if not selected:
        return ""
    # Keep saved workflows and API callers that already use stable IDs valid.
    if isinstance(get_preset_by_id(selected), dict):
        return selected
    for preset_id, label in preset_choice_pairs():
        if selected == label:
            return preset_id
    return selected
