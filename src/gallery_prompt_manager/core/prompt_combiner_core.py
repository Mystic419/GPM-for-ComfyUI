from __future__ import annotations

import re


def _clean_part(value: str) -> str:
    if not isinstance(value, str):
        return ""

    text = value.strip()
    if not text:
        return ""

    # Collapse all whitespace to avoid awkward spacing across multiline input.
    text = re.sub(r"\s+", " ", text)
    # Remove comma-space repetition around separators.
    text = re.sub(r"\s*,\s*", ", ", text)
    # Collapse repeated commas that can appear in user-edited text.
    text = re.sub(r"(,\s*){2,}", ", ", text)
    return text.strip(" ,")


def combine_prompt_parts(person_prompt: str, scene_prompt: str, lora_tags: str) -> str:
    parts = [
        _clean_part(person_prompt),
        _clean_part(scene_prompt),
        _clean_part(lora_tags),
    ]
    non_empty_parts = [part for part in parts if part]
    return ", ".join(non_empty_parts)
