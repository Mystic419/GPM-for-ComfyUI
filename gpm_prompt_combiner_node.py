from __future__ import annotations

import re


def _clean_part(value: str) -> str:
    if not isinstance(value, str):
        return ""

    text = value.strip()
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"(,\s*){2,}", ", ", text)
    return text.strip(" ,")


def _combine_prompt_parts(person_prompt: str, scene_prompt: str, lora_tags: str) -> str:
    parts = [
        _clean_part(person_prompt),
        _clean_part(scene_prompt),
        _clean_part(lora_tags),
    ]
    non_empty_parts = [part for part in parts if part]
    return ", ".join(non_empty_parts)


class GPMPromptCombiner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "person_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
                "scene_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
                "lora_tags": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("combined_prompt",)
    FUNCTION = "run"
    CATEGORY = "GPM"

    def run(self, person_prompt: str, scene_prompt: str, lora_tags: str):
        return (_combine_prompt_parts(person_prompt, scene_prompt, lora_tags),)
