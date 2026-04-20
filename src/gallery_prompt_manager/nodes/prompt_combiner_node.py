from __future__ import annotations

from gallery_prompt_manager.core.prompt_combiner_core import combine_prompt_parts


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
        combined_prompt = combine_prompt_parts(person_prompt, scene_prompt, lora_tags)
        return (combined_prompt,)


NODE_CLASS_MAPPINGS = {
    "GPM Prompt Combiner": GPMPromptCombiner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPM Prompt Combiner": "GPM Prompt Combiner",
}
