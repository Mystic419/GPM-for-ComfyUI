from gallery_prompt_manager.nodes.gallery_browser_node import GPMGalleryBrowser
from gallery_prompt_manager.nodes.prompt_combiner_node import GPMPromptCombiner

NODE_CLASS_MAPPINGS = {
    "GPM Gallery Browser": GPMGalleryBrowser,
    "GPM Prompt Combiner": GPMPromptCombiner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPM Gallery Browser": "GPM Gallery Browser",
    "GPM Prompt Combiner": "GPM Prompt Combiner",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "GPMGalleryBrowser",
    "GPMPromptCombiner",
]
