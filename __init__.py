from .gpm_dependency_status import print_startup_diagnostics

try:
    print_startup_diagnostics()
except Exception as exc:
    print(f"[GPM startup] Dependency diagnostics failed: {exc}")

from .gpm_gallery_browser_node import GPMGalleryBrowser
from .gpm_prompt_combiner_node import GPMPromptCombiner
from .gpm_vlm_internal_diagnostics_node import GPMVLMInternalDiagnostics
from .gpm_vlm_scanner_internal_node import GPMVLMScannerInternal, GPMVLMScannerInternalAdvanced
from .gpm_vlm_scanner_node import GPMVLMScanner

NODE_CLASS_MAPPINGS = {
    "GPM Gallery Browser": GPMGalleryBrowser,
    "GPM Prompt Combiner": GPMPromptCombiner,
    "GPM VLM Scanner": GPMVLMScanner,
    "GPM VLM Scanner (Internal)": GPMVLMScannerInternal,
    "GPM VLM Scanner (Internal Advanced)": GPMVLMScannerInternalAdvanced,
    "GPM VLM Internal Diagnostics": GPMVLMInternalDiagnostics,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPM Gallery Browser": "GPM Gallery Browser",
    "GPM Prompt Combiner": "GPM Prompt Combiner",
    "GPM VLM Scanner": "GPM VLM Scanner",
    "GPM VLM Scanner (Internal)": "GPM VLM Scanner (Internal)",
    "GPM VLM Scanner (Internal Advanced)": "GPM VLM Scanner (Internal Advanced)",
    "GPM VLM Internal Diagnostics": "GPM VLM Internal Diagnostics",
}

WEB_DIRECTORY = "./web"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
