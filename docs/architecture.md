# Architecture

## Overview
Gallery Prompt Manager is a ComfyUI custom node pack with three target responsibilities:
1. scan images into structured prompt metadata
2. browse image folders inside ComfyUI
3. combine selected prompt parts into a final prompt string

Current implemented scope includes:
- `GPM Gallery Browser` backend prototype
- `GPM Prompt Combiner`
- first `GPM VLM Scanner` implementation with GGUF-only backend mode
- `GPM VLM Internal Diagnostics` helper node for internal runtime environment checks

## Major components

### 1. Scanner layer
Status: implemented (v2, GGUF-only with API runtime plus in-process internal runtime).

Owns:
- recursive image discovery under a selected root folder
- preset-selected family mapping to fixed sidecar keys:
  - `SDXL` -> `sdxl_person` / `sdxl_scene`
  - `Pony` -> `pony_person` / `pony_scene`
  - `Natural Language` -> `natural_person` / `natural_scene`
- skip-existing or overwrite-family behavior
- preserving unrelated sidecar JSON fields while updating only selected family fields
- optional lightweight per-scan metadata under `gpm_meta`
- shared runtime abstraction for generation while keeping one scan loop:
  - API runtime (current external/local OpenAI-compatible endpoint path)
  - internal runtime (in-process `llama-cpp-python` vision path)
    - Qwen-VL-family models use a dedicated constructor profile (Qwen handler kwargs + filtered Llama kwargs)
    - explicit family support gate blocks unvalidated internal multimodal families before scanning
    - currently approved internal family for scan correctness: `Qwen2.5-VL`
  - internal model discovery helper for ComfyUI model folders (`models/llm`, `models/llm/GGUF`, `models/GGUF`)

Implemented files:
- `gpm_vlm_scanner_node.py`
- `gpm_vlm_internal_diagnostics_node.py`
- `gpm_vlm_backend.py`
- `gpm_vlm_presets.py`
- `gpm_vlm_runtime_base.py`
- `gpm_vlm_runtime_api.py`
- `gpm_vlm_runtime_internal.py`
- `gpm_vlm_model_discovery.py`

### 2. Metadata layer
Status: implemented for browser read path and scanner update path.

Owns:
- locating sibling `.json` files for selected images
- reading and validating JSON fields
- applying empty-string defaults for missing/invalid prompt fields
- preserving unrelated JSON fields on updates to selected family slots
- keeping global preset recipes in `gpm_vlm_presets.json` (not duplicated per image)

Implemented files:
- `src/gallery_prompt_manager/core/gallery_browser_core.py`

### 3. Gallery browser layer
Status: implemented as backend prototype.

Owns (implemented now):
- root/start folder selection
- one-folder-at-a-time listing
- folders-first then images sorting (alphabetical)
- entering folders
- going back to parent folder
- preventing navigation above root
- image selection in current folder
- sibling JSON prompt loading for selected image
- outputting image + prompt strings
- persisting browser UI state per node instance using ComfyUI node id keys

Owns (not implemented yet):
- save-back to JSON controls

Implemented files:
- `src/gallery_prompt_manager/nodes/gallery_browser_node.py`
- `src/gallery_prompt_manager/core/gallery_browser_core.py`

### 4. Prompt combiner layer
Status: implemented as v1 prototype.

Owns (implemented now):
- ordered merge in this sequence: `person_prompt` output, `scene_prompt` output, `lora_tags`
- ignores blank inputs
- joins non-empty parts with `", "`
- trims whitespace and normalizes comma spacing

Owns (not implemented yet):
- prompt-family style logic
- ban-list/model-family controls

Implemented files:
- `src/gallery_prompt_manager/nodes/prompt_combiner_node.py`
- `src/gallery_prompt_manager/core/prompt_combiner_core.py`

## Current node contract

### `GPM Gallery Browser`
Inputs:
- `root_folder` (STRING)
- `current_subfolder` (STRING)
- `action` (`refresh` | `enter_folder` | `back` | `select_image`)
- `entry_name` (STRING)

Outputs:
- `image`
- `person_prompt`
- `scene_prompt`
- `selected_image_path`

Notes:
- if sibling JSON is missing/invalid, prompts return empty strings
- non-image files and `.json` files are excluded from listing
- browsing is constrained to the chosen root folder

### `GPM Prompt Combiner`
Inputs:
- `person_prompt` (STRING)
- `scene_prompt` (STRING)
- `lora_tags` (STRING)

Outputs:
- `combined_prompt` (STRING)

Notes:
- output order is always person -> scene -> lora
- blank parts are dropped cleanly
- separator is `", "`

### `GPM VLM Internal Diagnostics`
Inputs:
- `model_name` (GGUF model choice)
- `mmproj_name` (mmproj choice)

Outputs:
- `summary_json` (STRING)
- `status_text` (STRING)

Notes:
- diagnostics-only node; does not load models
- reuses model/mmproj discovery and family inference logic from internal runtime helpers

### Internal scanner debug mode
Notes:
- `GPM VLM Scanner (Internal)` and `GPM VLM Scanner (Internal Advanced)` include optional `debug_mode` (`OFF`/`ON`).
- When `debug_mode=ON` and internal startup fails, `summary_json` includes concise startup diagnostics (llama-cpp version, inferred family, handler selection details, resolved paths, file sizes, and constructor exception text).
- Per-image sidecar metadata contract:
  - `gpm_meta.vlm_scan` is minimal and user-facing: `family`, `preset_id`, `backend`, `runtime`, `model`, `status`, `scanned_at`.
  - internal runtime debug details are written only when `debug_mode=ON`, under `gpm_meta.vlm_scan_debug`.
  - per-image debug trace includes support-gating fields: `detected_model_family`, `selected_chat_handler`, `family_support_status`, `support_reason`.

## Boundaries
- scanning logic is not mixed into browser code
- prompt combination logic is not mixed into browser code
- JSON parsing/navigation logic lives in plain helper functions for testability
- ComfyUI node wrappers are thin and delegate to helper/core modules

## Data flow (implemented)
1. User runs `GPM VLM Scanner` with a preset id and root folder.
2. Scanner resolves preset family, scans images recursively, and writes only that family's fixed sidecar keys.
3. User selects an image in `GPM Gallery Browser` and loads sidecar prompt fields for the currently selected profile family.
4. User optionally edits text and enters `lora_tags` in `GPM Prompt Combiner`.
5. Combiner outputs one normalized `combined_prompt` string.

## Planned JSON contract
Minimum compatible sidecar:

```json
{
  "sdxl_scene": "...",
  "sdxl_person": "..."
}
```

## Important files and folders
- `gpm_vlm_scanner_node.py` -> ComfyUI-facing VLM scanner node
- `gpm_vlm_backend.py` -> recursive scan orchestration + runtime selection + sidecar write logic
- `gpm_vlm_runtime_base.py` -> shared runtime interface/constants
- `gpm_vlm_runtime_api.py` -> OpenAI-compatible GGUF request runtime
- `gpm_vlm_runtime_internal.py` -> in-process GGUF runtime for internal VLM scanning
- `gpm_vlm_model_discovery.py` -> ComfyUI GGUF/mmproj file discovery and auto-pairing logic
- `gpm_vlm_presets.py` -> global preset storage and CRUD helper logic
- `src/gallery_prompt_manager/nodes/gallery_browser_node.py` -> ComfyUI-facing browser node
- `src/gallery_prompt_manager/core/gallery_browser_core.py` -> folder browsing + JSON helper logic
- `src/gallery_prompt_manager/nodes/prompt_combiner_node.py` -> ComfyUI-facing combiner node
- `src/gallery_prompt_manager/core/prompt_combiner_core.py` -> prompt merge/cleanup helper logic
- `tests/test_gallery_browser_core.py` -> browser core behavior tests
- `tests/test_prompt_combiner_core.py` -> combiner core behavior tests

## Notes
Keep this document updated when node contracts, package layout, or UI behavior changes.



