# Gallery Prompt Manager

Gallery Prompt Manager is a ComfyUI custom node pack for browsing image folders as prompt assets, scanning images into sibling JSON metadata, and combining prompt halves for reuse.

## Status
Private prototype in progress.

Implemented now:
- `GPM Gallery Browser` v1 backend prototype (folder navigation + image selection + sibling JSON load)
- `GPM Prompt Combiner` v1 (person + scene + optional LoRA tags -> one clean prompt string)
- `GPM VLM Scanner` v1 (recursive scan + fixed-family sidecar writes via preset-selected family)
- `GPM VLM Scanner (Internal)` v2 (subprocess-isolated internal runtime with ComfyUI model-folder dropdown UX)
- `GPM VLM Scanner (Internal Advanced)` v1 (same subprocess-isolated runtime with manual tuning controls)
- `GPM VLM Internal Diagnostics` v1 (environment/status helper for internal GGUF multimodal support)
- `GPM VLM Prompt Loader` v2 (selects source preset and emits normalized `preset_bundle`)
- `GPM VLM Prompt Saver` v2 (stateful editable preset node with explicit load/save/update/delete modes)
- Global VLM preset storage with built-in read-only defaults (`SDXL`, `Pony`, `Natural Language`)
- Internal scanner preset resolution wired to backend preset manager (`gpm_vlm_prompt_presets.py`) with safe built-in fallback

Not implemented yet:
- clickable thumbnail frontend
- advanced prompt versioning workflows

## Current nodes

### `GPM Gallery Browser`
Purpose:
- browse a folder tree under a chosen root folder
- enter folders and go back to parent folder
- select one image file
- load sibling JSON fields (`sdxl_person`, `sdxl_scene`) if present

Supported image files:
- `.jpg`
- `.jpeg`
- `.png`
- `.webp`
- `.bmp`

Ignored in browser listing:
- non-image files
- `.json` sidecars

JSON contract (v1):

```json
{
  "sdxl_person": "...",
  "sdxl_scene": "..."
}
```

Behavior on missing/invalid JSON:
- image output still works
- `person_prompt` output returns `""` when `sdxl_person` is missing/invalid
- `scene_prompt` output returns `""` when `sdxl_scene` is missing/invalid

Persistence behavior:
- each `GPM Gallery Browser` node instance persists its own `root_folder`, `current_subfolder`, `selected_image_rel`, `visible_rows`, and selection mode using the ComfyUI node id

### `GPM Prompt Combiner`
Purpose:
- combine `person_prompt`, `scene_prompt`, and `lora_tags` in that order
- ignore empty inputs
- join non-empty parts with `", "`
- trim and normalize whitespace/comma spacing to avoid awkward separators

Output:
- `combined_prompt`

### `GPM VLM Scanner`
Purpose:
- recursively scan image files under a root folder
- run GGUF VLM inference using a selected preset id
- map preset family to fixed sidecar keys:
  - `SDXL` -> `sdxl_person` / `sdxl_scene`
  - `Pony` -> `pony_person` / `pony_scene`
  - `Natural Language` -> `natural_person` / `natural_scene`
- preserve unrelated JSON fields when writing family fields
- support `SKIP_EXISTING` (default) and `OVERWRITE_FAMILY`
- writes minimal scan metadata to `gpm_meta.vlm_scan`:
  - `family`
  - `preset_id`
  - `backend`
  - `runtime`
  - `model`
  - `status`
  - `scanned_at`
- when internal scanner `debug_mode=ON`, verbose runtime/debug metadata is written to `gpm_meta.vlm_scan_debug`
  - plus per-image trace under `gpm_meta.vlm_scan_debug_trace`:
    - `source_image_filename`
    - `source_image_full_path`
    - `source_image_sha256`
    - `output_json_full_path`
    - `model_prompt_sent`
    - `raw_model_response`
    - `parsed_person_prompt`
    - `parsed_scene_prompt`
    - `detected_model_family`
    - `selected_chat_handler`
    - `family_support_status`
    - `support_reason`
  - debug guard: if a response looks like generic family/living-room text for a likely different image type (for example storefront/cafe filename hints), scans in debug mode will warn and skip overwriting existing JSON for that image

### `GPM VLM Scanner (Internal)`
Purpose:
- run the same scan orchestration as the API scanner through `runtime_mode=internal`
- load GGUF VLM + mmproj in an isolated subprocess worker via `llama-cpp-python`
- discover model files from ComfyUI model folders and expose dropdowns (`model_name`, `mmproj_name`)
- support `mmproj_name=(auto)` matching when one clear candidate exists
- use an explicit internal family support gate before scan execution:
  - internal scan correctness is currently verified for `Qwen2.5-VL` only
  - unverified families (including Gliese/Qwen3.x and other unvalidated multimodal families) are blocked with a clear startup error instead of scanning
- use a dedicated Qwen-VL runtime path (Qwen handler + filtered llama constructor kwargs)
  - multimodal request image payload remains family-aware (`qwen_vl` uses object-style `image_url`)
- optional `debug_mode=ON` emits concise startup compatibility diagnostics in `summary_json` when internal startup fails
- worker exits after each scan, which is the primary VRAM release mechanism
- `keep_model_loaded` is internal-only and always ON during a scan run (not a user-facing widget)
- preset control:
  - `prompt_preset` (defaults to `builtin-sdxl`)
  - options include SDXL, Pony, Natural Language, and user presets from backend preset manager
  - selected preset system prompt is used for scan requests
- scanner dropdowns show preset names; saved workflow IDs remain accepted for backward compatibility
  - invalid/missing preset ids safely fall back to `builtin-sdxl`

### `GPM VLM Scanner (Internal Advanced)`
Purpose:
- same subprocess-isolated scanner flow as the basic internal node
- exposes advanced runtime controls (`n_ctx`, `n_gpu_layers`, `temperature`, `top_p`, `max_tokens`, `threads`, `batch_size`)
- keeps the same no-executable-path UX as the basic internal node
- includes optional `debug_mode` toggle with the same startup diagnostics behavior as the basic internal node
- family preset controls:
  - `sdxl_preset` (default `builtin-sdxl`)
  - `pony_preset` (default `builtin-pony`)
  - `natural_preset` (default `builtin-natural-language`)
  - family toggles: `scan_sdxl`, `scan_pony`, `scan_natural`
  - each enabled family runs with its selected preset

Internal runtime note:
- For internal GGUF scanning, `keep_model_loaded` is always ON internally during the scan run.
- Both internal scanner nodes always run in subprocess mode.
- Worker exit after each scan releases VRAM reliably.

### `GPM VLM Internal Diagnostics`
Purpose:
- quickly report local Python/platform + `llama_cpp` import/version status
- inspect `llama_cpp.llama_chat_format` for multimodal handler attributes/classes
- infer internal family from selected `model_name`
- resolve selected `model_name`/`mmproj_name` to filesystem paths and report existence
- report whether inferred family appears supported by the installed build
- diagnostics only (does not load model)

### `GPM VLM Prompt Loader`
Purpose:
- load one preset by stable preset id (built-in or user)
- output one normalized preset bundle for Saver
- never save/mutate/delete presets

Outputs:
- `preset_bundle` (`GPM_VLM_PROMPT_PRESET`)
- `status_text`
- `preset_json`

### `GPM VLM Prompt Saver`
Purpose:
- apply explicit user actions for preset management
- save new user presets from a source preset template
- update existing user presets
- delete existing user presets

Actions:
- `LOAD FROM SOURCE`
- `Save As New User Preset`
- `Update Existing User Preset`
- `Delete User Preset`

Notes:
- built-ins can be loaded and used as source templates
- built-ins cannot be updated or deleted
- `source_preset` is a `GPM_VLM_PROMPT_PRESET` input (connect Loader `preset_bundle` with one noodle)
- `ban_list` input is one term per line; blank lines are ignored
- Saver updates preset storage only; scanner sidecar JSON fields are unchanged
- `LOAD FROM SOURCE` stores source metadata in node-local saver state for later actions
- `SAVE/UPDATE/DELETE` actions intentionally do not reload from connected Loader source to avoid overwriting edits
- `load_preset` plus **Load selected preset into fields** is a node-local proof-of-concept path: it fetches the selected preset without queueing the graph, records the same source state as `LOAD FROM SOURCE`, and fills the editable fields
- the button only reads presets and records the selected source; it never writes, updates, or deletes a preset

Internal model locations:
- `ComfyUI/models/llm/`
- `ComfyUI/models/llm/GGUF/`
- `ComfyUI/models/GGUF/`

Internal dependency note:
- Qwen-VL GGUF internal mode requires a vision-capable `llama-cpp-python` build that supports both Qwen VL chat handlers and the corresponding llama.cpp model backend.

Discovery behavior:
- main VLM models: `*.gguf` files excluding names containing `mmproj`
- mmproj files: `*.gguf` filenames containing `mmproj`

Backend preset-manager (active for internal scanner preset lookup):
- module: `gpm_vlm_prompt_presets.py`
- built-in presets remain read-only
- user-editable preset file path prefers `ComfyUI/user/default/GPM/vlm_prompt_presets.json` when ComfyUI `folder_paths.user_directory` is available
- fallback path outside ComfyUI runtime: `gpm_vlm_prompt_presets.user.json` beside the module
- scanner adapter: `gpm_vlm_prompt_preset_adapter.py`
  - list preset options for UI dropdowns
  - resolve preset id with family-default fallback
  - append ban-list guidance to system prompt only when `use_ban_list=true` and list is non-empty

Prompt preset workflow (Loader -> Saver):
1. Add `GPM VLM Prompt Loader`.
2. Select a preset id.
3. Connect Loader `preset_bundle` to Saver `source_preset`.
4. In Saver set action to `LOAD FROM SOURCE`, then run once.
5. Edit Saver fields (`preset_name`, `system_prompt`, `use_ban_list`, `ban_list`).
6. Switch Saver action to `SAVE AS NEW USER PRESET`, `UPDATE EXISTING USER PRESET`, or `DELETE USER PRESET`, then run.
7. During save/update/delete, Saver uses edited fields + loaded source state and does not reload from Loader input.
8. Refresh/restart ComfyUI if preset dropdowns do not immediately show new user presets.

Node-local load-button proof of concept:
1. Add `GPM VLM Prompt Saver` (a Loader connection is not needed for this check).
2. Choose a preset in `load_preset`.
3. Press **Load selected preset into fields**.
4. Confirm `preset_name`, `system_prompt`, `use_ban_list`, and `ban_list` fill immediately.
5. Edit the fields, choose a save action, and run the Saver node. The button has already recorded the same server-side source state that the legacy `LOAD FROM SOURCE` action records.

Manual Saver UX check:
1. Add Loader and Saver.
2. Connect Loader `preset_bundle` to Saver `source_preset`.
3. Select `builtin-sdxl` in Loader.
4. Set Saver action `LOAD FROM SOURCE` and run.
5. Confirm Saver fields visibly fill (`preset_name`, `system_prompt`, `use_ban_list`, `ban_list`).
6. Edit `preset_name` and `system_prompt`.
7. Set Saver action `SAVE AS NEW USER PRESET` and run.
8. Confirm saved preset contains edited values.
9. Change Loader selected preset and run Saver save again without running `LOAD FROM SOURCE`.
10. Confirm Saver save does not overwrite edits from the changed Loader input.

Scanner usage note:
- scanner nodes continue consuming saved presets by preset id dropdown through the existing preset adapter/backend flow.

## Prototype operation model (v1)
`GPM Gallery Browser` uses node controls for navigation:
- `root_folder`: start/root folder
- `current_subfolder`: active folder relative to root
- `action`: `refresh`, `enter_folder`, `back`, `select_image`
- `entry_name`: folder or image name in the current folder

UI feedback is returned in node UI fields:
- status
- current subfolder
- folder/image listing text (`[DIR]` then `[IMG]`)

Browser outputs:
- `image`
- `person_prompt`
- `scene_prompt`
- `selected_image_path`

## Project structure
- `src/` -> ComfyUI node package code
- `tests/` -> core-logic tests
- `docs/` -> durable project documentation
- `scripts/` -> helper scripts and verification helpers
- `tools/` -> external/maintenance tooling

## Getting started

Supported install path (recommended):
1. Install **Gallery Prompt Manager** from ComfyUI Manager (GitHub/listing flow).
2. Restart ComfyUI if Manager or your launcher requests it.
3. Use `GPM VLM Internal Diagnostics` if you want a quick `llama_cpp` and internal readiness check.

Advanced/manual fallback:
1. Copy/clone this repo into `ComfyUI/custom_nodes/`.
2. Install normal requirements in the same Python environment ComfyUI uses:
   - `python -m pip install -r .\requirements.txt`
3. Run `python .\install.py` (special `llama-cpp-python` wheel handling path).
4. Restart ComfyUI.

Then:
1. Add `GPM Gallery Browser`, `GPM Prompt Combiner`, and one scanner node (`GPM VLM Scanner`, `GPM VLM Scanner (Internal)`, or `GPM VLM Scanner (Internal Advanced)`) from category `GPM`.
2. Set browser `root_folder`, then use `action` + `entry_name` to navigate/select.
3. Connect browser `person_prompt` + `scene_prompt` into combiner inputs; optionally set `lora_tags`.

Browser UI includes:
- prompt profile selector: `SDXL`, `Pony`, `Natural Language` (SDXL implemented end-to-end now)
- selection selector: `Manual`, `Sequential`, `Random`
- Sequential advances through the visible images in folder order on each queued execution. Random uses each visible image once before beginning a new shuffled cycle.
- `Save to JSON` button for writing active profile prompt edits to the selected image sibling JSON

One-time JSON migration helper:
```powershell
python .\scripts\migrate_prompt_keys.py <your_image_root>
```

## Dependency policy
- `requirements.txt` intentionally contains only normal, low-risk dependencies.
- `llama-cpp-python` is intentionally not listed in `requirements.txt` and is handled as a special-case install in `install.py`.
- `install.py` checks `import llama_cpp` first:
  - if already installed, it does not reinstall
  - if missing, it can try CUDA/cuBLAS index path when CUDA is applicable
  - in `auto`/`cuda`, unsupported CUDA wheel families do not trigger local source build; installer falls back to CPU wheel install
  - local CUDA source build is advanced opt-in only via `GPM_LLAMA_INSTALL_MODE=cuda-build`
  - if CUDA wheel or source build path is unavailable/fails, it falls back to plain `pip install --upgrade llama-cpp-python`
- optional install mode override is available with `GPM_LLAMA_INSTALL_MODE` (`auto`, `cpu`, `cuda`, `cuda-build`).
- on GPM module import, startup diagnostics print dependency status (`Pillow`, `llama_cpp`, internal support import, readiness) without running pip.
- Manager installs should normally not require manual commands; `install.py` remains available for special `llama-cpp-python` wheel handling when needed.
- Internal scanner readiness depends on both:
  - successful `llama_cpp` import, and
  - successful import of GPM internal support modules.
- Scanner prompt tuning and system-prompt/model-family refinement are separate future work and are not changed by install flow.

## Documentation
- `docs/setup.md`
- `docs/architecture.md`
- `docs/troubleshooting.md`
- `docs/decisions.md`
- `ROADMAP.md`
- `TASKS.md`


