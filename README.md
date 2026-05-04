# Gallery Prompt Manager

Gallery Prompt Manager is a ComfyUI custom node pack for browsing image folders as prompt assets, scanning images into sibling JSON metadata, and combining prompt halves for reuse.

## Status
Private prototype in progress.

Implemented now:
- `GPM Gallery Browser` v1 backend prototype (folder navigation + image selection + sibling JSON load)
- `GPM Prompt Combiner` v1 (person + scene + optional LoRA tags -> one clean prompt string)
- `GPM VLM Scanner` v1 (recursive scan + fixed-family sidecar writes via preset-selected family)
- `GPM VLM Scanner (Internal)` v2 (in-process `llama-cpp-python` runtime with ComfyUI model-folder dropdown UX)
- `GPM VLM Scanner (Internal Advanced)` v1 (same in-process runtime with manual tuning controls)
- `GPM VLM Internal Diagnostics` v1 (environment/status helper for internal GGUF multimodal support)
- Global VLM preset storage with built-in read-only defaults (`SDXL`, `Pony`, `Natural Language`)

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
- each `GPM Gallery Browser` node instance persists its own `root_folder`, `current_subfolder`, `selected_image_rel`, and `visible_rows` using the ComfyUI node id

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
- load GGUF VLM + mmproj directly in-process via `llama-cpp-python` (no subprocess server)
- discover model files from ComfyUI model folders and expose dropdowns (`model_name`, `mmproj_name`)
- support `mmproj_name=(auto)` matching when one clear candidate exists
- use an explicit internal family support gate before scan execution:
  - internal scan correctness is currently verified for `Qwen2.5-VL` only
  - unverified families (including Gliese/Qwen3.x and other unvalidated multimodal families) are blocked with a clear startup error instead of scanning
- use a dedicated Qwen-VL runtime path (Qwen handler + filtered llama constructor kwargs)
  - multimodal request image payload remains family-aware (`qwen_vl` uses object-style `image_url`)
- optional `debug_mode=ON` emits concise startup compatibility diagnostics in `summary_json` when internal startup fails

### `GPM VLM Scanner (Internal Advanced)`
Purpose:
- same internal in-process scanner flow as the basic internal node
- exposes advanced runtime controls (`n_ctx`, `n_gpu_layers`, `temperature`, `top_p`, `max_tokens`, `threads`, `batch_size`, `keep_model_loaded`)
- keeps the same no-executable-path UX as the basic internal node
- includes optional `debug_mode` toggle with the same startup diagnostics behavior as the basic internal node

### `GPM VLM Internal Diagnostics`
Purpose:
- quickly report local Python/platform + `llama_cpp` import/version status
- inspect `llama_cpp.llama_chat_format` for multimodal handler attributes/classes
- infer internal family from selected `model_name`
- resolve selected `model_name`/`mmproj_name` to filesystem paths and report existence
- report whether inferred family appears supported by the installed build
- diagnostics only (does not load model)

Internal model locations:
- `ComfyUI/models/llm/`
- `ComfyUI/models/llm/GGUF/`
- `ComfyUI/models/GGUF/`

Internal dependency note:
- Qwen-VL GGUF internal mode requires a vision-capable `llama-cpp-python` build that supports both Qwen VL chat handlers and the corresponding llama.cpp model backend.

Discovery behavior:
- main VLM models: `*.gguf` files excluding names containing `mmproj`
- mmproj files: `*.gguf` filenames containing `mmproj`

Global preset storage:
- file: `gpm_vlm_presets.json` in the node package directory
- auto-created if missing
- built-ins are read-only defaults
- user presets are supported as global recipes and are not duplicated into image sidecars
- built-in ids: `builtin-sdxl`, `builtin-pony`, `builtin-natural-language`
- managed user preset ids: `sdxl_user`, `pony_user`, `natural_user` (auto-created from matching built-ins if missing)
- preset schema includes `system_prompt`, `ban_list`, `temperature`, `top_p`, `max_tokens`

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
- randomize selector: `OFF`, `ON`
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


