# CHANGELOG

All notable user-visible changes should be recorded here.

## Unreleased

### Added
- First private prototype implementation of `GPM Gallery Browser` node.
- Folder navigation helpers that keep traversal inside a chosen root folder.
- Image filtering for `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`.
- Sibling JSON prompt loading for `sdxl_person` and `sdxl_scene`.
- Graceful fallback behavior when sidecar JSON is missing or invalid.
- Initial tests for browser core listing, navigation, and JSON loading behavior.
- First private prototype implementation of `GPM Prompt Combiner` node.
- Prompt combiner core helper/tests for ordered merge of person + scene + optional LoRA tags with separator cleanup.
- Prompt-profile-ready gallery UI controls: `SDXL`, `Pony`, `Natural Language`.
- Gallery randomizer controls: `OFF`, `ON`.
- One-time migration script `scripts/migrate_prompt_keys.py` for JSON key rename.
- Manual `Save to JSON` action in gallery browser UI to persist active-profile prompt edits for the currently selected image.
- First implementation of `GPM VLM Scanner` node with recursive root-folder image discovery and GGUF-only backend mode.
- New preset manager helper module `gpm_vlm_presets.py` with global `gpm_vlm_presets.json` storage and built-in read-only presets (`SDXL`, `Pony`, `Natural Language`).
- New scanner backend helper module `gpm_vlm_backend.py` for family-slot mapping, skip/overwrite logic, robust scan summaries, and sidecar JSON updates that preserve unrelated fields.
- Runtime abstraction groundwork for scanner backends with shared orchestration:
  - `gpm_vlm_runtime_base.py`
  - `gpm_vlm_runtime_api.py` (current OpenAI-compatible endpoint behavior)
  - `gpm_vlm_runtime_internal.py` (in-process GGUF runtime path)
- Lightweight recommended-model metadata for built-in VLM presets using `mradermacher/Gliese-Qwen3.5-9B-Abliterated-Caption-GGUF`.
- New `GPM VLM Scanner (Internal Advanced)` node for in-process GGUF scanning with optional runtime tuning controls.
- New `gpm_vlm_model_discovery.py` helper for ComfyUI model-folder GGUF/mmproj discovery and mmproj auto-pairing.
- New `GPM VLM Internal Diagnostics` node for fast environment checks (Python/platform, `llama_cpp` import/version, multimodal handler visibility, family support detection, and model/mmproj path resolution without loading models).
- Optional `debug_mode` input on internal scanner nodes that adds concise startup compatibility payload fields on internal startup failure (family/handler selection path, constructor kwargs summary, resolved paths, file sizes, llama-cpp version).
- Internal diagnostics now expose backend probe details (`backend_probe`, probe method/error, import/version), runtime request parameters (`n_gpu_layers`, `n_ctx`, `n_batch`, `threads`), selected handler/family, resolved paths, and constructor kwarg filtering results in JSON.
- New `install.py` staged installer for dependency reliability:
  - normal requirements install from `requirements.txt`
  - explicit special-case `llama-cpp-python` handling when `llama_cpp` import is missing
  - post-install dependency/status report with readiness summary (`READY`/`PARTIAL`/`NOT READY`)
- Llama install policy modes in installer: `auto` (default), `cpu`, `cuda` (via `GPM_LLAMA_INSTALL_MODE`) with mode-aware verification/reporting fields.
- New import-time GPM startup dependency diagnostics block (visibility-only; no auto-pip) covering Pillow/llama_cpp/internal-support status and internal readiness.

### Changed
- Reduced temporary gallery restore debug logging; kept only concise warning/fallback logs for persistence issues.
- Project status moved from planning-only to active prototype implementation.
- README/setup/architecture/task docs updated for the implemented browser-node scope.
- Hard schema switch for scanner/browser metadata from `person_prompt`/`scene_prompt` to `sdxl_person`/`sdxl_scene` (no legacy fallback reads).
- Package node registration now includes `GPM VLM Scanner` while preserving existing browser and combiner node registration.
- Internal runtime path now runs in-process (`llama-cpp-python` vision flow) instead of launching a local HTTP server process.
- `GPM VLM Scanner (Internal)` node inputs now use model/mmproj dropdowns and remove executable/path/host/port arguments.
- Shared scan backend now records internal runtime metadata (`runtime_mode`, `model_name`, `mmproj_name`, resolved model paths, keep-loaded setting) for internal scans.
- Internal GGUF runtime now uses a dedicated Qwen-VL loading path modeled after maintained ComfyUI Qwen-VL nodes:
  - Qwen-specific chat handler kwargs (`clip_model_path` + optional image-token controls) with constructor-signature filtering
  - Qwen-specific Llama constructor profile (`swa_full`, `top_k`, `pool_size`, image token bounds) with constructor-signature filtering
  - LLava-family models keep the existing generic internal path
- Internal startup diagnostics now record requested vs filtered Llama constructor kwargs and dropped kwargs, making environment/build incompatibilities easier to diagnose.
- Internal runtime startup metadata now includes best-effort backend capability detection (CUDA/Vulkan/Metal/HIP/SYCL/OpenCL when detectable), plain-English backend status, GPU offload request flags, and debug-mode hinting when GPU offload is requested but probes look CPU-only.
- Dependency policy clarified:
  - `requirements.txt` remains minimal and excludes `llama-cpp-python`
  - special `llama-cpp-python` installation is centralized in `install.py`
  - install reporting now explicitly surfaces mode, special-install attempt/result, llama-cpp version, and mode-aware readiness (`READY`/`PARTIAL`/`NOT READY`)
- Installer flow now matches standard ComfyUI llama/VLM custom-node expectations:
  - no in-UI installer node
  - no startup auto-install behavior
  - `install.py` handles wheel-first `llama-cpp-python` installation only when `llama_cpp` is missing
  - CUDA path uses cuBLAS wheel index strategy; CPU path uses release wheel URL strategy
- Startup diagnostics remain lightweight and non-invasive, but no longer depend on install-state markers.
- Install/docs positioning is now explicit:
  - ComfyUI Manager is the supported install path
  - manual clone/copy is an advanced fallback
  - `install.py` is retained for special `llama-cpp-python` wheel handling only
- Per-image scan sidecar metadata is now intentionally minimal and stable by default:
  - `gpm_meta.vlm_scan` now stores only `family`, `preset_id`, `backend`, `runtime`, `model`, `status`, and `scanned_at`.
  - verbose internal runtime/debug details were moved out of normal `vlm_scan` and now write to `gpm_meta.vlm_scan_debug` only when internal scanner `debug_mode=ON`.
- Internal runtime multimodal request construction is now family-aware:
  - `qwen_vl` sends object-style `image_url` payloads
  - `llava` sends string-style `image_url` payloads
- Internal runtime now explicitly resets per-image request state each scan call and exposes consumable per-image debug traces for scan orchestration.
- Internal runtime now separates "model may load" from "family approved for reliable scan correctness" with a centralized family support gate.
- Internal scanner startup debug metadata now includes support-gating fields: `detected_model_family`, `selected_chat_handler`, `family_support_status`, `support_reason`.

### Fixed
- Randomizer `ON` now reselects one visible image per execution cycle in queued runs (backend-driven); `OFF` preserves manual selection and prompt edits during runs.
- Added `IS_CHANGED` handling for `GPM Gallery Browser` so `ON` forces rerun each queued cycle (prevents ComfyUI cache reuse), while `OFF` remains cache-friendly.
- Save endpoint now preserves unrelated keys in existing sibling JSON files and only updates active-profile prompt keys.
- Scoped `Save to JSON` to the clicked browser-node instance by sending/validating `node_id`; save requests now include per-node debug context and no-op safely when no image is selected.
- Browser prompt text areas are now editable, and execution uses the current editor text instead of reloading sibling JSON on each run.
- Persisted `selected_image_rel` alongside root/current browser state and restore it safely only when the image still exists under the restored folder context.
- Strengthened selected tile styling with a bold red border and subtle glow/background change for clearer visual selection state.
- Removed the leftover cut-off line above top controls by fully hiding internal persistence widgets.
- Fixed gallery vertical resize behavior so extra node height expands the middle grid area instead of leaving dead space below prompt panels.
- Fixed repeated-click shift bug by skipping selection rerender/prompt reload when clicking the already-selected image.
- Fixed gallery persistence collisions by storing browser state per ComfyUI node id instead of one shared global state.
- Fixed gallery restore on reload by waiting for a valid ComfyUI node id before requesting persisted state and by enforcing strict node-id-keyed backend state loading.
- Fixed restore order so reload uses existing hidden widget state first and only falls back to per-node JSON state when widget restore state is empty.
- Fixed a high-severity internal scan correctness risk where mismatched multimodal image payload format could lead to generic/off-image outputs.
- Added debug-only per-image trace writes to sidecars under `gpm_meta.vlm_scan_debug_trace` (image path/hash, output path, prompt sent, raw response, parsed prompts).
- Added lightweight debug guard: when debug mode is ON and a response looks like generic family/living-room text for a likely different image type, scanner warns and skips overwriting existing sidecar JSON for that image.
- Internal scanner now blocks unvalidated internal multimodal families (including Gliese/Qwen3.x variants) from running scan writes; only Qwen2.5-VL is currently approved for internal scan correctness.

### Removed
- `GPM Install Dependencies` node and its registration from node mappings.
- Install-state marker flow (`.gpm_install_state.json`) and marker-specific startup messaging.










