# Setup

## Requirements
- Python 3.12.x (tested target: 3.12.12)
- Local ComfyUI install
- Pillow, NumPy, Torch available in the ComfyUI Python environment

## Install

### Supported path (ComfyUI Manager)
1. Install **Gallery Prompt Manager** from ComfyUI Manager (GitHub/listing flow).
2. Restart ComfyUI if Manager or your launcher asks for restart.
3. Confirm `GPM Gallery Browser`, `GPM Prompt Combiner`, `GPM VLM Scanner`, `GPM VLM Scanner (Internal)`, and `GPM VLM Scanner (Internal Advanced)` appear under category `GPM`.

### Advanced/manual fallback
1. Clone this repo.
2. Place/symlink it under `ComfyUI/custom_nodes/`.
3. Install normal requirements from the same Python executable/environment ComfyUI uses:
   - `python -m pip install -r .\requirements.txt`
4. Run `python .\install.py` (special `llama-cpp-python` wheel handling path).
5. Restart ComfyUI.

### Installer behavior
- `install.py` checks `import llama_cpp` first.
  - if already importable, installer prints status and exits without reinstalling
  - if missing, installer can try a CUDA/cuBLAS wheel index path when CUDA is applicable (`jllllll` cuBLAS wheel index family)
  - if CUDA wheel install fails (or CUDA is not used), installer falls back to plain `pip install --upgrade llama-cpp-python`
- `requirements.txt` is intentionally minimal and does not include `llama-cpp-python`.
- llama install mode can be overridden with:
  - `GPM_LLAMA_INSTALL_MODE=auto` (default)
  - `GPM_LLAMA_INSTALL_MODE=cpu`
  - `GPM_LLAMA_INSTALL_MODE=cuda`

### Startup diagnostics behavior
- On normal GPM import at ComfyUI startup, a concise dependency status block is printed:
  - Pillow import status
  - `llama_cpp` import status
  - `llama-cpp-python` version (if available)
  - internal support module import status
  - internal scanner readiness (`READY` / `PARTIAL` / `NOT READY`)
- Import-time diagnostics are visibility-only:
  - no pip install commands are run
  - no automatic repair is attempted

PowerShell examples:
```powershell
# normal requirements
python -m pip install -r .\requirements.txt

# llama-cpp special install path (default auto mode)
python .\install.py

# explicit cpu mode
$env:GPM_LLAMA_INSTALL_MODE='cpu'; python .\install.py

# explicit cuda mode
$env:GPM_LLAMA_INSTALL_MODE='cuda'; python .\install.py
```

### Final dependency report fields
`install.py` prints at minimum:
- Python executable
- platform
- selected llama install mode (`auto` / `cpu` / `cuda`)
- CUDA detection status
- `llama_cpp` import status
- `llama-cpp-python` version (if importable)
- internal support import status (`gpm_vlm_internal_multimodal`)

## Prototype use
1. Add `GPM VLM Scanner` for external OpenAI-compatible endpoints, or add `GPM VLM Scanner (Internal)` / `GPM VLM Scanner (Internal Advanced)` for in-process local GGUF scanning.
2. For internal nodes, place model files in ComfyUI model folders:
   - `ComfyUI/models/llm/`
   - `ComfyUI/models/llm/GGUF/`
   - `ComfyUI/models/GGUF/`
3. In internal nodes, select `model_name` and `mmproj_name` from dropdowns (use `mmproj_name=(auto)` when pairing is clear).
4. Set scanner `root_folder`, `preset_id`, and overwrite/limit controls.
5. Use `overwrite_mode=SKIP_EXISTING` for default non-destructive scans, or `OVERWRITE_FAMILY` to replace only the selected family slot.
6. Add `GPM Gallery Browser` node.
7. Set `root_folder` to your top-level image folder.
8. Keep `current_subfolder` blank for root, or set a relative subfolder.
9. Use:
   - `action=enter_folder` + `entry_name=<folder name>` to go deeper
   - `action=back` to go to parent (stays within root)
   - `action=select_image` + `entry_name=<image file name>` to select image
10. Read node UI feedback (`status`, `current_subfolder`, `entries`) and outputs.
11. Add `GPM Prompt Combiner` and connect browser `person_prompt` and `scene_prompt` outputs.
12. Optionally set `lora_tags`; use `combined_prompt` as your final positive prompt string.
13. In browser UI, choose prompt profile (`SDXL`, `Pony`, `Natural Language`) and optional randomize mode (`OFF`, `ON`).

## Node outputs
`GPM Gallery Browser`:
- `image`
- `person_prompt`
- `scene_prompt`
- `selected_image_path`

If sibling JSON is missing/invalid, prompt outputs are empty strings.

Current JSON keys used by browser/scanner:
- `sdxl_person`
- `sdxl_scene`
- `pony_person`
- `pony_scene`
- `natural_person`
- `natural_scene`

`GPM Prompt Combiner`:
- `combined_prompt`

## Verification
PowerShell:
```powershell
.\scripts\verify.ps1
pytest
```

## Notes
- VLM presets are stored globally in `gpm_vlm_presets.json` beside the node files and auto-created with read-only built-ins (`SDXL`, `Pony`, `Natural Language`) if missing.
- Internal scanner correctness is currently validated for `Qwen2.5-VL` only (with a vision-capable `llama-cpp-python` build).
- Other internal multimodal families may still load, but are blocked from scan execution until validated for correctness in GPM.
- Scanner prompt tuning/model-family prompt refinement is separate future work; install flow only manages dependency readiness.
- Browser prompt text areas are editable; edits affect node output for the current workflow run.
- Use `Save to JSON` to persist current active-profile prompt text to the selected image sibling JSON (`*.json`).
- In `ON`, the browser re-randomizes on each execution cycle (including queued runs) using visible images in the current folder only.
- Clickable gallery tiles are implemented for selection and JSON prompt loading on image click.
- Combiner v1 intentionally keeps formatting simple: person -> scene -> lora with blank filtering and comma/whitespace cleanup.


