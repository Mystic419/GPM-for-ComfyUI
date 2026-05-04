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
  - in `auto`/`cuda`, unsupported CUDA wheel families do not trigger local source builds; installer falls back to CPU wheel install
  - local CUDA source build is available only in `cuda-build` mode and uses a 90-minute timeout before CPU fallback
  - if CUDA wheel/source build install fails (or CUDA is not used), installer falls back to plain `pip install --upgrade llama-cpp-python`
- `requirements.txt` is intentionally minimal and does not include `llama-cpp-python`.
- llama install mode can be overridden with:
  - `GPM_LLAMA_INSTALL_MODE=auto` (default)
  - `GPM_LLAMA_INSTALL_MODE=cpu`
  - `GPM_LLAMA_INSTALL_MODE=cuda`
  - `GPM_LLAMA_INSTALL_MODE=cuda-build` (advanced opt-in local CUDA source build)

### Maintainer wheel build helper (Windows CUDA)
- `scripts/build_llama_cpp_wheel.ps1` is maintainer/developer-only tooling for building a local `llama-cpp-python` wheel.
- Normal users should install GPM through ComfyUI Manager and use the regular `install.py` flow.
- This script runs `pip wheel` only; it does not install the wheel and does not uninstall/modify existing `llama_cpp`.
- Source builds can take a long time depending on machine and toolchain state.
- Phase 1 target is Windows CUDA wheel building.
- Default `CudaArchitectures` is `86` (RTX 30-series Ampere).
- Other architectures can be passed later, but should be validated before release use.
- After `scripts/build_llama_cpp_wheel.ps1` runs, use the archived architecture-specific wheel artifact it prints (not the raw pip wheel filename).
- Archive naming pattern:
  - `llama_cpp_python-<version>-<python_tag>-win_amd64-cu130-sm<arch>.whl`
- Installer local wheel behavior:
  - archived architecture-specific filenames are preferred for storage/distribution and local matching
  - when an archived wheel is selected, `install.py` copies it to a temporary pip-valid filename before install:
    - `llama_cpp_python-<version>-py3-none-win_amd64.whl`
  - raw pip wheel filenames are fallback/local build leftovers only
- Local maintainer wheel manifest (documentation-only for now):
  - `dist/gpm_wheels/wheels_manifest.local.json` records locally built maintainer CUDA wheels.
  - Because `dist/` is gitignored, `docs/wheels_manifest.example.json` is the checked-in template.
  - Current status in the manifest:
    - `sm86` is validated (validated on RTX 3060 / Ampere).
    - `sm89` and `sm120` are built but require community validation.
  - This manifest does not change installer behavior yet; hosted-download installer support comes later.
- `install.py` now checks local wheel directories before CPU fallback in CUDA paths:
  - `.\wheels\`
  - `.\dist\gpm_wheels\`
- Phase 1 local wheel matching is intentionally narrow: Windows + Python 3.12 + `win_amd64` + `cu130` + detected GPU SM arch.
- Supported local archived wheel arches (when matching wheel files are present):
  - `sm86` (RTX 30-series / Ampere): validated
  - `sm89` (RTX 40-series / Ada): built, maintainer-unvalidated (community validation needed)
  - `sm120` (RTX 50-series / Blackwell): built, maintainer-unvalidated (community validation needed)
- If detected SM arch is unsupported (or matching wheel is missing), installer skips local wheel and falls back to CPU unless user opts into `cuda-build`.
- Wheel files are local/distribution artifacts and are not committed to git; distribution workflow will be added later.

PowerShell examples:
```powershell
# Build default tested wheel (0.3.22, sm86) using active python
.\scripts\build_llama_cpp_wheel.ps1

# Build with explicit ComfyUI/testing venv python
.\scripts\build_llama_cpp_wheel.ps1 -PythonExe "F:\My_AI\Data\Packages\testing\venv\Scripts\python.exe"

# Build alternate architecture/version
.\scripts\build_llama_cpp_wheel.ps1 -CudaArchitectures "89" -Version "0.3.22"

# Test install from archived architecture-specific wheel (no source build flags)
.\scripts\test_llama_cpp_wheel_install.ps1 -PythonExe "F:\My_AI\Data\Packages\testing\venv\Scripts\python.exe" -WheelPath "D:\antigravity\GPM\dist\gpm_wheels\llama_cpp_python-0.3.22-cp312-win_amd64-cu130-sm86.whl"
```

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

# explicit cuda mode (prebuilt CUDA wheel attempt; no source build)
$env:GPM_LLAMA_INSTALL_MODE='cuda'; python .\install.py

# advanced local CUDA source-build mode
$env:GPM_LLAMA_INSTALL_MODE='cuda-build'; python .\install.py
```

### Final dependency report fields
`install.py` prints at minimum:
- Python executable
- platform
- selected llama install mode (`auto` / `cpu` / `cuda` / `cuda-build`)
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


