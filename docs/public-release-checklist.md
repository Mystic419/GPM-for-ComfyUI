# Public release checklist

Use this checklist before publishing a tagged GPM release.

## Repository

- [x] License selected: Apache License 2.0. Confirm `LICENSE` and `NOTICE` remain included in the release.
- [ ] Review the README, included workflows, and screenshots against the current ComfyUI release.
- [ ] Confirm no local preset data, model files, sidecars, logs, or private paths are tracked.
- [ ] Verify `git status` contains only intentional changes.
- [ ] Review `CHANGELOG.md` and add a version/date heading for the release.

## Verification

- [ ] Run `scripts/verify.ps1`.
- [ ] Run `pytest` in the same Python environment used by ComfyUI.
- [ ] Start ComfyUI with a clean browser refresh and confirm GPM nodes load.
- [ ] Import every workflow in `workflows/` and confirm missing third-party nodes/models are clearly identified.
- [ ] Run a small Qwen2.5-VL scan using `SKIP_EXISTING` and confirm sidecars are written.
- [ ] Confirm the Gallery Browser loads those sidecars and the Prompt Combiner emits the expected prompt.

## Release notes

- [ ] State the supported ComfyUI/Python environment used for verification.
- [ ] State that Qwen2.5-VL is the validated internal scanner family.
- [ ] State that Qwen3.x/Qwen3.5 models are intentionally blocked pending multimodal validation.
- [ ] Link the workflow examples and setup/troubleshooting guides.
- [ ] Create a version tag and GitHub release only after the checks above pass.
