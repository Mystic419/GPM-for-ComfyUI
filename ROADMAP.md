# ROADMAP

## Current phase
Public-release preparation and post-release hardening

## Near-term priorities
- Finalize the first public node set, workflows, and installation guidance
- Validate a clean install against the current ComfyUI release
- Add a repository license and publish the first tagged release
- Continue Qwen2.5-VL scan-quality regression coverage

## Mid-term goals
- Improve Gallery Browser usability for larger folders
- Expand prompt-combiner controls where they add clear workflow value
- Validate additional internal vision-model families before enabling them
- Add thumbnail caching and folder refresh support
- Add tests for JSON parsing, prompt sanitization, prompt combination, and save-back behavior

## Later ideas
- Richer gallery UX for large folders
- Search, filter, and sort inside the browser node
- Batch rescan by stale metadata or scanner version
- Optional drag/drop asset board or favorites system
- Optional workflow examples for SDXL, Flux, and natural-language prompt chains
- Optional support for additional local inference backends beyond LM Studio
- Optional prompt-style presets that ship with the package

## Risks / constraints
- ComfyUI gallery UI work may be more difficult than the scanner/prompt logic
- Large image folders may require pagination or lazy loading
- Local vision model behavior may vary across endpoints
- Prompt leakage between scene and person descriptions may require iterative cleanup rules
- Prompt-style expectations vary widely across model families
- ComfyUI frontend extension behavior may differ across versions
