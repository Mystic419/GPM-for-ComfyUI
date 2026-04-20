# ROADMAP

## Current phase
Private prototype hardening and node-set expansion

## Near-term priorities
- Finalize the v0 node set and their responsibilities
- Refactor the existing image scanner to write sibling JSON only
- Define the gallery node UX for image selection + prompt editing
- Lock how prompt style, ban list, and save-to-JSON behavior work
- Create the initial ComfyUI custom node package structure
- Build the first verification path for local development

## Mid-term goals
- Ship a working clickable gallery browser node
- Ship a folder scanner node with OpenAI-compatible endpoint settings
- Expand prompt combiner with optional prompt-style controls and ban-list handling
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
