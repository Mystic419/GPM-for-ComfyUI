from __future__ import annotations

import gc
import json
from typing import Any

from .gpm_vlm_runtime_internal import GPMGGUFInternalRuntime


def _best_effort_torch_cuda_empty_cache() -> tuple[bool, str]:
    try:
        import torch  # type: ignore
    except Exception:
        return False, "torch unavailable"
    try:
        if bool(getattr(torch, "cuda", None)) and torch.cuda.is_available():
            torch.cuda.empty_cache()
            return True, ""
        return False, "cuda unavailable"
    except Exception as exc:
        return False, str(exc)


class GPMVLMFreeVRAM:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["FREE_VRAM", "NO_OP"], {"default": "FREE_VRAM"}),
                "aggressive_comfy_cleanup": (["ON", "OFF"], {"default": "ON"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("summary_json", "status_text")
    FUNCTION = "free_vram"
    CATEGORY = "GPM"

    def free_vram(self, action: str, aggressive_comfy_cleanup: str):
        summary: dict[str, Any] = {
            "action": str(action),
            "aggressive_comfy_cleanup": str(aggressive_comfy_cleanup),
            "gpm_internal_cache_cleared": False,
            "gpm_internal_cache_had_runtime": False,
            "comfy_unload_all_models_called": False,
            "comfy_soft_empty_cache_called": False,
            "torch_cuda_empty_cache_called": False,
            "gc_collect_called": False,
            "warnings": [],
        }

        if str(action) == "NO_OP":
            status_text = "No-op: cleanup skipped by action=NO_OP"
            return json.dumps(summary, indent=2, ensure_ascii=False), status_text

        runtime_info = GPMGGUFInternalRuntime.clear_cached_runtime(reason="manual free vram node")
        summary["gpm_internal_cache_cleared"] = bool(runtime_info.get("cleared_cached_runtime", False))
        summary["gpm_internal_cache_had_runtime"] = bool(runtime_info.get("had_cached_runtime", False))

        try:
            gc.collect()
            summary["gc_collect_called"] = True
        except Exception as exc:
            summary["warnings"].append(f"gc.collect failed: {exc}")

        torch_ok, torch_error = _best_effort_torch_cuda_empty_cache()
        summary["torch_cuda_empty_cache_called"] = bool(torch_ok)
        if torch_error:
            summary["warnings"].append(f"torch cache clear skipped/failed: {torch_error}")

        if str(aggressive_comfy_cleanup) == "ON":
            try:
                import comfy.model_management as model_management  # type: ignore

                unload_fn = getattr(model_management, "unload_all_models", None)
                if callable(unload_fn):
                    try:
                        unload_fn()
                        summary["comfy_unload_all_models_called"] = True
                    except Exception as exc:
                        summary["warnings"].append(f"comfy unload_all_models failed: {exc}")
                else:
                    summary["warnings"].append("comfy unload_all_models unavailable")

                soft_empty_fn = getattr(model_management, "soft_empty_cache", None)
                if callable(soft_empty_fn):
                    try:
                        soft_empty_fn()
                        summary["comfy_soft_empty_cache_called"] = True
                    except Exception as exc:
                        summary["warnings"].append(f"comfy soft_empty_cache failed: {exc}")
                else:
                    summary["warnings"].append("comfy soft_empty_cache unavailable")
            except Exception as exc:
                summary["warnings"].append(f"comfy.model_management unavailable: {exc}")
        else:
            summary["warnings"].append("aggressive comfy cleanup skipped (OFF)")

        status_parts = [
            "GPM internal VLM cache cleared",
            (
                "ComfyUI unload_all_models called"
                if summary["comfy_unload_all_models_called"]
                else "ComfyUI unload_all_models skipped"
            ),
            (
                "ComfyUI soft_empty_cache called"
                if summary["comfy_soft_empty_cache_called"]
                else "ComfyUI soft_empty_cache skipped"
            ),
            (
                "torch CUDA cache cleared"
                if summary["torch_cuda_empty_cache_called"]
                else "torch CUDA cache skipped"
            ),
        ]
        status_text = " | ".join(status_parts)
        return json.dumps(summary, indent=2, ensure_ascii=False), status_text
