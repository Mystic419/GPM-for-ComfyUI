from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

RUNTIME_MODE_API = "api"
RUNTIME_MODE_INTERNAL = "internal"
SUPPORTED_RUNTIME_MODES = {RUNTIME_MODE_API, RUNTIME_MODE_INTERNAL}

# Presets are currently validated primarily against this GGUF family.
# Alternate GGUF models are still allowed and not blocked.
RECOMMENDED_GGUF_MODEL_REPO = "mradermacher/Gliese-Qwen3.5-9B-Abliterated-Caption-GGUF"


class GPMVLMRuntime(ABC):
    runtime_mode: str = RUNTIME_MODE_API

    def start(self) -> tuple[bool, str]:
        return True, ""

    def stop(self) -> None:
        return None

    @abstractmethod
    def generate(self, image_path: Path, preset: dict[str, Any]) -> tuple[str, str, str]:
        raise NotImplementedError

