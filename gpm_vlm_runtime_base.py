from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

RUNTIME_MODE_API = "api"
RUNTIME_MODE_INTERNAL = "internal"
SUPPORTED_RUNTIME_MODES = {RUNTIME_MODE_API, RUNTIME_MODE_INTERNAL}

# The internal scanner’s validated multimodal family. This is descriptive
# metadata for built-in presets, not a recommendation to download one repo.
VALIDATED_INTERNAL_MODEL_FAMILY = "Qwen2.5-VL"

class GPMVLMRuntime(ABC):
    runtime_mode: str = RUNTIME_MODE_API

    def start(self) -> tuple[bool, str]:
        return True, ""

    def stop(self) -> None:
        return None

    @abstractmethod
    def generate(self, image_path: Path, preset: dict[str, Any]) -> tuple[str, str, str]:
        raise NotImplementedError

