import json
import os
import sys
import types
import importlib.util
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
PACKAGE_NAME = "gpm_testpkg"
if PACKAGE_NAME not in sys.modules:
    pkg = types.ModuleType(PACKAGE_NAME)
    pkg.__path__ = [ROOT_DIR]
    sys.modules[PACKAGE_NAME] = pkg


def _load_module(module_basename: str):
    full_name = f"{PACKAGE_NAME}.{module_basename}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    file_path = os.path.join(ROOT_DIR, f"{module_basename}.py")
    spec = importlib.util.spec_from_file_location(full_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {module_basename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


gpm_vlm_presets = _load_module("gpm_vlm_presets")
GPMVLMPresetStore = gpm_vlm_presets.GPMVLMPresetStore
get_preset_generation_settings = gpm_vlm_presets.get_preset_generation_settings


def test_store_initializes_builtin_and_user_family_presets(tmp_path: Path):
    store = GPMVLMPresetStore(tmp_path / "gpm_vlm_presets.json")
    presets = store.list_presets()
    ids = {str(item.get("id", "")) for item in presets}

    assert "builtin-sdxl" in ids
    assert "builtin-pony" in ids
    assert "builtin-natural-language" in ids
    assert "sdxl_user" in ids
    assert "pony_user" in ids
    assert "natural_user" in ids


def test_preset_generation_settings_are_clamped():
    temperature, top_p, max_tokens = get_preset_generation_settings(
        {
            "temperature": 9.0,
            "top_p": -2.0,
            "max_tokens": 999999,
        }
    )
    assert temperature == 2.0
    assert top_p == 0.0
    assert max_tokens == 2048


def test_preset_normalization_preserves_unknown_fields(tmp_path: Path):
    store = GPMVLMPresetStore(tmp_path / "gpm_vlm_presets.json")
    presets = store.list_presets()
    payload_path = tmp_path / "gpm_vlm_presets.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    for item in payload["presets"]:
        if item.get("id") == "sdxl_user":
            item["custom_marker"] = "keep-me"
            break
    payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    updated = store.get_preset("sdxl_user")
    assert updated is not None
    assert updated.get("custom_marker") == "keep-me"

    # Trigger write/repair path and ensure custom field remains.
    store.list_presets()
    payload_path = tmp_path / "gpm_vlm_presets.json"
    text = payload_path.read_text(encoding="utf-8")
    assert "keep-me" in text
