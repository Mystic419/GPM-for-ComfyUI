import importlib.util
import os
import sys
import types
from pathlib import Path


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
PACKAGE_NAME = "gpm_testpkg_prompt_adapter"
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


def test_default_preset_ids_by_family():
    mod = _load_module("gpm_vlm_prompt_preset_adapter")
    assert mod.get_default_preset_id_for_family("sdxl") == "builtin-sdxl"
    assert mod.get_default_preset_id_for_family("pony") == "builtin-pony"
    assert mod.get_default_preset_id_for_family("natural") == "builtin-natural-language"


def test_invalid_preset_id_falls_back_to_builtin():
    mod = _load_module("gpm_vlm_prompt_preset_adapter")
    preset, status = mod.get_vlm_prompt_preset("does-not-exist", fallback_family="pony")
    assert preset["id"] == "builtin-pony"
    assert preset["family"] == "Pony"
    assert "warning:" in status.lower()


def test_user_preset_prompt_retrieval(tmp_path: Path, monkeypatch):
    backend = _load_module("gpm_vlm_prompt_presets")
    mod = _load_module("gpm_vlm_prompt_preset_adapter")
    monkeypatch.setenv("GPM_USER_DATA_DIR", str(tmp_path))
    backend.save_user_preset(
        {
            "id": "user-1",
            "name": "User One",
            "family": "natural",
            "editable": True,
            "system_prompt": "custom prompt",
            "use_ban_list": False,
            "ban_list": [],
            "version": 1,
        }
    )
    preset, status = mod.get_vlm_prompt_preset("user-1", fallback_family="natural")
    assert status == ""
    assert preset["id"] == "user-1"
    assert preset["family"] == "Natural Language"
    assert preset["system_prompt"] == "custom prompt"


def test_ban_list_appends_instruction(tmp_path: Path, monkeypatch):
    backend = _load_module("gpm_vlm_prompt_presets")
    mod = _load_module("gpm_vlm_prompt_preset_adapter")
    monkeypatch.setenv("GPM_USER_DATA_DIR", str(tmp_path))
    backend.save_user_preset(
        {
            "id": "user-ban",
            "name": "User Ban",
            "family": "sdxl",
            "editable": True,
            "system_prompt": "base prompt",
            "use_ban_list": True,
            "ban_list": ["term1", "term2"],
            "version": 1,
        }
    )
    preset, _ = mod.get_vlm_prompt_preset("user-ban", fallback_family="sdxl")
    assert "base prompt" in preset["system_prompt"]
    assert "Avoid using these terms unless they are explicitly visible and necessary: term1, term2." in preset[
        "system_prompt"
    ]


def test_malformed_user_preset_json_does_not_crash_listing(tmp_path: Path, monkeypatch):
    mod = _load_module("gpm_vlm_prompt_preset_adapter")
    user_file = tmp_path / "vlm_prompt_presets.json"
    user_file.write_text("{bad json", encoding="utf-8")
    monkeypatch.setenv("GPM_USER_DATA_DIR", str(tmp_path))
    options = mod.list_vlm_prompt_preset_options()
    ids = {item["id"] for item in options}
    assert "builtin-sdxl" in ids
    preset, status = mod.get_vlm_prompt_preset("builtin-sdxl", fallback_family="sdxl")
    assert preset["id"] == "builtin-sdxl"
    assert "warning:" in status.lower()
