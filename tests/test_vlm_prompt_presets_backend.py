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


presets_mod = _load_module("gpm_vlm_prompt_presets")


def test_builtins_load():
    builtins = presets_mod.get_builtin_presets()
    ids = {item["id"] for item in builtins}
    assert "builtin-sdxl" in ids
    assert "builtin-pony" in ids
    assert "builtin-natural-language" in ids
    for item in builtins:
        assert item["use_ban_list"] is True
        prompt = item["system_prompt"].lower()
        assert "natural, realistic details" in prompt
        assert "vulva or vaginal opening" in prompt
        assert "penis" in prompt


def test_missing_user_file_is_safe(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GPM_USER_DATA_DIR", str(tmp_path))
    presets = presets_mod.get_all_presets()
    assert any(item.get("id") == "builtin-sdxl" for item in presets)
    assert presets_mod.get_last_status() == ""


def test_malformed_user_json_is_safe(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GPM_USER_DATA_DIR", str(tmp_path))
    path = tmp_path / "vlm_prompt_presets.json"
    path.write_text("{invalid-json", encoding="utf-8")
    presets = presets_mod.get_all_presets()
    assert any(item.get("id") == "builtin-sdxl" for item in presets)
    assert "warning:" in presets_mod.get_last_status().lower()


def test_clone_builtin_creates_editable_user_preset(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GPM_USER_DATA_DIR", str(tmp_path))
    cloned = presets_mod.clone_preset("builtin-sdxl", "My SDXL")
    assert cloned["editable"] is True
    assert cloned["family"] == "sdxl"
    assert cloned["name"] == "My SDXL"
    assert cloned["id"].startswith("user-")


def test_save_user_preset_and_delete(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GPM_USER_DATA_DIR", str(tmp_path))
    saved = presets_mod.save_user_preset(
        {
            "id": "user-custom-1",
            "name": "Custom NL",
            "family": "natural",
            "editable": True,
            "system_prompt": "caption style",
            "use_ban_list": True,
            "ban_list": ["score_9", "source_"],
            "version": 1,
        }
    )
    assert saved["id"] == "user-custom-1"
    found = presets_mod.get_preset_by_id("user-custom-1")
    assert found is not None
    updated = presets_mod.save_user_preset(
        {
            "id": "user-custom-1",
            "name": "Custom NL Updated",
            "family": "natural",
            "editable": True,
            "system_prompt": "caption style updated",
            "use_ban_list": False,
            "ban_list": ["  keep_me  ", ""],
            "version": 1,
        }
    )
    assert updated["id"] == "user-custom-1"
    persisted = presets_mod.get_preset_by_id("user-custom-1")
    assert persisted is not None
    assert persisted["name"] == "Custom NL Updated"
    assert persisted["system_prompt"] == "caption style updated"
    assert persisted["use_ban_list"] is False
    assert persisted["ban_list"] == ["keep_me"]
    deleted = presets_mod.delete_user_preset("user-custom-1")
    assert deleted is True
    assert presets_mod.get_preset_by_id("user-custom-1") is None


def test_builtin_overwrite_and_delete_rejected(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GPM_USER_DATA_DIR", str(tmp_path))
    try:
        presets_mod.save_user_preset(
            {
                "id": "builtin-sdxl",
                "name": "Nope",
                "family": "sdxl",
                "editable": True,
                "system_prompt": "x",
                "use_ban_list": False,
                "ban_list": [],
                "version": 1,
            }
        )
        raise AssertionError("expected save to fail for built-in id")
    except ValueError:
        pass
    try:
        presets_mod.delete_user_preset("builtin-pony")
        raise AssertionError("expected delete to fail for built-in id")
    except ValueError:
        pass


def test_invalid_family_and_ban_list_type_rejected(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GPM_USER_DATA_DIR", str(tmp_path))
    ok, err = presets_mod.validate_preset(
        {
            "id": "user-bad-1",
            "name": "Bad",
            "family": "badfamily",
            "editable": True,
            "system_prompt": "x",
            "use_ban_list": False,
            "ban_list": [],
            "version": 1,
        }
    )
    assert ok is False
    assert "family" in err

    ok2, err2 = presets_mod.validate_preset(
        {
            "id": "user-bad-2",
            "name": "Bad2",
            "family": "sdxl",
            "editable": True,
            "system_prompt": "x",
            "use_ban_list": False,
            "ban_list": "not-a-list",
            "version": 1,
        }
    )
    assert ok2 is False
    assert "ban_list" in err2


def test_save_rejects_non_list_ban_list(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GPM_USER_DATA_DIR", str(tmp_path))
    try:
        presets_mod.save_user_preset(
            {
                "id": "user-bad-ban-1",
                "name": "Bad Ban",
                "family": "sdxl",
                "editable": True,
                "system_prompt": "x",
                "use_ban_list": False,
                "ban_list": "not-a-list",
                "version": 1,
            }
        )
        raise AssertionError("expected save to fail for non-list ban_list")
    except ValueError:
        pass


def test_save_rejects_non_string_ban_list_items(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GPM_USER_DATA_DIR", str(tmp_path))
    try:
        presets_mod.save_user_preset(
            {
                "id": "user-bad-ban-2",
                "name": "Bad Ban Items",
                "family": "sdxl",
                "editable": True,
                "system_prompt": "x",
                "use_ban_list": True,
                "ban_list": ["ok", 123],
                "version": 1,
            }
        )
        raise AssertionError("expected save to fail for non-string ban_list items")
    except ValueError:
        pass


def test_missing_use_ban_list_defaults_true_for_user_preset(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GPM_USER_DATA_DIR", str(tmp_path))
    path = tmp_path / "vlm_prompt_presets.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "presets": [
                    {
                        "id": "user-legacy-no-ban-flag",
                        "name": "Legacy",
                        "family": "sdxl",
                        "editable": True,
                        "system_prompt": "legacy prompt",
                        "ban_list": [],
                        "version": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    loaded = presets_mod.get_preset_by_id("user-legacy-no-ban-flag")
    assert loaded is not None
    assert loaded["use_ban_list"] is True
