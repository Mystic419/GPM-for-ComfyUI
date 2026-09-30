import importlib.util
import json
import os
import sys
import types
from pathlib import Path


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
PACKAGE_NAME = "gpm_testpkg_prompt_loader_saver_v2"
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


def _loader():
    mod = _load_module("gpm_vlm_prompt_loader_node")
    return mod.GPMVLMPromptLoaderNode()


def _saver():
    mod = _load_module("gpm_vlm_prompt_saver_node")
    return mod.GPMVLMPromptSaverNode()

def _normalize_saver_return(raw):
    if isinstance(raw, dict) and isinstance(raw.get("result"), tuple):
        return raw["result"]
    return raw


def _empty_bundle():
    return {"ok": False, "status": "no source", "id": "", "name": "", "family": "sdxl", "editable": False}


def test_loader_emits_preset_bundle_and_json(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GPM_USER_DATA_DIR", str(tmp_path))
    node = _loader()
    bundle, status, preset_json = node.load("builtin-sdxl")
    payload = json.loads(preset_json)
    assert bundle["ok"] is True
    assert bundle["id"] == "builtin-sdxl"
    assert bundle["family"] == "sdxl"
    assert bundle["use_ban_list"] is True
    assert bundle["ban_list"] == []
    assert status == "Loaded preset 'builtin-sdxl'."
    assert payload["id"] == "builtin-sdxl"


def test_loader_old_schema_normalizes_fields(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("GPM_USER_DATA_DIR", str(tmp_path))
    loader_mod = _load_module("gpm_vlm_prompt_loader_node")

    def _fake_get(_: str):
        return {
            "id": "legacy-1",
            "name": "Legacy",
            "family": "Natural Language",
            "is_builtin": True,
            "system_prompt": "legacy prompt",
            "ban_list": [" x ", ""],
        }

    monkeypatch.setattr(loader_mod, "get_preset_by_id", _fake_get)
    node = loader_mod.GPMVLMPromptLoaderNode()
    bundle, _, _ = node.load("legacy-1")
    assert bundle["family"] == "natural"
    assert bundle["editable"] is False
    assert bundle["use_ban_list"] is True
    assert bundle["ban_list"] == ["x"]
    assert bundle["version"] == 1


def test_saver_load_from_source_reads_bundle_and_does_not_save(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GPM_USER_DATA_DIR", str(tmp_path))
    presets = _load_module("gpm_vlm_prompt_presets")
    loader = _loader()
    saver = _saver()
    bundle, _, _ = loader.load("builtin-pony")
    raw = saver.run(
        "LOAD FROM SOURCE",
        bundle,
        "",
        "",
        "KEEP",
        "",
        unique_id="node-a",
    )
    status, saved_id, preset_json = _normalize_saver_return(raw)
    payload = json.loads(preset_json)
    assert isinstance(raw, dict)
    assert raw["ui"]["gpm_vlm_prompt_saver_loaded_name"][0] == bundle["name"]
    assert raw["ui"]["gpm_vlm_prompt_saver_loaded_system_prompt"][0] == bundle["system_prompt"]
    assert raw["ui"]["gpm_vlm_prompt_saver_loaded_use_ban_list"][0] == "ON"
    assert raw["ui"]["gpm_vlm_prompt_saver_loaded_ban_list"][0] == ""
    assert "loaded source 'builtin-pony'. ui payload emitted." in status.lower()
    assert saved_id == "builtin-pony"
    assert payload["ok"] is True
    assert payload["action"] == "LOAD FROM SOURCE"
    assert payload["loaded_id"] == "builtin-pony"
    assert payload["loaded_name"] == bundle["name"]
    assert payload["loaded_system_prompt"] == bundle["system_prompt"]
    assert payload["loaded_prompt_length"] == len(bundle["system_prompt"])
    assert payload["loaded_use_ban_list"] is True
    assert payload["loaded_ban_list"] == bundle["ban_list"]
    assert payload["loaded_ban_list_count"] == len(bundle["ban_list"])
    assert payload["ui_payload_emitted"] is True
    assert presets.get_preset_by_id("user-should-not-exist") is None


def test_node_local_load_helper_stores_the_same_source_state(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GPM_USER_DATA_DIR", str(tmp_path))
    saver_mod = _load_module("gpm_vlm_prompt_saver_node")
    monkeypatch.setattr(saver_mod, "STATE_FILE_PATH", tmp_path / "saver-state.json")
    loader = _loader()
    bundle, _, _ = loader.load("builtin-natural-language")

    loaded = saver_mod.load_source_into_saver_state("node-local-load", bundle)
    state = saver_mod._get_state("node-local-load")

    assert loaded["ok"] is True
    assert loaded["id"] == "builtin-natural-language"
    assert state["loaded_source_id"] == "builtin-natural-language"
    assert state["loaded_source_system_prompt"] == bundle["system_prompt"]


def test_loader_explicit_false_use_ban_list_is_preserved(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("GPM_USER_DATA_DIR", str(tmp_path))
    loader_mod = _load_module("gpm_vlm_prompt_loader_node")

    def _fake_get(_: str):
        return {
            "id": "legacy-false",
            "name": "Legacy False",
            "family": "SDXL",
            "system_prompt": "prompt",
            "use_ban_list": False,
            "ban_list": [],
        }

    monkeypatch.setattr(loader_mod, "get_preset_by_id", _fake_get)
    node = loader_mod.GPMVLMPromptLoaderNode()
    bundle, _, _ = node.load("legacy-false")
    assert bundle["use_ban_list"] is False


def test_saver_save_as_new_uses_edited_fields_and_not_live_source(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GPM_USER_DATA_DIR", str(tmp_path))
    loader = _loader()
    saver = _saver()
    bundle, _, _ = loader.load("builtin-sdxl")
    _normalize_saver_return(saver.run("LOAD FROM SOURCE", bundle, "", "", "KEEP", "", unique_id="node-b"))
    status, saved_id, preset_json = _normalize_saver_return(saver.run(
        "SAVE AS NEW USER PRESET",
        _empty_bundle(),
        "Edited Name",
        "edited prompt",
        "ON",
        "one\ntwo",
        unique_id="node-b",
    ))
    payload = json.loads(preset_json)
    assert "saved new user preset" in status.lower()
    assert saved_id.startswith("user-")
    assert payload["name"] == "Edited Name"
    assert payload["system_prompt"] == "edited prompt"
    assert payload["family"] == "sdxl"
    assert payload["ban_list"] == ["one", "two"]
    assert payload["use_ban_list"] is True


def test_saver_save_as_new_blank_prompt_preserves_loaded_prompt_and_family(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GPM_USER_DATA_DIR", str(tmp_path))
    loader = _loader()
    saver = _saver()
    bundle, _, _ = loader.load("builtin-natural-language")
    _normalize_saver_return(saver.run("LOAD FROM SOURCE", bundle, "", "", "KEEP", "", unique_id="node-c"))
    _, saved_id, preset_json = _normalize_saver_return(saver.run(
        "SAVE AS NEW USER PRESET",
        _empty_bundle(),
        "",
        "",
        "KEEP",
        "",
        unique_id="node-c",
    ))
    payload = json.loads(preset_json)
    assert saved_id.startswith("user-")
    assert payload["family"] == "natural"
    assert payload["system_prompt"] != ""


def test_update_existing_uses_loaded_metadata_and_refuses_builtin(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GPM_USER_DATA_DIR", str(tmp_path))
    presets = _load_module("gpm_vlm_prompt_presets")
    saver = _saver()

    # Built-in loaded then update should refuse.
    loader = _loader()
    built_in_bundle, _, _ = loader.load("builtin-sdxl")
    _normalize_saver_return(saver.run("LOAD FROM SOURCE", built_in_bundle, "", "", "KEEP", "", unique_id="node-d"))
    status, _, preset_json = _normalize_saver_return(saver.run(
        "UPDATE EXISTING USER PRESET",
        _empty_bundle(),
        "Nope",
        "Nope",
        "OFF",
        "",
        unique_id="node-d",
    ))
    payload = json.loads(preset_json)
    assert "cannot update built-in preset" in status.lower()
    assert payload["ok"] is False

    # User preset loaded then update should keep same id/family.
    presets.save_user_preset(
        {
            "id": "user-edit-1",
            "name": "Edit Me",
            "family": "pony",
            "editable": True,
            "system_prompt": "old",
            "use_ban_list": False,
            "ban_list": [],
            "version": 1,
        }
    )
    user_bundle, _, _ = loader.load("user-edit-1")
    _normalize_saver_return(saver.run("LOAD FROM SOURCE", user_bundle, "", "", "KEEP", "", unique_id="node-e"))
    status2, saved_id2, preset_json2 = _normalize_saver_return(saver.run(
        "UPDATE EXISTING USER PRESET",
        _empty_bundle(),
        "Edited User",
        "new prompt",
        "ON",
        "x",
        unique_id="node-e",
    ))
    payload2 = json.loads(preset_json2)
    assert "updated user preset" in status2.lower()
    assert saved_id2 == "user-edit-1"
    assert payload2["id"] == "user-edit-1"
    assert payload2["family"] == "pony"
    assert payload2["name"] == "Edited User"
    assert payload2["system_prompt"] == "new prompt"


def test_delete_refuses_builtin_and_deletes_user(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GPM_USER_DATA_DIR", str(tmp_path))
    presets = _load_module("gpm_vlm_prompt_presets")
    loader = _loader()
    saver = _saver()

    builtin_bundle, _, _ = loader.load("builtin-pony")
    _normalize_saver_return(saver.run("LOAD FROM SOURCE", builtin_bundle, "", "", "KEEP", "", unique_id="node-f"))
    status, _, preset_json = _normalize_saver_return(
        saver.run("DELETE USER PRESET", _empty_bundle(), "", "", "KEEP", "", unique_id="node-f")
    )
    payload = json.loads(preset_json)
    assert "cannot delete built-in preset" in status.lower()
    assert payload["deleted"] is False

    presets.save_user_preset(
        {
            "id": "user-delete-1",
            "name": "Delete Me",
            "family": "sdxl",
            "editable": True,
            "system_prompt": "x",
            "use_ban_list": False,
            "ban_list": [],
            "version": 1,
        }
    )
    user_bundle, _, _ = loader.load("user-delete-1")
    _normalize_saver_return(saver.run("LOAD FROM SOURCE", user_bundle, "", "", "KEEP", "", unique_id="node-g"))
    status2, _, preset_json2 = _normalize_saver_return(saver.run(
        "DELETE USER PRESET", _empty_bundle(), "", "", "KEEP", "", unique_id="node-g"
    ))
    payload2 = json.loads(preset_json2)
    assert "deleted user preset" in status2.lower()
    assert payload2["deleted"] is True
    assert presets.get_preset_by_id("user-delete-1") is None


def test_source_change_after_load_does_not_overwrite_save_fields(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GPM_USER_DATA_DIR", str(tmp_path))
    loader = _loader()
    saver = _saver()
    bundle_sdxl, _, _ = loader.load("builtin-sdxl")
    _normalize_saver_return(saver.run("LOAD FROM SOURCE", bundle_sdxl, "", "", "KEEP", "", unique_id="node-h"))
    bundle_pony, _, _ = loader.load("builtin-pony")
    _, _, preset_json = _normalize_saver_return(saver.run(
        "SAVE AS NEW USER PRESET",
        bundle_pony,
        "Keep My Edits",
        "my custom prompt",
        "OFF",
        "",
        unique_id="node-h",
    ))
    payload = json.loads(preset_json)
    assert payload["name"] == "Keep My Edits"
    assert payload["system_prompt"] == "my custom prompt"
    assert payload["family"] == "sdxl"


def test_blank_prompt_without_loaded_source_prompt_errors(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GPM_USER_DATA_DIR", str(tmp_path))
    saver = _saver()
    status, saved_id, preset_json = _normalize_saver_return(saver.run(
        "SAVE AS NEW USER PRESET",
        _empty_bundle(),
        "Name",
        "",
        "OFF",
        "",
        unique_id="node-i",
    ))
    payload = json.loads(preset_json)
    assert "system_prompt is blank" in status
    assert saved_id == ""
    assert payload["ok"] is False


def test_node_registration_exposes_loader_saver_and_hides_editor():
    init_path = os.path.join(ROOT_DIR, "__init__.py")
    with open(init_path, "r", encoding="utf-8") as handle:
        init_text = handle.read()
    assert "GPM VLM Prompt Loader" in init_text
    assert "GPM VLM Prompt Saver" in init_text
    assert '"GPM VLM Prompt Editor"' not in init_text


def test_loader_saver_and_package_import_smoke():
    loader_mod = _load_module("gpm_vlm_prompt_loader_node")
    saver_mod = _load_module("gpm_vlm_prompt_saver_node")
    assert hasattr(loader_mod, "GPMVLMPromptLoaderNode")
    assert hasattr(saver_mod, "GPMVLMPromptSaverNode")
    init_path = os.path.join(ROOT_DIR, "__init__.py")
    with open(init_path, "r", encoding="utf-8") as handle:
        init_text = handle.read()
    assert "GPM VLM Prompt Loader" in init_text
    assert "GPM VLM Prompt Saver" in init_text
    assert 'WEB_DIRECTORY = "./web"' in init_text


def test_web_directory_export_and_prompt_saver_js_smoke():
    root_init_path = Path(ROOT_DIR) / "__init__.py"
    root_init_text = root_init_path.read_text(encoding="utf-8")
    assert 'WEB_DIRECTORY = "./web"' in root_init_text
    assert '"WEB_DIRECTORY"' in root_init_text

    src_init_path = Path(ROOT_DIR) / "src" / "gallery_prompt_manager" / "__init__.py"
    src_init_text = src_init_path.read_text(encoding="utf-8")
    assert 'WEB_DIRECTORY = "./web"' in src_init_text
    assert '"WEB_DIRECTORY"' in src_init_text

    web_dir = Path(ROOT_DIR) / "web"
    prompt_saver_js = web_dir / "gpm_vlm_prompt_saver.js"
    assert web_dir.is_dir()
    assert prompt_saver_js.is_file()

    js_text = prompt_saver_js.read_text(encoding="utf-8")
    assert "const DEBUG_GPM_VLM_PROMPT_SAVER = false;" in js_text
    assert "const DEBUG_GPM_VLM_PROMPT_SAVER_EXECUTION = true;" in js_text
    assert '[GPM VLM Prompt Saver] script loaded' in js_text
    assert '[GPM VLM Prompt Saver] hooked node def' in js_text
    assert "function populateSaverWidgets(node)" in js_text
    assert "nodeType.prototype.onConfigure = function (info)" in js_text
    assert "requestAnimationFrame(() => {" in js_text
    assert "nodeType.prototype.onNodeCreated = function ()" in js_text
    assert '[GPM VLM Prompt Saver] hooked node def' in js_text
    assert '[GPM VLM Prompt Saver] loaded source stored:' in js_text
    assert '[GPM VLM Prompt Saver] loaded source applied to widgets:' in js_text
    assert 'api.addEventListener("executed"' in js_text
    assert "window.gpmDebugPromptSaverNodes = function ()" in js_text
    assert "window.gpmApplyPromptSaverLoadedState = function ()" in js_text
    assert "function loadSelectedPresetIntoSaverFields(node)" in js_text
    assert 'api.fetchApi("/gpm/vlm/prompt-saver/load"' in js_text
    assert '"load_preset"' in js_text
    assert "function installLoadButton(node)" in js_text
    assert "nodeCreated(node)" in js_text
    assert "loadedGraphNode(node)" in js_text
    assert 'if (DEBUG_GPM_VLM_PROMPT_SAVER) {' in js_text
    assert 'console.log("[GPM VLM Prompt Saver] checking node def", {' not in js_text
    assert 'console.log("[GPM VLM Prompt Saver] node candidate", {' not in js_text
