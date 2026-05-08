import importlib.util
import os
import sys
import types
from pathlib import Path


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
PACKAGE_NAME = "gpm_testpkg_internal_unload"
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


def test_internal_nodes_hide_lifecycle_controls():
    scanner_mod = _load_module("gpm_vlm_scanner_internal_node")
    base_inputs = scanner_mod.GPMVLMScannerInternal.INPUT_TYPES()["required"]
    adv_inputs = scanner_mod.GPMVLMScannerInternalAdvanced.INPUT_TYPES()["required"]

    assert "unload_on_complete" not in base_inputs
    assert "execution_mode" not in base_inputs
    assert "unload_on_complete" not in adv_inputs
    assert "execution_mode" not in adv_inputs
    assert "keep_model_loaded" not in base_inputs
    assert "keep_model_loaded" not in adv_inputs


def test_advanced_node_hardcodes_subprocess_unload_and_keep_loaded():
    scanner_mod = _load_module("gpm_vlm_scanner_internal_node")
    node = scanner_mod.GPMVLMScannerInternalAdvanced()
    captured = {}

    def fake_run_internal_scan(**kwargs):
        captured.update(kwargs)
        return "{}", "ok"

    original_run = scanner_mod._run_internal_scan
    scanner_mod._run_internal_scan = fake_run_internal_scan
    try:
        node.scan(
            root_folder=".",
            preset_id="builtin-sdxl",
            overwrite_mode="SKIP_EXISTING",
            scan_limit=0,
            write_scan_report="OFF",
            model_name="m.gguf",
            mmproj_name="mmproj.gguf",
            timeout_seconds=30,
            n_ctx=4096,
            n_gpu_layers=-1,
            temperature=0.2,
            top_p=0.95,
            max_tokens=512,
            threads=0,
            batch_size=512,
            debug_mode="OFF",
        )
    finally:
        scanner_mod._run_internal_scan = original_run
    assert captured["keep_model_loaded"] is True
    assert captured["unload_on_complete"] is True
    assert captured["execution_mode"].startswith("SUBPROCESS")


def test_basic_node_hardcodes_unload_on_complete_and_keep_loaded():
    scanner_mod = _load_module("gpm_vlm_scanner_internal_node")
    node = scanner_mod.GPMVLMScannerInternal()
    captured = {}

    def fake_run_internal_scan(**kwargs):
        captured.update(kwargs)
        return "{}", "ok"

    original_run = scanner_mod._run_internal_scan
    scanner_mod._run_internal_scan = fake_run_internal_scan
    try:
        node.scan(
            root_folder=".",
            preset_id="builtin-sdxl",
            overwrite_mode="SKIP_EXISTING",
            scan_limit=0,
            write_scan_report="OFF",
            model_name="m.gguf",
            mmproj_name="mmproj.gguf",
            timeout_seconds=30,
            debug_mode="OFF",
        )
    finally:
        scanner_mod._run_internal_scan = original_run

    assert captured["keep_model_loaded"] is True
    assert captured["unload_on_complete"] is True
    assert captured["execution_mode"].startswith("SUBPROCESS")


def test_run_internal_scan_passes_unload_flag_to_backend():
    scanner_mod = _load_module("gpm_vlm_scanner_internal_node")

    class _Store:
        def get_preset(self, _preset_id):
            return {"id": "builtin-sdxl", "temperature": 0.2, "top_p": 0.95, "max_tokens": 512}

    scanner_mod.GPMVLMPresetStore = _Store
    scanner_mod.get_preset_generation_settings = lambda _preset: (0.2, 0.95, 512)
    captured = {}

    def _fake_scan_images_with_preset(**kwargs):
        captured.update(kwargs)
        return {"ok": True, "processed": 1}

    scanner_mod.scan_images_with_preset = _fake_scan_images_with_preset

    summary_json, _ = scanner_mod._run_internal_scan(
        root_folder=".",
        preset_id="builtin-sdxl",
        overwrite_mode="SKIP_EXISTING",
        scan_limit=0,
        write_scan_report="OFF",
        model_name="m.gguf",
        mmproj_name="mmproj.gguf",
        timeout_seconds=30,
        n_ctx=4096,
        n_gpu_layers=-1,
        temperature=0.2,
        top_p=0.95,
        max_tokens=512,
        threads=0,
        batch_size=512,
        keep_model_loaded=True,
        unload_on_complete=True,
        debug_mode=False,
        node_runtime_lifecycle_mode="test_mode",
        execution_mode="IN_PROCESS (Advanced: faster reuse, may retain VRAM)",
    )
    assert captured["internal_keep_model_loaded"] is True
    assert captured["internal_unload_on_complete"] is True
    assert "\"internal_keep_model_loaded_requested\": true" in summary_json
    assert "\"internal_unload_on_complete_requested\": true" in summary_json
    assert "\"node_runtime_lifecycle_mode\": \"test_mode\"" in summary_json
    assert "\"internal_execution_mode\": \"in_process\"" in summary_json


def test_basic_summary_reports_lifecycle_defaults():
    scanner_mod = _load_module("gpm_vlm_scanner_internal_node")
    node = scanner_mod.GPMVLMScannerInternal()

    class _Store:
        def get_preset(self, _preset_id):
            return {"id": "builtin-sdxl", "temperature": 0.2, "top_p": 0.95, "max_tokens": 512}

    scanner_mod.GPMVLMPresetStore = _Store
    scanner_mod.get_preset_generation_settings = lambda _preset: (0.2, 0.95, 512)
    scanner_mod.scan_images_with_preset = lambda **_kwargs: {"ok": True, "processed": 1}
    original_subprocess_run = scanner_mod._run_internal_scan_subprocess
    scanner_mod._run_internal_scan_subprocess = (
        lambda **_kwargs: {"ok": True, "processed": 1, "worker_return_code": 0, "worker_elapsed_seconds": 0.01}
    )

    try:
        summary_json, _ = node.scan(
            root_folder=".",
            preset_id="builtin-sdxl",
            overwrite_mode="SKIP_EXISTING",
            scan_limit=0,
            write_scan_report="OFF",
            model_name="m.gguf",
            mmproj_name="mmproj.gguf",
            timeout_seconds=30,
            debug_mode="OFF",
        )
    finally:
        scanner_mod._run_internal_scan_subprocess = original_subprocess_run
    assert "\"internal_keep_model_loaded_requested\": true" in summary_json
    assert "\"internal_unload_on_complete_requested\": true" in summary_json
    assert "\"node_runtime_lifecycle_mode\": \"basic_internal_fixed_defaults\"" in summary_json
    assert "\"internal_execution_mode\": \"subprocess\"" in summary_json


def test_advanced_summary_reports_fixed_subprocess_lifecycle():
    scanner_mod = _load_module("gpm_vlm_scanner_internal_node")
    node = scanner_mod.GPMVLMScannerInternalAdvanced()

    class _Store:
        def get_preset(self, _preset_id):
            return {"id": "builtin-sdxl", "temperature": 0.2, "top_p": 0.95, "max_tokens": 512}

    scanner_mod.GPMVLMPresetStore = _Store
    scanner_mod.get_preset_generation_settings = lambda _preset: (0.2, 0.95, 512)
    scanner_mod.scan_images_with_preset = lambda **_kwargs: {"ok": True, "processed": 1}

    summary_json, _ = node.scan(
        root_folder=".",
        preset_id="builtin-sdxl",
        overwrite_mode="SKIP_EXISTING",
        scan_limit=0,
        write_scan_report="OFF",
        model_name="m.gguf",
        mmproj_name="mmproj.gguf",
        timeout_seconds=30,
        n_ctx=4096,
        n_gpu_layers=-1,
        temperature=0.2,
        top_p=0.95,
        max_tokens=512,
        threads=0,
        batch_size=512,
        debug_mode="OFF",
    )
    assert "\"internal_keep_model_loaded_requested\": true" in summary_json
    assert "\"internal_unload_on_complete_requested\": true" in summary_json
    assert "\"node_runtime_lifecycle_mode\": \"advanced_internal_subprocess_fixed_defaults\"" in summary_json
    assert "\"internal_execution_mode\": \"subprocess\"" in summary_json


def test_build_internal_scan_request_payload_shape():
    scanner_mod = _load_module("gpm_vlm_scanner_internal_node")
    payload = scanner_mod._build_internal_scan_request(
        root_folder="C:/imgs",
        preset_id="builtin-sdxl",
        overwrite_mode="SKIP_EXISTING",
        scan_limit=12,
        model_name="model.gguf",
        mmproj_name="mmproj.gguf",
        timeout_seconds=45,
        n_ctx=4096,
        n_gpu_layers=-1,
        temperature=0.2,
        top_p=0.95,
        max_tokens=512,
        threads=0,
        batch_size=512,
        debug_mode=True,
        write_scan_report="OFF",
        model_path_resolved="D:/ComfyUI/models/LLM/gguf/model.gguf",
        mmproj_path_resolved="D:/ComfyUI/models/LLM/gguf/mmproj.gguf",
    )
    assert payload["root_folder"] == "C:/imgs"
    assert payload["preset_id"] == "builtin-sdxl"
    assert payload["model_name"] == "model.gguf"
    assert payload["mmproj_name"] == "mmproj.gguf"
    assert payload["model_path_resolved"].endswith("model.gguf")
    assert payload["mmproj_path_resolved"].endswith("mmproj.gguf")
    assert payload["n_ctx"] == 4096
    assert payload["n_gpu_layers"] == -1


def test_run_internal_scan_resolves_paths_before_subprocess_launch():
    scanner_mod = _load_module("gpm_vlm_scanner_internal_node")
    captured = {}

    scanner_mod.resolve_model_and_mmproj_paths = lambda **_kwargs: (
        Path("D:/ComfyUI/models/LLM/gguf/model.gguf"),
        Path("D:/ComfyUI/models/LLM/gguf/mmproj.gguf"),
        "",
    )

    def _fake_subprocess_runner(**kwargs):
        captured.update(kwargs)
        return {"ok": True, "processed": 1}

    scanner_mod._run_internal_scan_subprocess = _fake_subprocess_runner
    summary_json, _ = scanner_mod._run_internal_scan(
        root_folder=".",
        preset_id="builtin-sdxl",
        overwrite_mode="SKIP_EXISTING",
        scan_limit=0,
        write_scan_report="OFF",
        model_name="model.gguf",
        mmproj_name="mmproj.gguf",
        timeout_seconds=30,
        n_ctx=4096,
        n_gpu_layers=-1,
        temperature=0.2,
        top_p=0.95,
        max_tokens=512,
        threads=0,
        batch_size=512,
        keep_model_loaded=True,
        unload_on_complete=True,
        debug_mode=False,
        node_runtime_lifecycle_mode="test_mode",
        execution_mode="SUBPROCESS (Recommended: releases VRAM after scan)",
    )
    assert captured["request"]["model_path_resolved"].endswith("model.gguf")
    assert captured["request"]["mmproj_path_resolved"].endswith("mmproj.gguf")
    assert "\"ok\": true" in summary_json.lower()


def test_run_internal_scan_returns_resolve_error_without_worker_launch():
    scanner_mod = _load_module("gpm_vlm_scanner_internal_node")
    scanner_mod.resolve_model_and_mmproj_paths = lambda **_kwargs: (None, None, "selected GGUF model was not found: bad")
    launched = {"value": False}
    scanner_mod._run_internal_scan_subprocess = lambda **_kwargs: (launched.__setitem__("value", True) or {})

    summary_json, _ = scanner_mod._run_internal_scan(
        root_folder=".",
        preset_id="builtin-sdxl",
        overwrite_mode="SKIP_EXISTING",
        scan_limit=0,
        write_scan_report="OFF",
        model_name="model.gguf",
        mmproj_name="mmproj.gguf",
        timeout_seconds=30,
        n_ctx=4096,
        n_gpu_layers=-1,
        temperature=0.2,
        top_p=0.95,
        max_tokens=512,
        threads=0,
        batch_size=512,
        keep_model_loaded=True,
        unload_on_complete=True,
        debug_mode=False,
        node_runtime_lifecycle_mode="test_mode",
        execution_mode="SUBPROCESS (Recommended: releases VRAM after scan)",
    )
    assert launched["value"] is False
    assert "selected GGUF model was not found: bad" in summary_json


def test_subprocess_path_handles_nonzero_exit():
    scanner_mod = _load_module("gpm_vlm_scanner_internal_node")

    class _Completed:
        returncode = 1
        stdout = "worker stdout text"
        stderr = "worker stderr text"

    def _fake_run(*args, **kwargs):
        cmd = args[0]
        output_json_path = ""
        for idx, part in enumerate(cmd):
            if part == "--output-json" and idx + 1 < len(cmd):
                output_json_path = cmd[idx + 1]
                break
        Path(output_json_path).write_text(
            '{"ok": false, "error": "real worker error", "processed": 0}',
            encoding="utf-8",
        )
        return _Completed()

    scanner_mod.subprocess.run = _fake_run
    summary = scanner_mod._run_internal_scan_subprocess(
        request={"preset_id": "builtin-sdxl"},
        timeout_seconds=10,
    )
    assert summary["ok"] is False
    assert summary["error"] == "real worker error"
    assert summary["worker_return_code"] == 1
    assert summary["worker_failed"] is True
    assert "nonzero code (1)" in summary["worker_exit_error"]
    assert "worker stderr text" in summary["worker_stderr_tail"]


def test_subprocess_path_handles_missing_output_json():
    scanner_mod = _load_module("gpm_vlm_scanner_internal_node")

    class _Completed:
        returncode = 0
        stdout = "worker stdout"
        stderr = ""

    scanner_mod.subprocess.run = lambda *args, **kwargs: _Completed()
    summary = scanner_mod._run_internal_scan_subprocess(
        request={"preset_id": "builtin-sdxl"},
        timeout_seconds=10,
    )
    assert summary["ok"] is False
    assert "did not produce output JSON" in summary["error"]
    assert summary["worker_return_code"] == 0


def test_backend_forces_release_when_internal_unload_on_complete_true():
    backend_mod = _load_module("gpm_vlm_backend")

    class _Runtime:
        def __init__(self):
            self.stop_calls = 0

        def start(self):
            return True, ""

        def stop(self):
            self.stop_calls += 1

        def generate(self, _image_path, _preset):
            return "p", "s", ""

        def summary_metadata(self):
            return {}

    runtime = _Runtime()
    captured = {"release_calls": 0}

    backend_mod._normalize_root_folder = lambda _root_folder: Path(".")
    backend_mod._discover_images = lambda _root: []
    backend_mod._build_runtime = lambda **_kwargs: (runtime, "")
    backend_mod.GPMGGUFInternalRuntime.release_instance_and_cache = classmethod(
        lambda cls, rt, reason="": (
            captured.__setitem__("release_calls", captured["release_calls"] + 1)
            or {"requested": True, "active_runtime_released": True, "cached_runtime_cleared": True, "reason": reason}
        )
    )

    summary = backend_mod.scan_images_with_preset(
        root_folder=".",
        preset={"id": "builtin-sdxl", "name": "SDXL", "family": "SDXL"},
        runtime_mode="internal",
        internal_model_name="m.gguf",
        internal_mmproj_name="mmproj.gguf",
        internal_keep_model_loaded=True,
        internal_unload_on_complete=True,
    )
    assert runtime.stop_calls == 1
    assert captured["release_calls"] == 1
    assert summary["unload_on_complete"] is True
    assert summary["runtime_cleanup"]["active_runtime_released"] is True


def test_backend_keeps_runtime_when_unload_on_complete_false():
    backend_mod = _load_module("gpm_vlm_backend")

    class _Runtime:
        def __init__(self):
            self.stop_calls = 0

        def start(self):
            return True, ""

        def stop(self):
            self.stop_calls += 1

        def generate(self, _image_path, _preset):
            return "p", "s", ""

        def summary_metadata(self):
            return {}

    runtime = _Runtime()
    captured = {"release_calls": 0}

    backend_mod._normalize_root_folder = lambda _root_folder: Path(".")
    backend_mod._discover_images = lambda _root: []
    backend_mod._build_runtime = lambda **_kwargs: (runtime, "")
    backend_mod.GPMGGUFInternalRuntime.release_instance_and_cache = classmethod(
        lambda cls, rt, reason="": (captured.__setitem__("release_calls", captured["release_calls"] + 1) or {})
    )

    summary = backend_mod.scan_images_with_preset(
        root_folder=".",
        preset={"id": "builtin-sdxl", "name": "SDXL", "family": "SDXL"},
        runtime_mode="internal",
        internal_model_name="m.gguf",
        internal_mmproj_name="mmproj.gguf",
        internal_keep_model_loaded=True,
        internal_unload_on_complete=False,
    )
    assert runtime.stop_calls == 1
    assert captured["release_calls"] == 0
    assert "runtime_cleanup" not in summary


def test_backend_runtime_uses_absolute_override_paths_without_name_resolution():
    backend_mod = _load_module("gpm_vlm_backend")

    model_path = Path(__file__).resolve()
    mmproj_path = Path(__file__).resolve()
    called = {"resolve_called": False}

    def _unexpected_resolve(**_kwargs):
        called["resolve_called"] = True
        return None, None, "should not be called"

    backend_mod.resolve_model_and_mmproj_paths = _unexpected_resolve
    runtime, runtime_error = backend_mod._build_runtime(
        runtime_mode="internal",
        gguf_api_url="http://127.0.0.1:1234/v1/chat/completions",
        gguf_model_name="",
        timeout_seconds=10,
        internal_model_name="dropdown-model.gguf",
        internal_mmproj_name="dropdown-mmproj.gguf",
        internal_model_path_override=str(model_path),
        internal_mmproj_path_override=str(mmproj_path),
    )
    assert runtime_error == ""
    assert runtime is not None
    assert called["resolve_called"] is False


def test_backend_runtime_missing_absolute_override_model_path_has_clear_error():
    backend_mod = _load_module("gpm_vlm_backend")
    missing_model_path = Path(__file__).resolve().parent / "__missing_model.gguf"
    runtime, runtime_error = backend_mod._build_runtime(
        runtime_mode="internal",
        gguf_api_url="http://127.0.0.1:1234/v1/chat/completions",
        gguf_model_name="",
        timeout_seconds=10,
        internal_model_name="dropdown-model.gguf",
        internal_mmproj_name="dropdown-mmproj.gguf",
        internal_model_path_override=str(missing_model_path),
        internal_mmproj_path_override=str(Path(__file__).resolve()),
    )
    assert runtime is None
    assert str(runtime_error).startswith("worker resolved model path was not found:")
