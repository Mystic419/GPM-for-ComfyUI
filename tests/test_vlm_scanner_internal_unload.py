import importlib.util
import json
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
    assert "keep_model_loaded" not in base_inputs
    assert "timeout_seconds" not in base_inputs
    assert "unload_on_complete" not in adv_inputs
    assert "execution_mode" not in adv_inputs
    assert "keep_model_loaded" not in adv_inputs
    assert "timeout_seconds" not in adv_inputs


def test_basic_internal_scanner_offers_all_builtin_prompt_families_by_readable_name():
    scanner_mod = _load_module("gpm_vlm_scanner_internal_node")
    prompt_choices = scanner_mod.GPMVLMScannerInternal.INPUT_TYPES()["required"]["prompt_preset"][0]
    assert "SDXL" in prompt_choices
    assert "Pony" in prompt_choices
    assert "Natural Language" in prompt_choices


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

        def poll(self):
            return self.returncode

    def _fake_popen(*args, **kwargs):
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
        kwargs["stderr"].write("worker stderr text")
        return _Completed()

    original_popen = scanner_mod.subprocess.Popen
    scanner_mod.subprocess.Popen = _fake_popen
    try:
        summary = scanner_mod._run_internal_scan_subprocess(request={"preset_id": "builtin-sdxl"})
    finally:
        scanner_mod.subprocess.Popen = original_popen
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

        def poll(self):
            return self.returncode

    original_popen = scanner_mod.subprocess.Popen
    scanner_mod.subprocess.Popen = lambda *args, **kwargs: _Completed()
    try:
        summary = scanner_mod._run_internal_scan_subprocess(request={"preset_id": "builtin-sdxl"})
    finally:
        scanner_mod.subprocess.Popen = original_popen
    assert summary["ok"] is False
    assert "did not produce output JSON" in summary["error"]
    assert summary["worker_return_code"] == 0


def test_unlimited_subprocess_scan_uses_one_worker_and_one_model_load():
    scanner_mod = _load_module("gpm_vlm_scanner_internal_node")
    captured_limits = []
    scanner_mod.resolve_model_and_mmproj_paths = lambda **_kwargs: (
        Path("D:/ComfyUI/models/LLM/gguf/model.gguf"),
        Path("D:/ComfyUI/models/LLM/gguf/mmproj.gguf"),
        "",
    )

    def _fake_worker(*, request, timeout_seconds):
        captured_limits.append(request["scan_limit"])
        return {
            "ok": True,
            "total_found": 100,
            "processed": 50,
            "failed": 0,
            "skipped": 0,
            "failures": [],
            "warnings": [],
            "batch_candidates_started": 100,
            "batch_has_more": False,
            "worker_elapsed_seconds": 1.0,
        }

    scanner_mod._run_internal_scan_subprocess = _fake_worker
    summary_json, _ = scanner_mod._run_internal_scan(
        root_folder=".",
        preset_id="builtin-sdxl",
        overwrite_mode="SKIP_EXISTING",
        scan_limit=0,
        write_scan_report="OFF",
        preset_payload={"id": "builtin-sdxl", "family": "SDXL"},
        model_name="model.gguf",
        mmproj_name="mmproj.gguf",
        timeout_seconds=180,
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
        execution_mode="SUBPROCESS",
    )
    summary = json.loads(summary_json)
    assert captured_limits == [0]
    assert summary["processed"] == 50


def test_subprocess_scan_defers_stalled_candidate_and_continues_remaining_images():
    scanner_mod = _load_module("gpm_vlm_scanner_internal_node")
    requests = []

    def _fake_worker(*, request, timeout_seconds):
        requests.append(dict(request))
        if len(requests) == 1:
            return {"ok": False, "worker_stalled": True, "worker_progress_seen": True}
        return {
            "ok": True,
            "batch_candidates_started": 2,
            "batch_has_more": False,
            "processed": 2,
            "failed": 0,
            "skipped": 0,
            "total_found": 3,
            "warnings": [],
            "failures": [],
        }

    original_runner = scanner_mod._run_internal_scan_subprocess
    original_resolve = scanner_mod.resolve_model_and_mmproj_paths
    scanner_mod._run_internal_scan_subprocess = _fake_worker
    scanner_mod.resolve_model_and_mmproj_paths = lambda **_kwargs: (Path("model.gguf"), Path("mmproj.gguf"), "")
    try:
        summary_json, _ = scanner_mod._run_internal_scan(
            root_folder=".", preset_id="builtin-sdxl", overwrite_mode="SKIP_EXISTING", scan_limit=0,
            write_scan_report="OFF", preset_payload={"id": "builtin-sdxl", "family": "SDXL"},
            model_name="model.gguf", mmproj_name="mmproj.gguf", timeout_seconds=180,
            n_ctx=4096, n_gpu_layers=-1, temperature=0.2, top_p=0.95, max_tokens=512,
            threads=0, batch_size=512, keep_model_loaded=True, unload_on_complete=True,
            debug_mode=False, node_runtime_lifecycle_mode="test_mode", execution_mode="SUBPROCESS",
        )
    finally:
        scanner_mod._run_internal_scan_subprocess = original_runner
        scanner_mod.resolve_model_and_mmproj_paths = original_resolve

    summary = json.loads(summary_json)
    assert [request["skip_first_eligible"] for request in requests] == [0, 1]
    assert summary["processed"] == 2
    assert summary["deferred_timeout_candidates"] == 1


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


def test_backend_retries_transient_strict_json_response_before_failing(tmp_path):
    backend_mod = _load_module("gpm_vlm_backend")
    image_path = tmp_path / "retry-me.png"
    image_path.write_bytes(b"not-decoded-by-fake-runtime")

    class _Runtime:
        def __init__(self):
            self.generate_calls = 0

        def start(self):
            return True, ""

        def stop(self):
            pass

        def generate(self, _image_path, _preset):
            self.generate_calls += 1
            if self.generate_calls < 3:
                return "", "", "internal runtime did not return strict JSON: incomplete response"
            return "person prompt", "scene prompt", ""

        def summary_metadata(self):
            return {}

    runtime = _Runtime()
    original_discover = backend_mod._discover_images
    original_build_runtime = backend_mod._build_runtime
    backend_mod._discover_images = lambda _root: [image_path]
    backend_mod._build_runtime = lambda **_kwargs: (runtime, "")
    try:
        summary = backend_mod.scan_images_with_preset(
            root_folder=str(tmp_path),
            preset={"id": "builtin-sdxl", "name": "SDXL", "family": "SDXL"},
            gguf_model_name="fake.gguf",
        )
    finally:
        backend_mod._discover_images = original_discover
        backend_mod._build_runtime = original_build_runtime

    assert runtime.generate_calls == 3
    assert summary["processed"] == 1
    assert summary["failed"] == 0
    payload = json.loads(image_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["sdxl_person"] == "person prompt"
    assert payload["sdxl_scene"] == "scene prompt"


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


def test_basic_scan_calls_internal_scan_once_without_warning():
    scanner_mod = _load_module("gpm_vlm_scanner_internal_node")
    node = scanner_mod.GPMVLMScannerInternal()
    calls = {"count": 0}
    original_resolve = scanner_mod._resolve_runtime_preset
    original_run = scanner_mod._run_internal_scan

    try:
        scanner_mod._resolve_runtime_preset = lambda _preset_id, _family: (
            {"id": "builtin-sdxl", "family": "SDXL", "system_prompt": "x"},
            "",
        )

        def _fake_run_internal_scan(**_kwargs):
            calls["count"] += 1
            return '{"ok": true, "processed": 1}', "scan ok"

        scanner_mod._run_internal_scan = _fake_run_internal_scan
        summary_json, status_text = node.scan(
            root_folder=".",
            prompt_preset="builtin-sdxl",
            overwrite_mode="SKIP_EXISTING",
            scan_limit=0,
            write_scan_report="OFF",
            model_name="m.gguf",
            mmproj_name="mmproj.gguf",
            timeout_seconds=30,
            debug_mode="OFF",
        )
        assert calls["count"] == 1
        assert status_text == "scan ok"
        payload = json.loads(summary_json)
        assert payload["ok"] is True
        assert payload["processed"] == 1
    finally:
        scanner_mod._resolve_runtime_preset = original_resolve
        scanner_mod._run_internal_scan = original_run


def test_basic_scan_calls_internal_scan_once_with_warning():
    scanner_mod = _load_module("gpm_vlm_scanner_internal_node")
    node = scanner_mod.GPMVLMScannerInternal()
    calls = {"count": 0}
    original_resolve = scanner_mod._resolve_runtime_preset
    original_run = scanner_mod._run_internal_scan

    try:
        scanner_mod._resolve_runtime_preset = lambda _preset_id, _family: (
            {"id": "builtin-sdxl", "family": "SDXL", "system_prompt": "x"},
            "warning: fallback used",
        )

        def _fake_run_internal_scan(**_kwargs):
            calls["count"] += 1
            return '{"ok": true, "processed": 2}', "scan ok"

        scanner_mod._run_internal_scan = _fake_run_internal_scan
        summary_json, status_text = node.scan(
            root_folder=".",
            prompt_preset="invalid-id",
            overwrite_mode="SKIP_EXISTING",
            scan_limit=0,
            write_scan_report="OFF",
            model_name="m.gguf",
            mmproj_name="mmproj.gguf",
            timeout_seconds=30,
            debug_mode="OFF",
        )
        assert calls["count"] == 1
        assert "scan ok" in status_text
        assert "warning: fallback used" in status_text
        payload = json.loads(summary_json)
        assert payload["ok"] is True
        assert payload["processed"] == 2
    finally:
        scanner_mod._resolve_runtime_preset = original_resolve
        scanner_mod._run_internal_scan = original_run


def test_advanced_scan_call_count_matches_enabled_families():
    scanner_mod = _load_module("gpm_vlm_scanner_internal_node")
    node = scanner_mod.GPMVLMScannerInternalAdvanced()
    calls = {"count": 0, "preset_ids": []}
    original_resolve = scanner_mod._resolve_runtime_preset
    original_run = scanner_mod._run_internal_scan

    def _fake_resolve_runtime_preset(preset_id, _fallback_family):
        return (
            {"id": str(preset_id), "family": "SDXL", "system_prompt": "x"},
            "",
        )

    def _fake_run_internal_scan(**kwargs):
        calls["count"] += 1
        calls["preset_ids"].append(str(kwargs.get("preset_id", "")))
        return '{"ok": true, "processed": 1, "failed": 0, "skipped": 0}', "scan ok"

    try:
        scanner_mod._resolve_runtime_preset = _fake_resolve_runtime_preset
        scanner_mod._run_internal_scan = _fake_run_internal_scan

        summary_json_1, _ = node.scan(
            root_folder=".",
            scan_sdxl="ON",
            scan_pony="OFF",
            scan_natural="OFF",
            sdxl_preset="builtin-sdxl",
            pony_preset="builtin-pony",
            natural_preset="builtin-natural-language",
            model_name="m.gguf",
            mmproj_name="mmproj.gguf",
        )
        assert calls["count"] == 1
        payload_1 = json.loads(summary_json_1)
        assert payload_1.get("requested_family") == "SDXL"

        calls["count"] = 0
        calls["preset_ids"] = []
        summary_json_2, _ = node.scan(
            root_folder=".",
            scan_sdxl="ON",
            scan_pony="ON",
            scan_natural="OFF",
            sdxl_preset="builtin-sdxl",
            pony_preset="builtin-pony",
            natural_preset="builtin-natural-language",
            model_name="m.gguf",
            mmproj_name="mmproj.gguf",
        )
        assert calls["count"] == 2
        payload_2 = json.loads(summary_json_2)
        assert isinstance(payload_2.get("scan_runs"), list)
        assert len(payload_2["scan_runs"]) == 2

        calls["count"] = 0
        calls["preset_ids"] = []
        summary_json_3, _ = node.scan(
            root_folder=".",
            scan_sdxl="ON",
            scan_pony="ON",
            scan_natural="ON",
            sdxl_preset="builtin-sdxl",
            pony_preset="builtin-pony",
            natural_preset="builtin-natural-language",
            model_name="m.gguf",
            mmproj_name="mmproj.gguf",
        )
        assert calls["count"] == 3
        payload_3 = json.loads(summary_json_3)
        assert isinstance(payload_3.get("scan_runs"), list)
        assert len(payload_3["scan_runs"]) == 3
    finally:
        scanner_mod._resolve_runtime_preset = original_resolve
        scanner_mod._run_internal_scan = original_run


def test_advanced_warning_does_not_duplicate_scan_and_merges_status():
    scanner_mod = _load_module("gpm_vlm_scanner_internal_node")
    node = scanner_mod.GPMVLMScannerInternalAdvanced()
    calls = {"count": 0}
    original_resolve = scanner_mod._resolve_runtime_preset
    original_run = scanner_mod._run_internal_scan

    def _fake_resolve_runtime_preset(preset_id, fallback_family):
        warning = "warning: fallback used" if fallback_family == "pony" else ""
        return (
            {"id": str(preset_id), "family": "Pony", "system_prompt": "x"},
            warning,
        )

    def _fake_run_internal_scan(**_kwargs):
        calls["count"] += 1
        return '{"ok": true, "processed": 1, "failed": 0, "skipped": 0}', "scan ok"

    try:
        scanner_mod._resolve_runtime_preset = _fake_resolve_runtime_preset
        scanner_mod._run_internal_scan = _fake_run_internal_scan
        summary_json, status_text = node.scan(
            root_folder=".",
            scan_sdxl="OFF",
            scan_pony="ON",
            scan_natural="OFF",
            sdxl_preset="builtin-sdxl",
            pony_preset="invalid-pony",
            natural_preset="builtin-natural-language",
            model_name="m.gguf",
            mmproj_name="mmproj.gguf",
        )
        assert calls["count"] == 1
        assert "warning: fallback used" in status_text
        payload = json.loads(summary_json)
        assert payload["ok"] is True
    finally:
        scanner_mod._resolve_runtime_preset = original_resolve
        scanner_mod._run_internal_scan = original_run


def test_basic_malformed_user_preset_fallback_does_not_duplicate_scan(tmp_path, monkeypatch):
    scanner_mod = _load_module("gpm_vlm_scanner_internal_node")
    node = scanner_mod.GPMVLMScannerInternal()
    calls = {"count": 0}
    (tmp_path / "vlm_prompt_presets.json").write_text("{bad json", encoding="utf-8")
    monkeypatch.setenv("GPM_USER_DATA_DIR", str(tmp_path))

    def _fake_run_internal_scan(**_kwargs):
        calls["count"] += 1
        return '{"ok": true, "processed": 1}', "scan ok"

    original_run = scanner_mod._run_internal_scan
    try:
        scanner_mod._run_internal_scan = _fake_run_internal_scan
        summary_json, status_text = node.scan(
            root_folder=".",
            prompt_preset="builtin-sdxl",
            overwrite_mode="SKIP_EXISTING",
            scan_limit=0,
            write_scan_report="OFF",
            model_name="m.gguf",
            mmproj_name="mmproj.gguf",
            timeout_seconds=30,
            debug_mode="OFF",
        )
        assert calls["count"] == 1
        assert "warning:" in status_text.lower()
        payload = json.loads(summary_json)
        assert payload["ok"] is True
    finally:
        scanner_mod._run_internal_scan = original_run


def test_advanced_malformed_user_preset_fallback_runs_once_per_enabled_family(tmp_path, monkeypatch):
    scanner_mod = _load_module("gpm_vlm_scanner_internal_node")
    node = scanner_mod.GPMVLMScannerInternalAdvanced()
    calls = {"count": 0}
    (tmp_path / "vlm_prompt_presets.json").write_text("{bad json", encoding="utf-8")
    monkeypatch.setenv("GPM_USER_DATA_DIR", str(tmp_path))

    def _fake_run_internal_scan(**_kwargs):
        calls["count"] += 1
        return '{"ok": true, "processed": 1, "failed": 0, "skipped": 0}', "scan ok"

    original_run = scanner_mod._run_internal_scan
    try:
        scanner_mod._run_internal_scan = _fake_run_internal_scan
        summary_json, status_text = node.scan(
            root_folder=".",
            scan_sdxl="ON",
            scan_pony="ON",
            scan_natural="OFF",
            sdxl_preset="builtin-sdxl",
            pony_preset="builtin-pony",
            natural_preset="builtin-natural-language",
            model_name="m.gguf",
            mmproj_name="mmproj.gguf",
        )
        assert calls["count"] == 2
        assert "warning:" in status_text.lower()
        payload = json.loads(summary_json)
        assert payload["ok"] is True
        assert len(payload["scan_runs"]) == 2
    finally:
        scanner_mod._run_internal_scan = original_run
