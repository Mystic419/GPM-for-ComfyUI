import importlib.util
import os
import sys
import types


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
PACKAGE_NAME = "gpm_testpkg_worker_bootstrap"
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


def test_worker_script_bootstrap_avoids_repo_package_init():
    worker_path = os.path.join(ROOT_DIR, "gpm_vlm_internal_worker.py")
    spec = importlib.util.spec_from_file_location("__main__", worker_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to build worker spec")
    module = importlib.util.module_from_spec(spec)
    original_argv = sys.argv
    sys.argv = [worker_path, "--help"]
    try:
        try:
            spec.loader.exec_module(module)
        except SystemExit:
            pass
    finally:
        sys.argv = original_argv
    assert "GPM" not in sys.modules


def test_worker_scan_invalid_root_returns_summary_not_import_error():
    worker_mod = _load_module("gpm_vlm_internal_worker")
    summary = worker_mod._run_worker_scan(
        {
            "root_folder": "Z:/this/path/does/not/exist",
            "preset_id": "builtin-sdxl",
            "overwrite_mode": "SKIP_EXISTING",
            "scan_limit": 0,
            "model_name": "m.gguf",
            "mmproj_name": "mmproj.gguf",
            "timeout_seconds": 10,
            "n_ctx": 4096,
            "n_gpu_layers": -1,
            "temperature": 0.2,
            "top_p": 0.95,
            "max_tokens": 128,
            "threads": 0,
            "batch_size": 512,
            "debug_mode": False,
        }
    )
    assert isinstance(summary, dict)
    assert summary.get("ok") is False
    assert "server" not in str(summary.get("error", "")).lower()


def test_worker_prefers_absolute_paths_when_present():
    worker_mod = _load_module("gpm_vlm_internal_worker")

    class _Store:
        def get_preset(self, _preset_id):
            return {"id": "builtin-sdxl", "temperature": 0.2, "top_p": 0.95, "max_tokens": 512}

    worker_mod.GPMVLMPresetStore = _Store
    worker_mod.get_preset_generation_settings = lambda _preset: (0.2, 0.95, 512)
    captured = {}

    def _fake_scan_images_with_preset(**kwargs):
        captured.update(kwargs)
        return {"ok": True, "processed": 1}

    worker_mod.scan_images_with_preset = _fake_scan_images_with_preset
    this_file = os.path.realpath(__file__)
    summary = worker_mod._run_worker_scan(
        {
            "root_folder": ".",
            "preset_id": "builtin-sdxl",
            "overwrite_mode": "SKIP_EXISTING",
            "scan_limit": 0,
            "model_name": "dropdown-model.gguf",
            "mmproj_name": "dropdown-mmproj.gguf",
            "model_path_resolved": this_file,
            "mmproj_path_resolved": this_file,
            "timeout_seconds": 10,
            "n_ctx": 4096,
            "n_gpu_layers": -1,
            "temperature": 0.2,
            "top_p": 0.95,
            "max_tokens": 128,
            "threads": 0,
            "batch_size": 512,
            "debug_mode": False,
        }
    )
    assert summary.get("ok") is True
    assert captured["internal_model_path_override"] == this_file
    assert captured["internal_mmproj_path_override"] == this_file


def test_worker_returns_clear_error_for_missing_resolved_model_path():
    worker_mod = _load_module("gpm_vlm_internal_worker")

    class _Store:
        def get_preset(self, _preset_id):
            return {"id": "builtin-sdxl", "temperature": 0.2, "top_p": 0.95, "max_tokens": 512}

    worker_mod.GPMVLMPresetStore = _Store
    missing = os.path.join(os.path.dirname(os.path.realpath(__file__)), "__missing_model.gguf")
    summary = worker_mod._run_worker_scan(
        {
            "root_folder": ".",
            "preset_id": "builtin-sdxl",
            "overwrite_mode": "SKIP_EXISTING",
            "scan_limit": 0,
            "model_name": "dropdown-model.gguf",
            "mmproj_name": "dropdown-mmproj.gguf",
            "model_path_resolved": missing,
            "mmproj_path_resolved": os.path.realpath(__file__),
            "timeout_seconds": 10,
            "n_ctx": 4096,
            "n_gpu_layers": -1,
            "temperature": 0.2,
            "top_p": 0.95,
            "max_tokens": 128,
            "threads": 0,
            "batch_size": 512,
            "debug_mode": False,
        }
    )
    assert summary.get("ok") is False
    assert "worker resolved model path was not found:" in str(summary.get("error", ""))
