import importlib.util
import os
import sys
import types


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
PACKAGE_NAME = "gpm_testpkg_free_vram"
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


def test_clear_cached_runtime_resets_cached_fields():
    runtime_internal = _load_module("gpm_vlm_runtime_internal")
    runtime_cls = runtime_internal.GPMGGUFInternalRuntime

    runtime_cls._cached_llm = object()
    runtime_cls._cached_chat_handler = object()
    runtime_cls._cached_signature = ("sig",)
    info = runtime_cls.clear_cached_runtime(reason="test clear")

    assert info["cleared_cached_runtime"] is True
    assert info["had_cached_runtime"] is True
    assert info["had_cached_handler"] is True
    assert runtime_cls._cached_llm is None
    assert runtime_cls._cached_chat_handler is None
    assert runtime_cls._cached_signature is None


def test_release_instance_and_cache_releases_llm_and_chat_handler():
    runtime_internal = _load_module("gpm_vlm_runtime_internal")
    runtime_cls = runtime_internal.GPMGGUFInternalRuntime

    class _Closable:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    class _Runtime:
        pass

    active_llm = _Closable()
    active_handler = _Closable()
    cached_llm = _Closable()
    cached_handler = _Closable()
    runtime = _Runtime()
    runtime._llm = active_llm
    runtime._chat_handler = active_handler

    runtime_cls._cached_llm = cached_llm
    runtime_cls._cached_chat_handler = cached_handler
    runtime_cls._cached_signature = ("sig",)

    info = runtime_cls.release_instance_and_cache(runtime, reason="test release")

    assert runtime._llm is None
    assert runtime._chat_handler is None
    assert runtime_cls._cached_llm is None
    assert runtime_cls._cached_chat_handler is None
    assert runtime_cls._cached_signature is None

    assert active_llm.close_calls == 1
    assert active_handler.close_calls == 1
    assert cached_llm.close_calls == 1
    assert cached_handler.close_calls == 1

    assert info["requested"] is True
    assert info["reason"] == "test release"
    assert info["active_llm_found"] is True
    assert info["active_llm_close_attempted"] is True
    assert info["active_handler_found"] is True
    assert info["active_handler_close_attempted"] is True
    assert info["cached_llm_found"] is True
    assert info["cached_llm_close_attempted"] is True
    assert info["cached_handler_found"] is True
    assert info["cached_handler_close_attempted"] is True
    assert info["memory_cleanup_attempted"] is True


def test_free_vram_node_not_registered_in_package_mappings():
    init_path = os.path.join(ROOT_DIR, "__init__.py")
    with open(init_path, "r", encoding="utf-8") as handle:
        init_text = handle.read()
    assert "GPM VLM Free VRAM" not in init_text
