import os
import sys
import types
import importlib.util

from PIL import Image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
PACKAGE_NAME = "gpm_testpkg_internal_runtime"
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


runtime_internal = _load_module("gpm_vlm_runtime_internal")


def test_compute_downscaled_size_noop_when_under_cap():
    width, height, downscaled = runtime_internal._compute_downscaled_size(800, 600, 1024)
    assert (width, height) == (800, 600)
    assert downscaled is False


def test_compute_downscaled_size_preserves_aspect_when_over_cap():
    width, height, downscaled = runtime_internal._compute_downscaled_size(3000, 1500, 1024)
    assert (width, height) == (1024, 512)
    assert downscaled is True


def test_build_internal_image_data_url_reports_downscale(tmp_path):
    image_path = tmp_path / "large.png"
    Image.new("RGB", (2400, 1600), color=(10, 20, 30)).save(image_path)

    data_url, meta = runtime_internal._build_internal_image_data_url(image_path, max_long_edge=1024)

    assert data_url.startswith("data:image/jpeg;base64,")
    assert meta["source_size"] == (2400, 1600)
    assert meta["inference_size"] == (1024, 682)
    assert meta["downscaled_for_inference"] is True
    assert meta["max_long_edge_cap"] == 1024
