import os
import sys
import types
import importlib.util

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
PACKAGE_NAME = "gpm_testpkg_runtime_api"
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


gpm_vlm_runtime_api = _load_module("gpm_vlm_runtime_api")
clean_split_prompts = gpm_vlm_runtime_api.clean_split_prompts


def test_clean_split_prompts_removes_scene_leak_from_person_for_sdxl():
    person_prompt = "blonde woman, curly hair, outdoor setting, archway background"
    scene_prompt = "stone archway, courtyard, soft sunlight"

    cleaned_person, cleaned_scene = clean_split_prompts(person_prompt, scene_prompt, "SDXL")

    assert cleaned_person == "blonde woman, curly hair"
    assert cleaned_scene == "stone archway, courtyard, soft sunlight"


def test_clean_split_prompts_keeps_valid_scene_architecture_fragments():
    person_prompt = "woman, smiling"
    scene_prompt = "gothic architecture, stone arches, chandelier, courtyard wall"

    cleaned_person, cleaned_scene = clean_split_prompts(person_prompt, scene_prompt, "SDXL")

    assert cleaned_person == "woman, smiling"
    assert cleaned_scene == "gothic architecture, stone arches, chandelier, courtyard wall"


def test_clean_split_prompts_preserves_empty_person_prompt():
    cleaned_person, cleaned_scene = clean_split_prompts("", "city street, evening lighting", "SDXL")

    assert cleaned_person == ""
    assert cleaned_scene == "city street, evening lighting"
