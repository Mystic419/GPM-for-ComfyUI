import os
import sys


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


from gpm_vlm_model_discovery import _model_choice_sort_key  # noqa: E402


def test_verified_qwen25_vl_model_sorts_before_experimental_qwen35_models():
    choices = [
        "LLM/gguf/Gliese-Qwen3.5-9B/model.Q4_K_S.gguf",
        "LLM/gguf/Qwen2.5-VL-7B/model.Q4_K_M.gguf",
        "LLM/gguf/qwen3.5-9b/model.Q5_K_S.gguf",
    ]
    assert sorted(choices, key=_model_choice_sort_key)[0] == choices[1]
