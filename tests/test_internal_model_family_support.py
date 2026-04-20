import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from gpm_vlm_internal_multimodal import (  # noqa: E402
    SUPPORT_STATUS_EXPERIMENTAL,
    SUPPORT_STATUS_SUPPORTED,
    SUPPORT_STATUS_UNSUPPORTED,
    assess_internal_model_family_support,
)


def test_qwen25_vl_is_approved_for_internal_scanning():
    decision = assess_internal_model_family_support(
        model_path="Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
        mmproj_path="Qwen2.5-VL-7B-mmproj.gguf",
        inferred_runtime_family="qwen_vl",
    )
    assert decision.detected_model_family == "qwen2_5_vl"
    assert decision.family_support_status == SUPPORT_STATUS_SUPPORTED
    assert decision.scan_approved is True


def test_gliese_qwen35_is_blocked_as_experimental():
    decision = assess_internal_model_family_support(
        model_path="Gliese-Qwen3.5-9B-Abliterated-Caption-Q4_K_M.gguf",
        mmproj_path="Gliese-Qwen3.5-9B-Abliterated-Caption-mmproj.gguf",
        inferred_runtime_family="qwen_vl",
    )
    assert decision.detected_model_family == "gliese_qwen3_5_vl"
    assert decision.family_support_status == SUPPORT_STATUS_EXPERIMENTAL
    assert decision.scan_approved is False
    assert "Qwen2.5-VL only" in decision.support_reason


def test_qwen3_vl_is_blocked_as_experimental():
    decision = assess_internal_model_family_support(
        model_path="Qwen3-VL-8B-Instruct-Q4_K_M.gguf",
        mmproj_path="Qwen3-VL-8B-Instruct-mmproj.gguf",
        inferred_runtime_family="qwen_vl",
    )
    assert decision.detected_model_family == "qwen3_vl"
    assert decision.family_support_status == SUPPORT_STATUS_EXPERIMENTAL
    assert decision.scan_approved is False


def test_llava_is_currently_blocked_for_internal_scanner_correctness():
    decision = assess_internal_model_family_support(
        model_path="llava-v1.6-7b-q4_k_m.gguf",
        mmproj_path="llava-v1.6-7b-mmproj.gguf",
        inferred_runtime_family="llava",
    )
    assert decision.detected_model_family == "llava"
    assert decision.family_support_status == SUPPORT_STATUS_UNSUPPORTED
    assert decision.scan_approved is False
