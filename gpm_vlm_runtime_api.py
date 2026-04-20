from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from PIL import Image, UnidentifiedImageError

from .gpm_vlm_runtime_base import GPMVLMRuntime, RUNTIME_MODE_API


def _build_user_prompt(preset: dict[str, Any]) -> str:
    family = str(preset.get("family", "SDXL"))
    ban_list = preset.get("ban_list", [])
    ban_text = ""
    if isinstance(ban_list, list) and ban_list:
        joined = ", ".join([str(item).strip() for item in ban_list if str(item).strip()])
        if joined:
            ban_text = f"Avoid these terms if possible: {joined}."

    return (
        f"Generate prompts for family '{family}'. "
        "Return strict JSON only with keys person_prompt and scene_prompt. "
        "Do not return markdown. Keep output concise and reusable. "
        f"{ban_text}"
    ).strip()


def _image_to_data_url(image_path: Path, max_side: int = 2048, jpeg_quality: int = 90) -> str:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        longest_side = max(width, height)
        if longest_side > max_side:
            scale = max_side / float(longest_side)
            new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            rgb = rgb.resize(new_size, Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match is None:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_model_prompts(payload: dict[str, Any], family: str) -> tuple[str, str]:
    family_to_json_fields: dict[str, tuple[str, str]] = {
        "SDXL": ("sdxl_person", "sdxl_scene"),
        "Pony": ("pony_person", "pony_scene"),
        "Natural Language": ("natural_person", "natural_scene"),
    }
    person_key, scene_key = family_to_json_fields.get(family, ("person_prompt", "scene_prompt"))
    person_candidates = [person_key, "person_prompt", "person", "subject", "prompt_person"]
    scene_candidates = [scene_key, "scene_prompt", "scene", "background", "prompt_scene"]

    person_prompt = ""
    for key in person_candidates:
        value = payload.get(key)
        if isinstance(value, str):
            person_prompt = value.strip()
            if person_prompt:
                break

    scene_prompt = ""
    for key in scene_candidates:
        value = payload.get(key)
        if isinstance(value, str):
            scene_prompt = value.strip()
            if scene_prompt:
                break

    return person_prompt, scene_prompt


def _sanitize_person_prompt_if_environment_only(person_prompt: str) -> str:
    text = str(person_prompt or "").strip()
    if not text:
        return ""

    lower_text = text.casefold()
    person_terms = {
        "man",
        "woman",
        "boy",
        "girl",
        "person",
        "character",
        "male",
        "female",
    }
    if any(re.search(rf"\b{re.escape(term)}\b", lower_text) for term in person_terms):
        return text

    environment_terms = {
        "room",
        "bedroom",
        "street",
        "alley",
        "interior",
        "exterior",
        "building",
        "ceiling",
        "wall",
        "floor",
        "window",
        "table",
        "chair",
    }
    if any(re.search(rf"\b{re.escape(term)}\b", lower_text) for term in environment_terms):
        return ""

    return text


def generate_with_openai_compatible_api(
    api_url: str,
    model_name: str,
    timeout_seconds: int,
    image_path: Path,
    preset: dict[str, Any],
) -> tuple[str, str, str]:
    try:
        image_data_url = _image_to_data_url(image_path)
    except UnidentifiedImageError as exc:
        return "", "", f"image decode error: {exc}"
    except Exception as exc:
        return "", "", f"image encode error: {exc}"

    payload = {
        "model": model_name,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": str(preset.get("system_prompt", "")).strip()},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _build_user_prompt(preset)},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
    }

    encoded_payload = json.dumps(payload).encode("utf-8")
    request_obj = urllib_request.Request(
        str(api_url).strip(),
        data=encoded_payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib_request.urlopen(request_obj, timeout=max(1, int(timeout_seconds))) as response:
            response_bytes = response.read()
    except urllib_error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return "", "", f"backend HTTP error {exc.code}: {body[:600]}"
    except urllib_error.URLError as exc:
        return "", "", f"backend connection error: {exc.reason}"
    except Exception as exc:
        return "", "", f"backend request failed: {exc}"

    try:
        parsed = json.loads(response_bytes.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        return "", "", f"backend response JSON decode failed: {exc}"

    try:
        content = parsed["choices"][0]["message"]["content"]
    except Exception:
        return "", "", "backend response format was not recognized"

    if isinstance(content, list):
        content = "\n".join(
            str(part.get("text", "")).strip()
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()

    output_payload = _extract_json_object(str(content))
    if output_payload is None:
        return "", "", f"backend did not return strict JSON: {str(content)[:600]}"

    family = str(preset.get("family", "SDXL"))
    person_prompt, scene_prompt = _normalize_model_prompts(output_payload, family)
    person_prompt = _sanitize_person_prompt_if_environment_only(person_prompt)
    return person_prompt, scene_prompt, ""


class GPMGGUFAPIRuntime(GPMVLMRuntime):
    runtime_mode = RUNTIME_MODE_API

    def __init__(self, api_url: str, model_name: str, timeout_seconds: int):
        self.api_url = str(api_url).strip() or "http://127.0.0.1:1234/v1/chat/completions"
        self.model_name = str(model_name).strip()
        self.timeout_seconds = max(1, int(timeout_seconds))

    def generate(self, image_path: Path, preset: dict[str, Any]) -> tuple[str, str, str]:
        return generate_with_openai_compatible_api(
            api_url=self.api_url,
            model_name=self.model_name,
            timeout_seconds=self.timeout_seconds,
            image_path=image_path,
            preset=preset,
        )

