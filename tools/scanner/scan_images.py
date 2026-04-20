#!/usr/bin/env python3
"""
Qwen 3.5 VLM Image Scanner -> Dual SDXL Prompts

Drop-in folder scanner for private prototyping:
- Runs from the folder where this script is located.
- Scans that folder recursively for images.
- Writes sibling JSON beside each image.
- Retries failures up to 3 times.
- Moves hard failures to ./failed preserving relative structure.
"""

import base64
import io
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests
from PIL import Image, UnidentifiedImageError


# -----------------------------
# Defaults
# -----------------------------

DEFAULT_API_URL = "http://localhost:1234/v1/chat/completions"
DEFAULT_MODEL = "gliese-qwen3.5-9b-abliterated-caption"
DEFAULT_MAX_SIDE = 2048
DEFAULT_JPEG_QUALITY = 90
DEFAULT_TIMEOUT = 180
DEFAULT_RETRIES = 3
DEFAULT_PAUSE = 0.75

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Terms that tend to make SDXL outputs worse or too literal for this use case.
FORBIDDEN_PERSON_PHRASES = [
    "handbag", "purse", "basket", "umbrella", "shopping bag",
    "phone", "cell phone", "smartphone",
    "coffee cup", "drink cup", "cup", "mug",
    "book", "notebook", "letter", "paper", "newspaper",
    "flowers in hand", "bouquet", "holding flowers",
    "holding umbrella", "holding phone", "holding a phone",
    "holding handbag", "holding purse", "holding basket",
    "holding shopping bag", "holding coffee", "holding cup",
]

SCENE_SUBJECT_TERMS = [
    "man", "woman", "boy", "girl",
    "male", "female", "person", "people",
    "character", "subject", "main subject",
]

# Exact age-ish phrases to normalize away.
AGE_PHRASES = [
    "young adult", "adult", "teenager", "teen", "child", "kid",
    "middle aged", "middle-aged", "elderly", "older", "old",
    "senior", "mature", "aged", "toddler", "preteen", "pre-teen",
]


# -----------------------------
# Prompt Template
# -----------------------------

SYSTEM_PROMPT = """You are a prompt extraction assistant for SDXL image generation.

Your job is to analyze an image and return STRICT JSON with exactly these keys:
{
  "sdxl_person": "...",
  "sdxl_scene": "..."
}

**CRITICAL RULES FOR EXPLICITNESS:**
1. AVOID VAGUE TERMS: NEVER use the generic summary term "nude body". You MUST specifically list every visible body part.
   - If breasts are visible, you MUST state "breasts" and describe the nipples (e.g., "erect nipples", "pale nipples").
   - If the pelvic region is visible, you MUST use explicit terms like "pussy", "anus", "ass", "labia", or "clitoris" depending on visibility.
2. FLUIDS: If bodily fluids are present in the image, describe them explicitly using terms like "cum", "saliva", "urine", "sweat", or "pussy juices".
3. VERBOSITY: Be extremely verbose. It is better to be too detailed and too long than to miss a detail. Describe every texture, color, and position visible.

Rules:
1. Return JSON only. No markdown. No code fences. No commentary.
2. Both prompts must be concise, comma-separated, generation-friendly SDXL prompts.
3. Do NOT use full sentences.
4. Do NOT include quality tags like masterpiece, best quality, etc.
5. Do NOT include negative prompts.
6. Keep each prompt focused and reusable.

sdxl_scene rules:
- Describe only the environment / location / atmosphere / lighting / composition / background.
- Exclude person identity, clothing, hairstyle, facial traits, and body details.
- Exclude handbags, baskets, umbrellas, phones, cups, books, and other handheld props unless they are a major part of the scene itself.
- Make it reusable with different people.

sdxl_person rules:
- Describe only the main person.
- Start with one of: man, woman, boy, girl
- Do NOT use exact age wording such as child, teenager, young adult, middle aged, elderly.
- Include hair, visible facial traits, clothing, footwear, body presentation if useful, pose, and expression.
- Exclude environment/background/location/weather/camera framing.
- Exclude minor handheld props and clutter such as handbags, baskets, umbrellas, phones, cups, books, flowers in hand, shopping bags.
- Keep it SDXL-friendly and modular for remixing with other scenes.

If the image has no clear scene, make sdxl_scene minimal.
If the image has no clear person, set sdxl_person to an empty string.
If the image has no clear background, set sdxl_scene to an empty string.
"""


# -----------------------------
# Helpers
# -----------------------------

def iter_images(root: Path, script_path: Path):
    script_resolved = script_path.resolve()
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.resolve() == script_resolved:
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = p.relative_to(root)
        if rel.parts and rel.parts[0].lower() == "failed":
            continue
        yield p


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def collapse_commas(text: str) -> str:
    parts = [x.strip() for x in text.split(",")]
    parts = [x for x in parts if x]
    deduped = []
    seen = set()
    for part in parts:
        key = part.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(part)
    return ", ".join(deduped)


def normalize_spacing(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" ,")
    return collapse_commas(text)


def remove_phrase_tokens(text: str, phrases) -> str:
    result = text
    for phrase in sorted(phrases, key=len, reverse=True):
        pattern = re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", re.IGNORECASE)
        result = pattern.sub("", result)
    result = re.sub(r",\s*,+", ", ", result)
    result = re.sub(r"\s+,", ",", result)
    return normalize_spacing(result)


def normalize_subject_label(text: str) -> str:
    """
    Forces the prompt to begin with one of: man / woman / boy / girl
    based on whatever the model gave us.
    """
    low = text.lower()

    # Priority: explicit allowed labels already present
    for label in ("woman", "man", "girl", "boy"):
        if re.search(rf"(?<!\w){label}(?!\w)", low):
            stripped = re.sub(rf"(?<!\w){label}(?!\w)", "", text, count=1, flags=re.IGNORECASE)
            stripped = normalize_spacing(stripped)
            return f"{label}, {stripped}" if stripped else label

    # Heuristics if VLM leaks generic labels
    female_markers = ["female", "lady", "womanly", "feminine"]
    male_markers = ["male", "gentleman", "masculine"]
    young_markers = ["boyish", "girlish"]

    if any(m in low for m in female_markers):
        text = remove_phrase_tokens(text, female_markers + AGE_PHRASES + ["person", "character", "subject"])
        return f"woman, {text}" if text else "woman"

    if any(m in low for m in male_markers):
        text = remove_phrase_tokens(text, male_markers + AGE_PHRASES + ["person", "character", "subject"])
        return f"man, {text}" if text else "man"

    # If youth markers exist, prefer girl/boy only if clearly gendered.
    if "female" in low and any(m in low for m in young_markers):
        text = remove_phrase_tokens(text, female_markers + young_markers + AGE_PHRASES + ["person", "character", "subject"])
        return f"girl, {text}" if text else "girl"

    if "male" in low and any(m in low for m in young_markers):
        text = remove_phrase_tokens(text, male_markers + young_markers + AGE_PHRASES + ["person", "character", "subject"])
        return f"boy, {text}" if text else "boy"

    # Fallback if the model was vague
    text = remove_phrase_tokens(text, AGE_PHRASES + ["person", "character", "subject", "main subject"])
    return f"woman, {text}" if text else "woman"


def sanitize_person_prompt(text: str) -> str:
    text = text.strip()
    if not text:
        return ""

    # Remove age-like phrases before normalization
    text = remove_phrase_tokens(text, AGE_PHRASES)

    # Remove clutter / problem props
    text = remove_phrase_tokens(text, FORBIDDEN_PERSON_PHRASES)

    # Generic cleanup
    text = remove_phrase_tokens(text, ["background", "indoors", "outdoors", "city", "street", "forest", "beach", "room"])
    text = normalize_subject_label(text)
    text = normalize_spacing(text)

    # Final hard cleanup just in case
    text = remove_phrase_tokens(text, AGE_PHRASES + FORBIDDEN_PERSON_PHRASES)
    return normalize_spacing(text)


def sanitize_scene_prompt(text: str) -> str:
    text = text.strip()
    if not text:
        return ""

    text = remove_phrase_tokens(text, SCENE_SUBJECT_TERMS)
    text = remove_phrase_tokens(text, FORBIDDEN_PERSON_PHRASES)
    text = normalize_spacing(text)

    # Extra cleanup for common leaked person descriptors
    leaks = [
        "long hair", "short hair", "blonde hair", "black hair", "brown hair", "silver hair",
        "dress", "jacket", "boots", "shirt", "skirt", "pants", "smile", "smiling",
        "standing pose", "standing", "sitting", "confident expression", "serious expression",
    ]
    text = remove_phrase_tokens(text, leaks)
    return normalize_spacing(text)


def inspect_image(image_path: Path) -> Tuple[str, str, Tuple[int, int], int]:
    with Image.open(image_path) as im:
        fmt = im.format or "UNKNOWN"
        mode = im.mode
        size = im.size
    file_bytes = image_path.stat().st_size
    return fmt, mode, size, file_bytes


def image_to_data_url(image_path: Path, max_side: int, jpeg_quality: int) -> str:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        longest = max(w, h)
        if longest > max_side:
            scale = max_side / float(longest)
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            im = im.resize(new_size, Image.LANCZOS)

        buffer = io.BytesIO()
        im.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None

    # Direct parse first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Fallback: find first {...} block
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


def call_vlm(
    image_path: Path,
    api_url: str,
    model: str,
    max_side: int,
    jpeg_quality: int,
    timeout: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        data_url = image_to_data_url(image_path, max_side=max_side, jpeg_quality=jpeg_quality)
    except UnidentifiedImageError as e:
        return None, f"Image decode error: {e}"
    except Exception as e:
        return None, f"Image prep error: {e}"

    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this image and return strict JSON only."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    }

    try:
        response = requests.post(api_url, json=payload, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        body = ""
        try:
            body = response.text[:2000]
        except Exception:
            pass
        return None, f"API Error: {e}\nResponse body: {body}"
    except Exception as e:
        return None, f"Request failed: {e}"

    try:
        data = response.json()
    except Exception as e:
        return None, f"Failed to decode API JSON response: {e}"

    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        return None, f"Unexpected API response format: {json.dumps(data)[:2000]}"

    if isinstance(content, list):
        # Some servers return segmented content
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        content = "\n".join(text_parts).strip()

    parsed = extract_json_from_text(content if isinstance(content, str) else str(content))
    if not parsed:
        return None, f"Model did not return valid JSON.\nRaw content: {str(content)[:2000]}"

    return parsed, None


def write_json_output(image_path: Path, person_prompt: str, scene_prompt: str) -> None:
    payload = {
        "sdxl_person": person_prompt,
        "sdxl_scene": scene_prompt,
    }
    image_path.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def move_to_failed(image_path: Path, input_root: Path, failed_root: Path) -> Path:
    rel = image_path.relative_to(input_root)
    dest = failed_root / rel
    ensure_dir(dest.parent)
    shutil.move(str(image_path), str(dest))
    return dest


def already_done(image_path: Path) -> bool:
    return image_path.with_suffix(".json").exists()


def main() -> int:
    script_path = Path(__file__).resolve()
    input_root = script_path.parent
    failed_root = input_root / "failed"
    ensure_dir(failed_root)

    images = sorted(iter_images(input_root, script_path))
    total = len(images)

    if total == 0:
        print(f"Scan root: {input_root}")
        print("No images found.")
        return 0

    print(f"Scan root: {input_root}")
    print(f"Found {total} image(s).")
    print(f"Failed  : {failed_root}")
    print(f"API     : {DEFAULT_API_URL}")
    print(f"Model   : {DEFAULT_MODEL}")
    print("-" * 72)

    done_count = 0
    fail_count = 0
    skipped_count = 0

    for idx, image_path in enumerate(images, start=1):
        remaining = total - idx
        rel = image_path.relative_to(input_root)

        if already_done(image_path):
            skipped_count += 1
            print(f"[{idx}/{total} | {remaining} left] Skipping existing JSON: {rel}")
            continue

        print(f"[{idx}/{total} | {remaining} left] Processing: {rel}")

        # Debug info helps identify the weird files that 400 out.
        try:
            fmt, mode, size, file_bytes = inspect_image(image_path)
            print(f"    Image info: format={fmt}, mode={mode}, size={size}, bytes={file_bytes}")
        except Exception as e:
            print(f"    Could not inspect image: {e}")

        success = False
        last_error = None

        for attempt in range(1, DEFAULT_RETRIES + 1):
            print(f"    Attempt {attempt}/{DEFAULT_RETRIES}...", end=" ")
            result, error = call_vlm(
                image_path=image_path,
                api_url=DEFAULT_API_URL,
                model=DEFAULT_MODEL,
                max_side=DEFAULT_MAX_SIDE,
                jpeg_quality=DEFAULT_JPEG_QUALITY,
                timeout=DEFAULT_TIMEOUT,
            )

            if error:
                last_error = error
                print("Failed.")
                print("    " + error.replace("\n", "\n    "))
                continue

            scene_prompt = sanitize_scene_prompt(str(result.get("sdxl_scene", "")).strip())
            person_prompt = sanitize_person_prompt(str(result.get("sdxl_person", "")).strip())
            write_json_output(image_path, person_prompt, scene_prompt)

            print("OK")
            if scene_prompt:
                print(f"    Scene  : {scene_prompt}")
            else:
                print("    Scene  : <empty>")
            if person_prompt:
                print(f"    Person : {person_prompt}")
            else:
                print("    Person : <empty>")

            done_count += 1
            success = True
            break

        if not success:
            fail_count += 1
            dest = move_to_failed(image_path, input_root, failed_root)
            print(f"    Moved to failed: {dest}")
            if last_error:
                log_path = failed_root / "_fail_log.txt"
                with log_path.open("a", encoding="utf-8") as f:
                    f.write("=" * 80 + "\n")
                    f.write(f"Image: {rel}\n")
                    f.write(last_error + "\n\n")

        if DEFAULT_PAUSE > 0:
            time.sleep(DEFAULT_PAUSE)

    print("-" * 72)
    print(f"Done. Success: {done_count} | Failed: {fail_count} | Skipped: {skipped_count} | Total: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
