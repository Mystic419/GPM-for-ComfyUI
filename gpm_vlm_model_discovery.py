from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

AUTO_MMPROJ_OPTION = "(auto)"

_NO_MODELS_OPTION = "<no gguf models found>"
_NO_MMPROJ_OPTION = "<no mmproj models found>"


@dataclass(frozen=True)
class GPMDiscoveredGGUF:
    model_choices: list[str]
    mmproj_choices: list[str]
    model_by_choice: dict[str, Path]
    mmproj_by_choice: dict[str, Path]


def _resolve_models_root() -> Path | None:
    try:
        import folder_paths  # type: ignore

        models_dir = getattr(folder_paths, "models_dir", None)
        if models_dir:
            root = Path(str(models_dir)).expanduser()
            if root.exists() and root.is_dir():
                return root
    except Exception:
        pass

    current = Path(__file__).resolve()
    for parent in current.parents:
        if parent.name.lower() == "comfyui":
            fallback = parent / "models"
            if fallback.exists() and fallback.is_dir():
                return fallback
    return None


def _candidate_gguf_dirs() -> list[Path]:
    models_root = _resolve_models_root()
    if models_root is None:
        return []

    raw_dirs = [
        models_root / "llm",
        models_root / "llm" / "GGUF",
        models_root / "GGUF",
    ]
    unique: list[Path] = []
    seen: set[str] = set()
    for path in raw_dirs:
        key = str(path.resolve()).casefold()
        if key in seen:
            continue
        seen.add(key)
        if path.exists() and path.is_dir():
            unique.append(path)
    return unique


def _choice_key(path: Path, models_root: Path | None) -> str:
    if models_root is not None:
        try:
            rel = path.resolve().relative_to(models_root.resolve())
            return rel.as_posix()
        except Exception:
            pass
    return str(path.resolve())


def _discover_gguf_files() -> GPMDiscoveredGGUF:
    models_root = _resolve_models_root()

    model_by_choice: dict[str, Path] = {}
    mmproj_by_choice: dict[str, Path] = {}

    for directory in _candidate_gguf_dirs():
        for path in directory.rglob("*.gguf"):
            if not path.is_file():
                continue
            choice = _choice_key(path, models_root)
            name_cf = path.name.casefold()
            if "mmproj" in name_cf:
                mmproj_by_choice.setdefault(choice, path)
            else:
                model_by_choice.setdefault(choice, path)

    model_choices = sorted(model_by_choice.keys(), key=str.casefold)
    mmproj_choices = [AUTO_MMPROJ_OPTION]
    mmproj_choices.extend(sorted(mmproj_by_choice.keys(), key=str.casefold))

    if not model_choices:
        model_choices = [_NO_MODELS_OPTION]
    if len(mmproj_choices) == 1:
        mmproj_choices.append(_NO_MMPROJ_OPTION)

    return GPMDiscoveredGGUF(
        model_choices=model_choices,
        mmproj_choices=mmproj_choices,
        model_by_choice=model_by_choice,
        mmproj_by_choice=mmproj_by_choice,
    )


def discover_gguf_model_choices() -> list[str]:
    return _discover_gguf_files().model_choices


def discover_mmproj_choices() -> list[str]:
    return _discover_gguf_files().mmproj_choices


def _is_missing_choice(choice: str) -> bool:
    return choice in {_NO_MODELS_OPTION, _NO_MMPROJ_OPTION, ""}


def _normalize_match_name(raw: str) -> str:
    text = str(raw).casefold().replace("mmproj", "")
    text = re.sub(r"\.gguf$", "", text)
    text = re.sub(r"[-_.]?q\d+(_[a-z0-9]+)?$", "", text)
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def _auto_pair_mmproj(model_path: Path, mmproj_paths: list[Path]) -> Path | None:
    model_key = _normalize_match_name(model_path.stem)
    if not model_key:
        return None

    best_score = 0
    best_paths: list[Path] = []
    for mmproj in mmproj_paths:
        mmproj_key = _normalize_match_name(mmproj.stem)
        if not mmproj_key:
            continue
        score = 0
        if mmproj_key == model_key:
            score = 4
        elif mmproj_key in model_key:
            score = 3
        elif model_key in mmproj_key:
            score = 2
        elif any(token and token in mmproj_key for token in re.split(r"[^a-z0-9]+", model_path.stem.casefold())):
            score = 1

        if score > best_score:
            best_score = score
            best_paths = [mmproj]
        elif score == best_score and score > 0:
            best_paths.append(mmproj)

    if best_score <= 0:
        return None
    if len(best_paths) == 1:
        return best_paths[0]
    return None


def resolve_model_and_mmproj_paths(model_name: str, mmproj_name: str) -> tuple[Path | None, Path | None, str]:
    discovered = _discover_gguf_files()
    model_choice = str(model_name).strip()
    mmproj_choice = str(mmproj_name).strip()

    if _is_missing_choice(model_choice):
        return None, None, "no GGUF VLM model was selected or discovered in ComfyUI model folders"

    model_path = discovered.model_by_choice.get(model_choice)
    if model_path is None:
        return None, None, f"selected GGUF model was not found: {model_choice}"

    mmproj_paths = list(discovered.mmproj_by_choice.values())
    if mmproj_choice == AUTO_MMPROJ_OPTION:
        matched = _auto_pair_mmproj(model_path, mmproj_paths)
        if matched is not None:
            return model_path, matched, ""
        if mmproj_paths:
            return None, None, (
                "mmproj auto-pair failed. Select mmproj_name manually from dropdown for this model."
            )
        return None, None, (
            "model requires mmproj, but no mmproj GGUF files were found. Place an mmproj file in ComfyUI/models/llm, "
            "ComfyUI/models/llm/GGUF, or ComfyUI/models/GGUF."
        )

    if _is_missing_choice(mmproj_choice):
        return None, None, "model requires mmproj, but no mmproj GGUF model was selected or discovered"

    mmproj_path = discovered.mmproj_by_choice.get(mmproj_choice)
    if mmproj_path is None:
        return None, None, f"selected mmproj model was not found: {mmproj_choice}"

    return model_path, mmproj_path, ""
