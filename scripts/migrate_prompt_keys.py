#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

KEY_RENAMES = {
    "person_prompt": "sdxl_person",
    "scene_prompt": "sdxl_scene",
}


def migrate_file(path: Path) -> tuple[bool, bool]:
    """
    Returns:
    - updated: file contents changed and written
    - skipped: no migration needed
    """
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("JSON root must be an object")

    has_old_keys = any(old_key in payload for old_key in KEY_RENAMES)
    if not has_old_keys:
        return False, True

    changed = False
    for old_key, new_key in KEY_RENAMES.items():
        if old_key not in payload:
            continue
        old_value = payload.pop(old_key)
        if new_key not in payload:
            payload[new_key] = old_value
        changed = True

    if not changed:
        return False, True

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return True, False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-time migration for GPM JSON sidecars: person_prompt/scene_prompt -> sdxl_person/sdxl_scene"
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Folder to scan recursively for .json files (default: current folder)",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"Invalid folder: {root}")
        return 1

    scanned = 0
    updated = 0
    skipped = 0
    failed = 0

    for json_path in root.rglob("*.json"):
        scanned += 1
        try:
            file_updated, file_skipped = migrate_file(json_path)
            if file_updated:
                updated += 1
            elif file_skipped:
                skipped += 1
        except Exception as exc:
            failed += 1
            print(f"FAILED: {json_path} ({exc})")

    print(f"Files scanned: {scanned}")
    print(f"Files updated: {updated}")
    print(f"Files skipped: {skipped}")
    print(f"Files failed : {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
