import json

from gallery_prompt_manager.core.gallery_browser_core import (
    enter_folder,
    get_selected_image_path,
    go_back,
    list_folder,
    load_sibling_prompts,
)


def test_list_folder_sorts_folders_then_images(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "b_folder").mkdir()
    (root / "A_folder").mkdir()
    (root / "z.png").write_bytes(b"fake")
    (root / "a.jpg").write_bytes(b"fake")
    (root / "note.txt").write_text("ignore", encoding="utf-8")

    listing = list_folder(str(root), "")

    assert listing.folders == ["A_folder", "b_folder"]
    assert listing.images == ["a.jpg", "z.png"]


def test_navigation_stays_inside_root(tmp_path):
    root = tmp_path / "root"
    child = root / "child"
    root.mkdir()
    child.mkdir()

    assert enter_folder(str(root), "", "child") == "child"
    assert go_back(str(root), "child") == ""
    assert go_back(str(root), "") == ""


def test_select_image_and_prompt_loading(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    image_path = root / "sample.png"
    image_path.write_bytes(b"fake")

    payload = {
        "sdxl_person": "person text",
        "sdxl_scene": "scene text",
    }
    image_path.with_suffix(".json").write_text(json.dumps(payload), encoding="utf-8")

    selected = get_selected_image_path(str(root), "", "sample.png")
    person, scene = load_sibling_prompts(selected)

    assert str(selected).endswith("sample.png")
    assert person == "person text"
    assert scene == "scene text"


def test_invalid_or_missing_json_returns_empty_prompts(tmp_path):
    root = tmp_path / "root"
    root.mkdir()

    image_path = root / "sample.png"
    image_path.write_bytes(b"fake")

    person, scene = load_sibling_prompts(image_path)
    assert person == ""
    assert scene == ""

    image_path.with_suffix(".json").write_text("{not-json", encoding="utf-8")
    person, scene = load_sibling_prompts(image_path)
    assert person == ""
    assert scene == ""
