from gallery_prompt_manager.core.prompt_combiner_core import combine_prompt_parts


def test_combine_prompt_parts_all_inputs():
    combined = combine_prompt_parts(
        "woman, long black hair",
        "city street, neon lights",
        "<lora:test:1>",
    )

    assert combined == "woman, long black hair, city street, neon lights, <lora:test:1>"


def test_combine_prompt_parts_ignores_empty_inputs():
    combined = combine_prompt_parts("", "forest, fog", "")

    assert combined == "forest, fog"


def test_combine_prompt_parts_cleans_whitespace_and_commas():
    combined = combine_prompt_parts(
        "  woman,,   long black hair  ",
        "  city street ,  neon lights  ",
        "  <lora:test:1>  ",
    )

    assert combined == "woman, long black hair, city street, neon lights, <lora:test:1>"
