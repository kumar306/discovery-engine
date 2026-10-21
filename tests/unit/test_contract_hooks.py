"""The generator hook that keeps schemathesis to strings the api accepts."""

from tests.support.schemathesis_hooks import without_blank_strings


def test_a_string_of_only_whitespace_becomes_a_letter_and_other_strings_stay() -> None:
    assert without_blank_strings(" ") == "a"
    assert without_blank_strings("\x0b") == "a"
    assert without_blank_strings("\u00a0\u2003") == "a"
    assert without_blank_strings(" x ") == " x "
    assert without_blank_strings("") == ""


def test_it_reaches_into_objects_and_lists_and_leaves_other_values_alone() -> None:
    body = {"title": " ", "bullets": ["ok", "\t"], "filters": {"brand": "\x0b"}, "k": 5, "on": None}
    assert without_blank_strings(body) == {
        "title": "a",
        "bullets": ["ok", "a"],
        "filters": {"brand": "a"},
        "k": 5,
        "on": None,
    }
