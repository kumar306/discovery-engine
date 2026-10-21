"""Shared by the two contract tests: keep generated text to what the apis accept."""

from typing import Any


def without_blank_strings(value: Any) -> Any:
    # the openapi document says a text field needs length 1, the apis strip whitespace first and
    # refuse the empty rest with a 422 (pinned by the whitespace tests of test_api_edges and
    # test_query_api), so a generated "\x0b" is a valid request by the document and an invalid one
    # by the api, the generator is kept to strings with a visible character instead of loosening the
    # api, python's strip is a superset of the whitespace the apis strip, so this never lets a
    # blank string through
    if isinstance(value, str):
        return "a" if value and not value.strip() else value
    if isinstance(value, list):
        return [without_blank_strings(item) for item in value]
    if isinstance(value, dict):
        return {name: without_blank_strings(item) for name, item in value.items()}
    return value
