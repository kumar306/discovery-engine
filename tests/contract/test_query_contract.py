"""Schemathesis reads /openapi.json of the query service, invents requests and checks the answers.

If the document promises a 422 with a list of field errors, a 413, a 503, or a body shape for
either flow, this is where a difference between the promise and the code is found.
"""

from typing import Any

import schemathesis
from hypothesis import HealthCheck, settings

from tests.support.schemathesis_hooks import without_blank_strings
from tests.unit.test_query_api import Rig, hit

rig = Rig()
rig.search.grouped = {"dense": [hit("a"), hit("b")], "bm25": [hit("b"), hit("c")]}
rig.search.chunks = {"dense": [hit("a", 0), hit("a", 1)], "bm25": [hit("c", 0)]}
schema = schemathesis.openapi.from_asgi("/openapi.json", rig.app)


@schemathesis.hook
def map_body(context: schemathesis.HookContext, body: Any) -> Any:
    # json schema calls 5.0 an integer, but k and the chunk limit must be whole numbers written as
    # one, 5.0 is refused on purpose, so the generator is kept to whole numbers instead of
    # loosening the api for it
    if isinstance(body, dict):
        for name in ("k", "per_retriever_limit"):
            if isinstance(body.get(name), float):
                body = {**body, name: int(body[name])}
    return without_blank_strings(body)


@schema.parametrize()
@settings(max_examples=40, deadline=None, suppress_health_check=list(HealthCheck))
def test_every_answer_matches_what_the_document_promises(case: schemathesis.Case) -> None:
    case.call_and_validate()
