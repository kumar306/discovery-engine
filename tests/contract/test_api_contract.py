"""Schemathesis reads /openapi.json, invents requests from it, and checks the real answers.

If the document promises a 422 with a list of field errors, or a body shape, or that nothing
answers 500, this is where a difference between the promise and the code is found.
"""

from typing import Any

import schemathesis
from hypothesis import HealthCheck, settings

from services.ingest.app import create_app
from services.ingest.settings import IngestSettings
from tests.unit.test_api_edges import RecordingIngest

app = create_app(RecordingIngest(), IngestSettings())
schema = schemathesis.openapi.from_asgi("/openapi.json", app)


@schemathesis.hook
def map_body(context: schemathesis.HookContext, body: Any) -> Any:
    # json schema calls 8388608.0 an integer, but the version must be a whole number written as
    # one, 1.0 is refused on purpose and a test in test_schemas pins that, so the generator is
    # kept to whole numbers instead of loosening the api for it
    if isinstance(body, dict) and isinstance(body.get("version"), float):
        body = {**body, "version": int(body["version"])}
    return body


@schema.parametrize()
@settings(max_examples=60, deadline=None, suppress_health_check=list(HealthCheck))
def test_every_answer_matches_what_the_document_promises(case: schemathesis.Case) -> None:
    case.call_and_validate()
