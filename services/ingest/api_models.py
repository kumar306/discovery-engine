from datetime import datetime

from pydantic import BaseModel

# these models are never used to build an answer, the routes build their own bodies
# they exist so that /openapi.json says exactly what the api returns, and a contract test then
# checks that the real answers have these shapes


class AcceptedBody(BaseModel):
    request_id: str
    doc_id: str
    version: int
    accepted_at: datetime


class DocumentStatusBody(BaseModel):
    request_id: str
    doc_id: str
    # newest version the api accepted
    latest_version: int
    # newest version the worker committed, null until the worker has handled one
    current_version: int | None
    # newest version live in qdrant, null until one is searchable
    indexed_version: int | None
    # true when the newest version the worker committed is a delete
    deleted: bool


class ErrorBody(BaseModel):
    error: str
    message: str
    request_id: str
    # a 409 says which version is held now, so the client can decide what to do
    current_version: int | None = None


class FieldError(BaseModel):
    field: str
    message: str
    type: str


class ValidationFailedBody(ErrorBody):
    errors: list[FieldError]
