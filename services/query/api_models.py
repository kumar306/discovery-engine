from pydantic import BaseModel

# the error bodies of the query service, the same shape as the ingest api's so a client reads both
# the routes build their own error bodies, these models make /openapi.json say what they look like


class ErrorBody(BaseModel):
    error: str
    message: str
    request_id: str


class FieldError(BaseModel):
    field: str
    message: str
    type: str


class ValidationFailedBody(ErrorBody):
    errors: list[FieldError]
