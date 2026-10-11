from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter

MAX_VERSION = 2**63 - 1

DocId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
Version = Annotated[int, Field(ge=0, le=MAX_VERSION)]
Locale = Literal["us", "es", "jp"]
Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


def text(max_length: int) -> StringConstraints:
    return StringConstraints(strip_whitespace=True, min_length=1, max_length=max_length)


Title = Annotated[str, text(400)]
Brand = Annotated[str, text(100)]
Color = Annotated[str, text(100)]
Description = Annotated[str, text(5000)]
Bullet = Annotated[str, text(1000)]


class UpsertRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    corpus: Literal["esci"]
    doc_id: DocId
    version: Version
    title: Title
    brand: Brand | None = None
    color: Color | None = None
    bullets: list[Bullet] = Field(default_factory=list, max_length=20)
    description: Description | None = None
    locale: Locale


class _EventBase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1]
    corpus: Literal["esci"]
    doc_id: DocId
    version: Version
    produced_at: AwareDatetime


class UpsertEvent(_EventBase):
    op: Literal["upsert"]
    s3_key: str
    content_sha: Sha256Hex
    size_bytes: Annotated[int, Field(ge=0)]


class DeleteEvent(_EventBase):
    op: Literal["delete"]


DocEvent = UpsertEvent | DeleteEvent
doc_event_adapter: TypeAdapter[DocEvent] = TypeAdapter(
    Annotated[DocEvent, Field(discriminator="op")]
)
