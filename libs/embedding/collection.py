import unicodedata
from enum import StrEnum

from qdrant_client import AsyncQdrantClient, models

from libs.embedding.settings import IndexSettings

DENSE = "dense"
SPARSE = "bm25"

# the payload fields a search or a flip filters on get an index
# without one qdrant scans the payload of every point to find the state, doc_id or version
# we ask for, with one it jumps to the matching points, so moving staging to live never
# means a full scan, a missing index only makes the filters slow
PAYLOAD_INDEXES = {
    "state": models.PayloadSchemaType.KEYWORD,
    "doc_id": models.PayloadSchemaType.KEYWORD,
    "locale": models.PayloadSchemaType.KEYWORD,
    "version": models.PayloadSchemaType.INTEGER,
    # the lower case forms filters compare, see filter_key
    "brand_key": models.PayloadSchemaType.KEYWORD,
    "color_key": models.PayloadSchemaType.KEYWORD,
}


def filter_key(text: str) -> str:
    """The form of a brand or a color that filters compare.

    The worker stores it as brand_key and color_key, the query service applies it to the value in a
    request, so "Nike", " NIKE " and "nike" all become "nike". Unicode is composed first, so an "e"
    plus an accent and a single accented "e" give the same key, and case folding is the full
    lower casing of Unicode ("Straße" becomes "strasse"), not only ASCII.
    """
    composed = unicodedata.normalize("NFC", text)
    return " ".join(unicodedata.normalize("NFC", composed.casefold()).split())


class QdrantPointState(StrEnum):
    # every search filters on live, so the other three are invisible to readers
    QDRANT_POINT_STAGING = "staging"
    QDRANT_POINT_LIVE = "live"
    QDRANT_POINT_SUPERSEDED = "superseded"
    QDRANT_POINT_DELETED = "deleted"


class CollectionMismatch(RuntimeError):
    """The collection is missing or was built for a different model, the worker must not run."""


# at worker startup: the worker has its model version baked into the settings property, built
# from the model name, the bm25 length and the revision, and this compares it with the one
# stored in the collection, so a worker for another model refuses to start
# the collection may not exist yet, that is flagged here too, with the fix in the message
async def verify_collection(client: AsyncQdrantClient, settings: IndexSettings) -> None:
    name = settings.qdrant_collection
    if not await client.collection_exists(name):
        raise CollectionMismatch(f"collection {name} does not exist, run make init")
    info = await client.get_collection(name)
    problems: list[str] = []
    dense = (info.config.params.vectors or {}).get(DENSE)  # type: ignore[union-attr]
    if not isinstance(dense, models.VectorParams):
        problems.append("the dense vector is missing")
    else:
        if dense.size != settings.dense_dim:
            problems.append(f"dense size is {dense.size}, expected {settings.dense_dim}")
        if dense.distance != models.Distance.COSINE:
            problems.append(f"dense distance is {dense.distance}, expected cosine")
    sparse = (info.config.params.sparse_vectors or {}).get(SPARSE)
    if sparse is None:
        problems.append("the bm25 sparse vector is missing")
    elif sparse.modifier != models.Modifier.IDF:
        problems.append("the bm25 sparse vector has no idf modifier")
    stored = (info.config.metadata or {}).get("model_version")
    if stored != settings.model_version:
        problems.append(f"model_version is {stored!r}, expected {settings.model_version!r}")
    for field, schema in PAYLOAD_INDEXES.items():
        index = info.payload_schema.get(field)
        if index is None or index.data_type != schema:
            problems.append(f"payload index on {field} is missing or has the wrong type")
    if problems:
        raise CollectionMismatch(f"collection {name}: " + "; ".join(problems))
