"""The worker builds its pipeline from its own settings, each part from the settings it owns."""

from typing import Any

from services.embed.settings import EmbedSettings
from services.embed.worker import build_pipeline
from tests.support.hash_embedder import HashEmbedder


def test_each_part_of_the_pipeline_gets_the_setting_that_belongs_to_it() -> None:
    settings = EmbedSettings(db_lock_timeout_s=5.0, qdrant_collection="chunks_blue")
    nothing: Any = None  # the parts are only wired here, nothing is called
    pipeline = build_pipeline(settings, nothing, nothing, nothing, HashEmbedder())
    assert pipeline._pg_chunk_repo._lock_timeout_ms == 5000
    assert pipeline._qdrant_index._qdrant_collection == "chunks_blue"
    assert pipeline._embed_settings is settings
