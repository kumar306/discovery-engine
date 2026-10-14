"""The dev setup script against the real stack: first run creates, second run changes nothing."""

import uuid

import pytest
from qdrant_client import QdrantClient

from scripts.dev_init import ensure_bucket, main
from services.embed.settings import EmbedSettings
from services.ingest.settings import IngestSettings

from .conftest import Database, moto_client


def test_a_missing_bucket_is_created_and_a_second_call_says_it_was_already_there() -> None:
    settings = IngestSettings(s3_bucket=f"dev-init-{uuid.uuid4().hex[:12]}")
    client = moto_client()
    try:
        assert ensure_bucket(settings) is True
        assert ensure_bucket(settings) is False
        names = [item["Name"] for item in client.list_buckets()["Buckets"]]
        assert names.count(settings.s3_bucket) == 1
    finally:
        client.delete_bucket(Bucket=settings.s3_bucket)


def test_main_sets_up_an_empty_database_and_bucket_and_a_repeat_changes_nothing(
    empty_database: Database, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bucket = f"dev-init-{uuid.uuid4().hex[:12]}"
    collection = f"search_test_{uuid.uuid4().hex[:10]}"
    monkeypatch.setenv("INGEST_DATABASE_DSN", empty_database.pooled_dsn)
    monkeypatch.setenv("INGEST_S3_BUCKET", bucket)
    monkeypatch.setenv("EMBED_QDRANT_COLLECTION", collection)
    client = moto_client()
    qdrant = QdrantClient(url=EmbedSettings().qdrant_url, timeout=10)
    try:
        main()
        first = capsys.readouterr().out
        main()
        second = capsys.readouterr().out
    finally:
        client.delete_bucket(Bucket=bucket)
        qdrant.delete_collection(collection)
        qdrant.close()
    assert "0001_init.sql" in first
    assert f"bucket {bucket}: created" in first
    assert "none, already up to date" in second
    assert f"bucket {bucket}: already there" in second
    assert f"collection {collection}: created" in first
    assert f"collection {collection}: already there" in second
