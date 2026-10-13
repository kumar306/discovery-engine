"""The production wiring: the same app the server runs, built from settings, served for real."""

import httpx
import pytest

from libs.schemas.hashing import canonical_bytes
from services.ingest.main import build
from services.ingest.settings import IngestSettings

from .conftest import Database, document, moto_client, outbox, request_for


async def test_the_wired_app_opens_its_pool_serves_a_write_and_closes_cleanly(
    db: Database, bucket: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # moto accepts any credentials, in a pod these come from the role
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    settings = IngestSettings(database_dsn=db.pooled_dsn, s3_bucket=bucket)
    app = build(settings)
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://api") as client,
    ):
        body = request_for("D1", 1).model_dump(mode="json")
        response = await client.post("/v1/documents", json=body)
    assert response.status_code == 202
    assert document(db.direct_dsn, "D1") is not None
    assert len(outbox(db.direct_dsn)) == 1
    stored = moto_client().get_object(Bucket=bucket, Key="esci/D1/v1.json")["Body"].read()
    assert stored == canonical_bytes(request_for("D1", 1))
