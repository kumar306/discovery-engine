"""The health and metrics routes of the ingest api, on the same port as the traffic."""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from prometheus_client import CollectorRegistry

from libs.observability.health import Probe
from services.ingest.app import create_app
from services.ingest.service import WriteRefused
from services.ingest.settings import IngestSettings
from tests.unit.test_api_edges import RecordingIngest, post, product


async def fine() -> None:
    return None


async def broken() -> None:
    raise ConnectionError("down")


class Api:
    def __init__(self, service: RecordingIngest, client: httpx.AsyncClient) -> None:
        self.service = service
        self.client = client
        self.registry: CollectorRegistry = CollectorRegistry()

    def sample(self, name: str, **labels: str) -> float | None:
        return self.registry.get_sample_value(name, labels)


async def make_api(
    probes: dict[str, Probe] | None = None, **settings: Any
) -> tuple[Api, httpx.AsyncClient]:
    service = RecordingIngest()
    registry = CollectorRegistry()
    app = create_app(
        service,
        IngestSettings(**settings),
        probes=probes,
        registry=registry,
    )
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://api")
    api = Api(service, client)
    api.registry = registry
    return api, client


@pytest.fixture
async def api() -> AsyncIterator[Api]:
    served, client = await make_api({"postgres": fine, "s3": fine})
    async with client:
        yield served


# liveness and readiness


async def test_the_api_is_alive_when_it_answers(api: Api) -> None:
    response = await api.client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"alive": True}


async def test_ready_when_postgres_and_s3_answer(api: Api) -> None:
    response = await api.client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {"ready": True, "checks": {"postgres": "ok", "s3": "ok"}}


async def test_not_ready_when_a_dependency_is_down_but_still_alive() -> None:
    _, client = await make_api({"postgres": fine, "s3": broken})
    async with client:
        response = await client.get("/readyz")
        assert response.status_code == 503
        assert response.json()["checks"] == {"postgres": "ok", "s3": "failed: ConnectionError"}
        assert (await client.get("/healthz")).status_code == 200  # restarting would not fix s3


async def test_with_no_probes_the_api_is_ready() -> None:
    _, client = await make_api()
    async with client:
        assert (await client.get("/readyz")).json() == {"ready": True, "checks": {}}


# the health routes answer while the api is full, and a full api is still ready


async def test_health_and_metrics_answer_while_the_api_is_full_and_it_stays_ready() -> None:
    served, client = await make_api(
        {"postgres": fine}, max_inflight_requests=1, admission_wait_s=0.05
    )
    served.service.release.clear()  # the first request will hold its slot
    async with client:
        holder = asyncio.create_task(post(client, product()))
        await asyncio.wait_for(served.service.entered.wait(), timeout=5)
        refused = await post(client, product(doc_id="D2"))
        assert refused.status_code == 503  # the slot is taken, load is shed
        for path in ("/healthz", "/readyz", "/metrics"):
            assert (await client.get(path)).status_code == 200, path
        served.service.release.set()
        assert (await holder).status_code == 202


# the request metrics


async def test_requests_are_counted_by_route_and_status_class(api: Api) -> None:
    await post(api.client, product())  # 202
    await post(api.client, {"bogus": 1})  # 422
    api.service.result = WriteRefused("stale_version", 5)
    await post(api.client, product())  # 409
    api.service.error = RuntimeError("boom")
    await post(api.client, product())  # 500
    route = "/v1/documents"
    assert api.sample("ingest_requests_total", route=route, status_class="2xx") == 1
    assert api.sample("ingest_requests_total", route=route, status_class="4xx") == 2
    assert api.sample("ingest_requests_total", route=route, status_class="5xx") == 1
    assert api.sample("ingest_request_seconds_count", route=route) == 4


async def test_a_shed_request_is_counted_as_a_server_error() -> None:
    served, client = await make_api(max_inflight_requests=1, admission_wait_s=0.05)
    served.service.release.clear()
    async with client:
        holder = asyncio.create_task(post(client, product()))
        await asyncio.wait_for(served.service.entered.wait(), timeout=5)
        await post(client, product(doc_id="D2"))
        served.service.release.set()
        await holder
    assert served.sample("ingest_requests_total", route="/v1/documents", status_class="5xx") == 1


async def test_the_label_is_the_route_and_not_the_path_so_ids_do_not_make_new_series(
    api: Api,
) -> None:
    for number in range(30):
        await api.client.delete(f"/v1/documents/DOC{number}", params={"version": 3})
    template = "/v1/documents/{doc_id}"
    assert api.sample("ingest_requests_total", route=template, status_class="2xx") == 30
    series = [
        sample
        for family in api.registry.collect()
        for sample in family.samples
        if sample.name == "ingest_requests_total"
    ]
    assert len(series) == 1  # thirty documents, one series


async def test_a_path_that_matches_no_route_is_counted_as_other(api: Api) -> None:
    for path in ("/nope", "/v1/other", "/wp-admin/setup.php"):
        assert (await api.client.get(path)).status_code == 404
    assert api.sample("ingest_requests_total", route="other", status_class="4xx") == 3


async def test_the_health_and_metrics_routes_do_not_count_themselves(api: Api) -> None:
    for path in ("/healthz", "/readyz", "/metrics") * 5:
        await api.client.get(path)
    series = [
        sample
        for family in api.registry.collect()
        for sample in family.samples
        if sample.name == "ingest_requests_total"
    ]
    assert series == []  # a probe every ten seconds would otherwise outnumber the real traffic


async def test_metrics_shows_the_counts_in_the_prometheus_format(api: Api) -> None:
    await post(api.client, product())
    text = (await api.client.get("/metrics")).text
    assert 'ingest_requests_total{route="/v1/documents",status_class="2xx"} 1.0' in text
