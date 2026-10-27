"""The metrics are an interface: dashboards and alerts depend on the names and the labels."""

from typing import Any

import pytest
from prometheus_client import CollectorRegistry

from libs.observability.metrics import (
    STAGE_SECONDS_BUCKETS,
    EmbedMetrics,
    IngestMetrics,
    MaintenanceMetrics,
    QueryMetrics,
    RelayMetrics,
)

# name (without the _total suffix prometheus adds to counters) -> (type, label names)
EMBED: dict[str, tuple[str, tuple[str, ...]]] = {
    "embed_events": ("counter", ("result",)),
    "embed_event_seconds": ("histogram", ()),
    "embed_dead_letters": ("counter", ("error_type",)),
    "embed_held_partitions": ("gauge", ()),
    "embed_oldest_hold_age_seconds": ("gauge", ()),
    "embed_chunk_being_deleted": ("counter", ()),
}
MAINTENANCE: dict[str, tuple[str, tuple[str, ...]]] = {
    "maintenance_job_runs": ("counter", ("job", "outcome")),
    "maintenance_job_last_success_seconds": ("gauge", ("job",)),
    "embed_stale_chunks_oldest_age_seconds": ("gauge", ()),
    "embed_stale_chunks_deleted": ("counter", ()),
}
RELAY: dict[str, tuple[str, tuple[str, ...]]] = {
    "outbox_backlog": ("gauge", ()),
}
INGEST: dict[str, tuple[str, tuple[str, ...]]] = {
    "ingest_requests": ("counter", ("route", "status_class")),
    "ingest_request_seconds": ("histogram", ("route",)),
}


QUERY: dict[str, tuple[str, tuple[str, ...]]] = {
    "query_requests": ("counter", ("route", "status_class")),
    "query_request_seconds": ("histogram", ("route",)),
    "query_stage_seconds": ("histogram", ("flow", "stage")),
    "query_encode_wait_seconds": ("histogram", ()),
    "query_encode_queued": ("gauge", ()),
    "query_admission_in_flight": ("gauge", ()),
    "query_admission_waiting": ("gauge", ()),
    "query_admission_shed": ("counter", ()),
    "query_degraded": ("counter", ("flow", "retriever")),
}


def families(registry: CollectorRegistry) -> dict[str, str]:
    return {family.name: family.type for family in registry.collect()}


@pytest.mark.parametrize(
    ("build", "expected"),
    [
        (EmbedMetrics, EMBED),
        (MaintenanceMetrics, MAINTENANCE),
        (RelayMetrics, RELAY),
        (IngestMetrics, INGEST),
        (QueryMetrics, QUERY),
    ],
)
def test_each_process_exposes_exactly_the_agreed_metrics(
    build: Any, expected: dict[str, tuple[str, tuple[str, ...]]]
) -> None:
    registry = CollectorRegistry()
    build(registry)
    # nothing is missing and nothing extra was added without changing this table
    assert families(registry) == {name: kind for name, (kind, _) in expected.items()}


def test_the_label_names_are_fixed() -> None:
    registry = CollectorRegistry()
    embed, maintenance = EmbedMetrics(registry), MaintenanceMetrics(registry)
    assert embed.events._labelnames == ("result",)
    assert embed.dead_letters._labelnames == ("error_type",)
    assert maintenance.job_runs._labelnames == ("job", "outcome")
    assert maintenance.job_last_success._labelnames == ("job",)
    # a label that is not on the list is refused, so an id can never become a label by accident
    with pytest.raises(ValueError, match="Incorrect label names"):
        embed.events.labels(doc_id="D1")


def test_a_counter_counts_and_a_gauge_is_set() -> None:
    registry = CollectorRegistry()
    embed = EmbedMetrics(registry)
    embed.events.labels(result="applied").inc()
    embed.events.labels(result="applied").inc()
    embed.events.labels(result="skipped").inc()
    embed.held_partitions.set(2)
    assert registry.get_sample_value("embed_events_total", {"result": "applied"}) == 2
    assert registry.get_sample_value("embed_events_total", {"result": "skipped"}) == 1
    assert registry.get_sample_value("embed_held_partitions") == 2


def test_the_event_time_histogram_has_buckets_from_milliseconds_to_seconds() -> None:
    registry = CollectorRegistry()
    EmbedMetrics(registry).event_seconds.observe(0.3)
    bounds = sorted(
        float(sample.labels["le"])
        for family in registry.collect()
        for sample in family.samples
        if sample.name == "embed_event_seconds_bucket"
    )
    # a skipped event takes milliseconds, a batch of embeddings can take many seconds
    assert bounds[0] <= 0.01
    assert bounds[-2] >= 10.0  # the last bound is +Inf
    assert bounds == sorted(set(bounds))


def test_every_process_has_its_own_registry_and_a_second_copy_in_one_registry_is_an_error() -> None:
    one, two = CollectorRegistry(), CollectorRegistry()
    EmbedMetrics(one)
    EmbedMetrics(two)  # a separate registry does not clash
    with pytest.raises(ValueError, match="Duplicated timeseries"):
        EmbedMetrics(one)


def bucket_bounds(registry: CollectorRegistry, name: str) -> list[float]:
    return sorted(
        float(sample.labels["le"])
        for family in registry.collect()
        for sample in family.samples
        if sample.name == f"{name}_bucket"
    )


def test_the_stage_buckets_can_tell_a_4_milliseconds_encode_from_a_9_milliseconds_one() -> None:
    # a bge-small encode of one query takes about 5 ms: with the request buckets (5, 10, 25 ms)
    # every encode fell into the first bucket and no quantile could be read
    assert list(STAGE_SECONDS_BUCKETS) == sorted(set(STAGE_SECONDS_BUCKETS))
    assert len([b for b in STAGE_SECONDS_BUCKETS if b < 0.005]) >= 4
    assert len([b for b in STAGE_SECONDS_BUCKETS if 0.005 <= b <= 0.015]) >= 3
    assert STAGE_SECONDS_BUCKETS[-1] >= 2.5  # the 2 s retriever timeout still lands in a bucket


def test_the_stage_histogram_uses_the_stage_buckets_and_the_request_histogram_does_not() -> None:
    registry = CollectorRegistry()
    metrics = QueryMetrics(registry)
    metrics.stage_seconds.labels(flow="search", stage="encode_dense").observe(0.0048)
    metrics.request_seconds.labels(route="/v1/search").observe(0.04)
    stage = bucket_bounds(registry, "query_stage_seconds")
    request = bucket_bounds(registry, "query_request_seconds")
    assert stage[:-1] == list(STAGE_SECONDS_BUCKETS)  # the last bound is +Inf
    assert request[0] == 0.005
    assert stage[0] < request[0]


def test_the_encoder_wait_histogram_uses_the_stage_buckets() -> None:
    registry = CollectorRegistry()
    QueryMetrics(registry).encode_wait_seconds.observe(0.0004)
    assert bucket_bounds(registry, "query_encode_wait_seconds")[:-1] == list(STAGE_SECONDS_BUCKETS)
