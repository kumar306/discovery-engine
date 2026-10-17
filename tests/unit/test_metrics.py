"""The metrics are an interface: dashboards and alerts depend on the names and the labels."""

from typing import Any

import pytest
from prometheus_client import CollectorRegistry

from libs.observability.metrics import EmbedMetrics, MaintenanceMetrics, RelayMetrics

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


def families(registry: CollectorRegistry) -> dict[str, str]:
    return {family.name: family.type for family in registry.collect()}


@pytest.mark.parametrize(
    ("build", "expected"),
    [(EmbedMetrics, EMBED), (MaintenanceMetrics, MAINTENANCE), (RelayMetrics, RELAY)],
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
