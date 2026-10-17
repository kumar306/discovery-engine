from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

# every metric of a process is defined here, with its name and its label names fixed, because
# dashboards and alerts are written against them
# labels are small closed sets (a result, a job, an error type), never a doc id or a request id,
# one label value that grows with the data would give prometheus a new series for each of them
# each process builds its own set on its own registry, so a test can use a fresh registry and
# nothing is registered at import time

# a skipped event takes milliseconds, a batch of embeddings can take many seconds
EVENT_SECONDS_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)


class EmbedMetrics:
    """What the indexing worker counts."""

    def __init__(self, registry: CollectorRegistry) -> None:
        # result: applied, skipped, dead_lettered or held
        self.events = Counter(
            "embed_events_total", "Events the worker finished with.", ["result"], registry=registry
        )
        self.event_seconds = Histogram(
            "embed_event_seconds",
            "Time to handle one event, retries included.",
            buckets=EVENT_SECONDS_BUCKETS,
            registry=registry,
        )
        # error_type: the class name of the error, a finite set in our code
        self.dead_letters = Counter(
            "embed_dead_letters_total",
            "Events written to the dead letter topic.",
            ["error_type"],
            registry=registry,
        )
        self.held_partitions = Gauge(
            "embed_held_partitions",
            "Partitions paused until a failing step recovers.",
            registry=registry,
        )
        # the alert on a stuck partition is written against this one
        self.oldest_hold_age = Gauge(
            "embed_oldest_hold_age_seconds",
            "Seconds the longest held partition has been held, zero when none is.",
            registry=registry,
        )
        self.chunk_being_deleted = Counter(
            "embed_chunk_being_deleted_total",
            "Times an event waited because the compactor was deleting a chunk it needs.",
            registry=registry,
        )


class MaintenanceMetrics:
    """What the maintenance process counts."""

    def __init__(self, registry: CollectorRegistry) -> None:
        # job: compactor. outcome: ok or error
        self.job_runs = Counter(
            "maintenance_job_runs_total",
            "Passes of a maintenance job.",
            ["job", "outcome"],
            registry=registry,
        )
        self.job_last_success = Gauge(
            "maintenance_job_last_success_seconds",
            "Unix time of the last pass of a job that did not fail.",
            ["job"],
            registry=registry,
        )
        self.stale_oldest_age = Gauge(
            "embed_stale_chunks_oldest_age_seconds",
            "Age of the oldest chunk waiting in the stale queue.",
            registry=registry,
        )
        self.stale_deleted = Counter(
            "embed_stale_chunks_deleted_total",
            "Stale chunks whose point and queue row were removed.",
            registry=registry,
        )


class RelayMetrics:
    """What the outbox relay counts."""

    def __init__(self, registry: CollectorRegistry) -> None:
        self.outbox_backlog = Gauge(
            "outbox_backlog",
            "Events written by the ingest api and not yet sent.",
            registry=registry,
        )
