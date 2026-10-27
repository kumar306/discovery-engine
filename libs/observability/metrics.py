from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

# every metric of a process is defined here, with its name and its label names fixed, because
# dashboards and alerts are written against them
# labels are small closed sets (a result, a job, an error type), never a doc id or a request id,
# one label value that grows with the data would give prometheus a new series for each of them
# each process builds its own set on its own registry, so a test can use a fresh registry and
# nothing is registered at import time

# a skipped event takes milliseconds, a batch of embeddings can take many seconds
EVENT_SECONDS_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)


# an api call takes milliseconds when it is refused and some seconds when s3 or postgres is slow
REQUEST_SECONDS_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

# one step of a search request: fusing takes a fraction of a millisecond, encoding one query about
# five (so the edges around 5 ms are close together, or every encode would land in one bucket and no
# quantile could be read), a Qdrant search tens of milliseconds, a retriever that timed out 2 s
STAGE_SECONDS_BUCKETS = (
    0.00025,
    0.0005,
    0.001,
    0.002,
    0.003,
    0.005,
    0.0075,
    0.01,
    0.015,
    0.025,
    0.04,
    0.06,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
)


class IngestMetrics:
    """What the ingest api counts, one series per route and status class, never per document."""

    def __init__(self, registry: CollectorRegistry) -> None:
        # route: the route template such as /v1/documents/{doc_id}, or other when none matched
        # status_class: 2xx, 4xx or 5xx
        self.requests = Counter(
            "ingest_requests_total",
            "Requests answered by the ingest api.",
            ["route", "status_class"],
            registry=registry,
        )
        self.request_seconds = Histogram(
            "ingest_request_seconds",
            "Time to answer one request.",
            ["route"],
            buckets=REQUEST_SECONDS_BUCKETS,
            registry=registry,
        )


class QueryMetrics:
    """What the query service counts, per route and per stage, never per query or per product."""

    def __init__(self, registry: CollectorRegistry) -> None:
        # route: the route template such as /v1/search, or other when none matched
        # status_class: 2xx, 4xx or 5xx
        self.requests = Counter(
            "query_requests_total",
            "Requests answered by the query service.",
            ["route", "status_class"],
            registry=registry,
        )
        self.request_seconds = Histogram(
            "query_request_seconds",
            "Time to answer one request.",
            ["route"],
            buckets=REQUEST_SECONDS_BUCKETS,
            registry=registry,
        )
        # flow: search or chunks
        # stage: dense and sparse are a whole retriever (encode, then search), encode_dense,
        # search_dense, encode_sparse and search_sparse are its two steps, fuse is the fusion;
        # a step that did not finish (it raised, or the retriever timed out) is not observed
        self.stage_seconds = Histogram(
            "query_stage_seconds",
            "Time spent in one stage of a flow.",
            ["flow", "stage"],
            buckets=STAGE_SECONDS_BUCKETS,
            registry=registry,
        )
        # no labels: how long a dense query job waited for a model thread before its call began,
        # near zero while the encoder keeps up, growing when it is the thing that saturates
        # a job that gives up while still queued never runs, so it never reports a wait: under
        # heavy overload this histogram hides its own worst cases, read it beside query_degraded
        # and the dense stage of query_stage_seconds
        self.encode_wait_seconds = Histogram(
            "query_encode_wait_seconds",
            "Time a dense query job waited in the queue in front of the model thread.",
            buckets=STAGE_SECONDS_BUCKETS,
            registry=registry,
        )
        # no labels, all four are read or counted at the admission limit and the encoder queue:
        # the number of requests holding a permit, waiting for one, and refused because none
        # came free in time (a 503 overloaded, not any other 503); the dense jobs queued in front
        # of the model threads. The gauges are read at the moment of a scrape, so a spike shorter
        # than the scrape interval is not seen in them
        self.encode_queued = Gauge(
            "query_encode_queued",
            "Dense query jobs queued in front of the model threads right now.",
            registry=registry,
        )
        self.admission_in_flight = Gauge(
            "query_admission_in_flight",
            "Requests holding an admission permit right now.",
            registry=registry,
        )
        self.admission_waiting = Gauge(
            "query_admission_waiting",
            "Requests waiting for an admission permit right now.",
            registry=registry,
        )
        self.admission_shed = Counter(
            "query_admission_shed",
            "Requests refused with a 503 because no admission permit came free in time.",
            registry=registry,
        )
        # retriever: dense or sparse, the one that was left out of an answer
        self.degraded = Counter(
            "query_degraded_total",
            "Answers served without one of the two retrievers.",
            ["flow", "retriever"],
            registry=registry,
        )


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
