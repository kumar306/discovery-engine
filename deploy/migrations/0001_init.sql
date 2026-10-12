-- one row per document: the three version numbers say how far each stage got
create table documents (
    corpus             text        not null,
    doc_id             text        not null,
    -- newest version the ingest api accepted, written by ingest
    latest_version     bigint      not null check (latest_version >= 0),
    -- sha of the newest accepted content, null when that version is a delete
    latest_content_sha text        check (latest_content_sha ~ '^[0-9a-f]{64}$'),
    -- newest version the worker committed, written by the worker
    current_version    bigint      check (current_version >= 0),
    -- newest version that is live in qdrant, what readers trust
    indexed_version    bigint      check (indexed_version >= 0),
    deleted            boolean     not null default false,
    created_at         timestamptz not null default now(),
    updated_at         timestamptz not null default now(),
    primary key (corpus, doc_id),
    constraint current_not_ahead_of_latest
        check (current_version is null or current_version <= latest_version),
    constraint indexed_not_ahead_of_current
        check (indexed_version is null or (current_version is not null and indexed_version <= current_version))
);

-- events waiting to be published to kafka, the relay claims rows in id order
create table outbox (
    id           bigint generated always as identity primary key,
    doc_id       text        not null,
    request_id   text        not null,
    event_schema text        not null,
    payload      jsonb       not null,
    created_at   timestamptz not null default now(),
    published_at timestamptz
);

-- the relay only ever looks for rows that are not published yet
create index outbox_unpublished_idx on outbox (id) where published_at is null;
