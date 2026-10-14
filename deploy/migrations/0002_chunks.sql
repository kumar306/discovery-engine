-- the chunks of the version the worker last committed, one row per distinct chunk text
create table chunks (
    corpus       text  not null,
    doc_id       text  not null,
    -- uuid5 of document, model version and content hash, also the id of the qdrant point
    chunk_id     uuid  not null,
    content_hash text  not null check (content_hash ~ '^[0-9a-f]{64}$'),
    -- every place this text occurs in the document, the first one orders the chunks
    positions    int[] not null check (cardinality(positions) >= 1 and 0 <= all(positions)),
    primary key (corpus, doc_id, chunk_id),
    foreign key (corpus, doc_id) references documents (corpus, doc_id)
);

-- points that no version needs any more, the compactor deletes them from qdrant later
create table stale_chunks (
    corpus   text        not null,
    doc_id   text        not null,
    chunk_id uuid        not null,
    stale_at timestamptz not null default now(),
    primary key (corpus, doc_id, chunk_id),
    foreign key (corpus, doc_id) references documents (corpus, doc_id)
);
