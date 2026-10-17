-- set when a compactor takes the row, null while it waits
-- one column is both the flag and the timer, so the two can never disagree after a crash
alter table stale_chunks add column deletion_started_at timestamptz;

-- the claim looks for the oldest rows past the grace period without reading the whole queue
create index stale_chunks_stale_at_idx on stale_chunks (stale_at);
