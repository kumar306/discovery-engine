"""The first schema: the right shape, and the database refuses states that must never exist."""

from typing import Any

import psycopg
import pytest
from psycopg import sql

from .conftest import Database

SHA = "a" * 64

DOCUMENT_COLUMNS = {
    "corpus": ("text", "NO"),
    "doc_id": ("text", "NO"),
    "latest_version": ("bigint", "NO"),
    "latest_content_sha": ("text", "YES"),
    "current_version": ("bigint", "YES"),
    "indexed_version": ("bigint", "YES"),
    "deleted": ("boolean", "NO"),
    "created_at": ("timestamp with time zone", "NO"),
    "updated_at": ("timestamp with time zone", "NO"),
}

OUTBOX_COLUMNS = {
    "id": ("bigint", "NO"),
    "doc_id": ("text", "NO"),
    "request_id": ("text", "NO"),
    "event_schema": ("text", "NO"),
    "payload": ("jsonb", "NO"),
    "created_at": ("timestamp with time zone", "NO"),
    "published_at": ("timestamp with time zone", "YES"),
}


def columns(dsn: str, table: str) -> dict[str, tuple[str, str]]:
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(
            "select column_name, data_type, is_nullable from information_schema.columns "
            "where table_schema = 'public' and table_name = %s",
            (table,),
        ).fetchall()
    return {name: (data_type, nullable) for name, data_type, nullable in rows}


def primary_key(dsn: str, table: str) -> list[str]:
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(
            "select a.attname from pg_index i "
            "join pg_attribute a on a.attrelid = i.indrelid and a.attnum = any(i.indkey) "
            "where i.indrelid = %s::regclass and i.indisprimary "
            "order by array_position(i.indkey, a.attnum)",
            (table,),
        ).fetchall()
    return [name for (name,) in rows]


def insert_document(dsn: str, **values: Any) -> None:
    row: dict[str, Any] = {"corpus": "esci", "doc_id": "D1", "latest_version": 5}
    row.update(values)
    statement = sql.SQL("insert into documents ({}) values ({})").format(
        sql.SQL(", ").join(sql.Identifier(name) for name in row),
        sql.SQL(", ").join(sql.Placeholder() for _ in row),
    )
    with psycopg.connect(dsn) as conn:
        conn.execute(statement, list(row.values()))


def test_documents_has_the_expected_columns_and_key(migrated_database: Database) -> None:
    assert columns(migrated_database.direct_dsn, "documents") == DOCUMENT_COLUMNS
    assert primary_key(migrated_database.direct_dsn, "documents") == ["corpus", "doc_id"]


def test_outbox_has_the_expected_columns_and_key(migrated_database: Database) -> None:
    assert columns(migrated_database.direct_dsn, "outbox") == OUTBOX_COLUMNS
    assert primary_key(migrated_database.direct_dsn, "outbox") == ["id"]


def test_unpublished_outbox_rows_have_their_own_small_index(migrated_database: Database) -> None:
    with psycopg.connect(migrated_database.direct_dsn) as conn:
        row = conn.execute(
            "select indexdef from pg_indexes "
            "where tablename = 'outbox' and indexname = 'outbox_unpublished_idx'"
        ).fetchone()
    assert row is not None
    assert "WHERE (published_at IS NULL)" in row[0]


@pytest.mark.parametrize(
    "values",
    [
        pytest.param({"current_version": 5, "indexed_version": 5}, id="fully indexed"),
        pytest.param({}, id="accepted, nothing applied yet"),
        pytest.param({"current_version": 4, "indexed_version": 3}, id="behind but ordered"),
        pytest.param({"current_version": 5}, id="committed, not live yet"),
        pytest.param({"latest_content_sha": SHA}, id="a real content sha"),
        pytest.param({"latest_content_sha": None}, id="a delete has no sha"),
        pytest.param({"latest_version": 0}, id="version zero is allowed"),
    ],
)
def test_valid_documents_are_accepted(db: Database, values: dict[str, Any]) -> None:
    insert_document(db.direct_dsn, **values)


@pytest.mark.parametrize(
    "values",
    [
        pytest.param({"current_version": 6}, id="current ahead of latest"),
        pytest.param({"current_version": 5, "indexed_version": 6}, id="indexed ahead of current"),
        pytest.param({"indexed_version": 1}, id="indexed with nothing committed"),
        pytest.param({"latest_version": -1}, id="negative latest"),
        pytest.param({"current_version": -1}, id="negative current"),
        pytest.param({"latest_content_sha": "xyz"}, id="sha that is not hex"),
        pytest.param({"latest_content_sha": SHA.upper().replace("A", "F")}, id="uppercase sha"),
        pytest.param({"latest_content_sha": SHA[:-1]}, id="sha that is too short"),
    ],
)
def test_impossible_documents_are_refused_by_the_database(
    db: Database, values: dict[str, Any]
) -> None:
    with pytest.raises(psycopg.errors.CheckViolation):
        insert_document(db.direct_dsn, **values)


def test_a_document_key_exists_once(db: Database) -> None:
    insert_document(db.direct_dsn)
    with pytest.raises(psycopg.errors.UniqueViolation):
        insert_document(db.direct_dsn)


def test_the_same_doc_id_may_exist_in_another_corpus(db: Database) -> None:
    insert_document(db.direct_dsn, corpus="esci")
    insert_document(db.direct_dsn, corpus="other")


ADD_EVENT = (
    "insert into outbox (doc_id, request_id, event_schema, payload) values ('D1', 'r', 's', '{}')"
)


def test_outbox_ids_come_from_the_database_in_order_and_cannot_be_chosen(db: Database) -> None:
    with psycopg.connect(db.direct_dsn) as conn:
        ids = [conn.execute(ADD_EVENT + " returning id").fetchone() for _ in range(3)]
    # the fixture restarts the identity before every test
    assert ids == [(1,), (2,), (3,)]
    chosen = (
        "insert into outbox (id, doc_id, request_id, event_schema, payload) "
        "values (99, 'D1', 'r', 's', '{}')"
    )
    with pytest.raises(psycopg.errors.GeneratedAlways), psycopg.connect(db.direct_dsn) as conn:
        conn.execute(chosen)


def test_a_new_outbox_row_is_unpublished_and_needs_a_payload(db: Database) -> None:
    with psycopg.connect(db.direct_dsn) as conn:
        row = conn.execute(ADD_EVENT + " returning published_at, created_at is not null").fetchone()
    assert row == (None, True)
    without_payload = (
        "insert into outbox (doc_id, request_id, event_schema) values ('D1', 'r', 's')"
    )
    with pytest.raises(psycopg.errors.NotNullViolation), psycopg.connect(db.direct_dsn) as conn:
        conn.execute(without_payload)
