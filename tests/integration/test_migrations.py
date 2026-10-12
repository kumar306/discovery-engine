"""The migration runner: numbered sql files applied once, in order, safely, and never rewritten."""

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psycopg
import pytest

from libs.db.migrate import MIGRATIONS_DIR, MigrationError, migrate

from .conftest import Database


def write(directory: Path, files: dict[str, str]) -> Path:
    for name, sql in files.items():
        (directory / name).write_text(sql, encoding="utf-8")
    return directory


def tables(dsn: str) -> set[str]:
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(
            "select table_name from information_schema.tables where table_schema = 'public'"
        ).fetchall()
    return {name for (name,) in rows}


def recorded(dsn: str) -> list[tuple[str, str]]:
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(
            "select version, name from schema_migrations order by version"
        ).fetchall()
    return rows


THREE = {
    "0001_first.sql": "create table one (id int primary key);",
    "0002_second.sql": "create table two (id int primary key, one_id int references one (id));",
    "0003_third.sql": "alter table two add column note text;",
}


def test_the_first_run_applies_every_migration_and_records_it(empty_database: Database) -> None:
    applied = migrate(empty_database.pooled_dsn)
    assert applied == sorted(path.name for path in MIGRATIONS_DIR.glob("*.sql"))
    assert {"documents", "outbox", "schema_migrations"} <= tables(empty_database.direct_dsn)
    first = (MIGRATIONS_DIR / "0001_init.sql").read_text(encoding="utf-8")
    with psycopg.connect(empty_database.direct_dsn) as conn:
        row = conn.execute("select version, name, checksum from schema_migrations").fetchone()
    assert row == ("0001", "0001_init.sql", hashlib.sha256(first.encode("utf-8")).hexdigest())


def test_a_second_run_changes_nothing(empty_database: Database) -> None:
    migrate(empty_database.pooled_dsn)
    with psycopg.connect(empty_database.direct_dsn) as conn:
        before = conn.execute("select version, applied_at from schema_migrations").fetchall()
    assert migrate(empty_database.pooled_dsn) == []
    with psycopg.connect(empty_database.direct_dsn) as conn:
        after = conn.execute("select version, applied_at from schema_migrations").fetchall()
    assert after == before


def test_migrations_apply_in_number_order_each_once(
    empty_database: Database, tmp_path: Path
) -> None:
    # the second file refers to a table from the first, so a wrong order would fail
    write(tmp_path, THREE)
    assert migrate(empty_database.pooled_dsn, tmp_path) == list(THREE)
    assert [v for v, _ in recorded(empty_database.direct_dsn)] == ["0001", "0002", "0003"]


def test_a_new_file_is_applied_on_the_next_run_and_old_ones_are_left_alone(
    empty_database: Database, tmp_path: Path
) -> None:
    write(tmp_path, {"0001_first.sql": THREE["0001_first.sql"]})
    migrate(empty_database.pooled_dsn, tmp_path)
    write(tmp_path, {"0002_second.sql": THREE["0002_second.sql"]})
    assert migrate(empty_database.pooled_dsn, tmp_path) == ["0002_second.sql"]


def test_a_broken_migration_rolls_back_the_whole_file_and_stops_there(
    empty_database: Database, tmp_path: Path
) -> None:
    write(
        tmp_path,
        {
            "0001_first.sql": THREE["0001_first.sql"],
            # the first statement is fine, the second one fails: neither may stay
            "0002_broken.sql": "create table half_done (id int); create table one (id int);",
            "0003_third.sql": "create table never_reached (id int);",
        },
    )
    with pytest.raises(MigrationError, match=r"0002_broken\.sql"):
        migrate(empty_database.pooled_dsn, tmp_path)
    assert tables(empty_database.direct_dsn) >= {"one"}
    assert not tables(empty_database.direct_dsn) & {"half_done", "never_reached"}
    assert [v for v, _ in recorded(empty_database.direct_dsn)] == ["0001"]


def test_after_the_broken_file_is_fixed_the_rest_goes_through(
    empty_database: Database, tmp_path: Path
) -> None:
    write(
        tmp_path, {"0001_first.sql": THREE["0001_first.sql"], "0002_broken.sql": "not sql at all;"}
    )
    with pytest.raises(MigrationError):
        migrate(empty_database.pooled_dsn, tmp_path)
    write(tmp_path, {"0002_broken.sql": "create table fixed (id int);"})
    assert migrate(empty_database.pooled_dsn, tmp_path) == ["0002_broken.sql"]
    assert "fixed" in tables(empty_database.direct_dsn)


def test_editing_a_file_that_was_already_applied_is_refused(
    empty_database: Database, tmp_path: Path
) -> None:
    write(tmp_path, {"0001_first.sql": THREE["0001_first.sql"]})
    migrate(empty_database.pooled_dsn, tmp_path)
    write(tmp_path, {"0001_first.sql": "create table one (id int primary key, sneaky int);"})
    with pytest.raises(MigrationError, match="edited after it was applied"):
        migrate(empty_database.pooled_dsn, tmp_path)


def test_a_migration_that_was_applied_but_whose_file_is_gone_is_refused(
    empty_database: Database, tmp_path: Path
) -> None:
    write(tmp_path, THREE)
    migrate(empty_database.pooled_dsn, tmp_path)
    (tmp_path / "0002_second.sql").unlink()
    with pytest.raises(MigrationError, match="its file is gone"):
        migrate(empty_database.pooled_dsn, tmp_path)


@pytest.mark.parametrize(
    "bad_name",
    [
        "1_init.sql",
        "0001-init.sql",
        "0001_Init.sql",
        "init.sql",
        "0001_init.SQL.sql",
        "00001_init.sql",
    ],
)
def test_badly_named_files_are_refused_before_anything_runs(
    empty_database: Database, tmp_path: Path, bad_name: str
) -> None:
    write(tmp_path, {"0001_ok.sql": "create table ok (id int);", bad_name: "select 1;"})
    with pytest.raises(MigrationError, match="not named like"):
        migrate(empty_database.pooled_dsn, tmp_path)
    assert "ok" not in tables(empty_database.direct_dsn)


def test_two_files_with_the_same_number_are_refused(
    empty_database: Database, tmp_path: Path
) -> None:
    write(tmp_path, {"0001_a.sql": "select 1;", "0001_b.sql": "select 2;"})
    with pytest.raises(MigrationError, match="same number"):
        migrate(empty_database.pooled_dsn, tmp_path)


def test_two_runners_at_once_apply_every_migration_exactly_once(
    empty_database: Database, tmp_path: Path
) -> None:
    # without the lock one runner would hit "table already exists" and fail
    write(tmp_path, THREE)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(migrate, empty_database.pooled_dsn, tmp_path) for _ in range(2)]
        results = [future.result(timeout=30) for future in futures]
    assert sorted(name for applied in results for name in applied) == list(THREE)
    assert [v for v, _ in recorded(empty_database.direct_dsn)] == ["0001", "0002", "0003"]


def test_a_change_of_line_endings_is_not_an_edit(empty_database: Database, tmp_path: Path) -> None:
    # git on windows may check files out with crlf, that must not look like a rewritten migration
    path = tmp_path / "0001_first.sql"
    path.write_bytes(b"create table one (id int primary key);\ncreate table other (id int);\n")
    migrate(empty_database.pooled_dsn, tmp_path)
    path.write_bytes(b"create table one (id int primary key);\r\ncreate table other (id int);\r\n")
    assert migrate(empty_database.pooled_dsn, tmp_path) == []
