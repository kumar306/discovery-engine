import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import psycopg

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "deploy" / "migrations"

_FILE_NAME = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")

# every runner asks for the same advisory lock number
# if two runners run at once the second one waits for the lock, then rereads what is recorded
# and applies only what is not there yet, so nothing is ever applied twice
_LOCK_KEY = 7_302_024


class MigrationError(Exception):
    """A migration cannot be applied or the history on disk does not match the database."""


@dataclass(frozen=True)
class Migration:
    version: str
    name: str
    sql: str
    checksum: str


def load_migrations(directory: Path) -> list[Migration]:
    migrations: list[Migration] = []
    for path in sorted(directory.glob("*.sql")):
        match = _FILE_NAME.match(path.name)
        if match is None:
            raise MigrationError(f"{path.name} is not named like 0001_some_name.sql")
        text = path.read_text(encoding="utf-8")
        checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
        migrations.append(Migration(match.group(1), path.name, text, checksum))
    versions = [migration.version for migration in migrations]
    if len(versions) != len(set(versions)):
        raise MigrationError("two migration files share the same number")
    return migrations


def migrate(dsn: str, directory: Path = MIGRATIONS_DIR) -> list[str]:
    migrations = load_migrations(directory)
    applied: list[str] = []
    with psycopg.connect(dsn, autocommit=True, prepare_threshold=None) as conn:
        # same lock, checks the history on disk against the database before anything is applied
        with conn.transaction():
            _lock_and_prepare(conn)
            _check_history(conn, migrations)
        for migration in migrations:
            # autocommit is on, so nothing is wrapped for us and this block is the transaction
            # it holds the lock, the migration sql and its row in schema_migrations
            # they commit together or roll back together, one transaction per migration
            with conn.transaction():
                _lock_and_prepare(conn)
                if migration.version in _recorded(conn):
                    continue
                try:
                    conn.execute(migration.sql)
                    conn.execute(
                        "insert into schema_migrations (version, name, checksum) "
                        "values (%s, %s, %s)",
                        (migration.version, migration.name, migration.checksum),
                    )
                except psycopg.Error as error:
                    raise MigrationError(f"{migration.name} failed: {error}") from error
                applied.append(migration.name)
    return applied


def _lock_and_prepare(conn: psycopg.Connection) -> None:
    # this advisory lock is transaction scoped not session scoped
    # so it releases on commit or rollback, no lock is left behind on the connection
    # it also guards creating the history table, only one runner at a time gets past this line
    conn.execute("select pg_advisory_xact_lock(%s)", (_LOCK_KEY,))
    conn.execute(
        "create table if not exists schema_migrations ("
        "version text primary key, name text not null, checksum text not null, "
        "applied_at timestamptz not null default now())"
    )


def _recorded(conn: psycopg.Connection) -> dict[str, str]:
    rows = conn.execute("select version, checksum from schema_migrations").fetchall()
    return {version: checksum for version, checksum in rows}


def _check_history(conn: psycopg.Connection, migrations: list[Migration]) -> None:
    on_disk = {migration.version: migration for migration in migrations}
    for version, checksum in _recorded(conn).items():
        migration = on_disk.get(version)
        if migration is None:
            raise MigrationError(f"migration {version} was applied but its file is gone")
        if migration.checksum != checksum:
            raise MigrationError(f"{migration.name} was edited after it was applied")
