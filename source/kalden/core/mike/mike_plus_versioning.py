"""Non-invasive, user-triggered change history for a MIKE+ SQLite database.

The MIKE+ database is opened read-only and backed up to a temporary snapshot.
All history is stored in a separate SQLite database. No MIKE+ schema writes.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import tempfile
from typing import Iterator, Sequence


DEFAULT_TABLES = ("msm_Node", "msm_Link")


def _q(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _value(value: object) -> list:
    """Typed encoding keeps SQLite NULL, strings, ints, floats and BLOBs distinct."""
    if value is None:
        return ["null", None]
    if isinstance(value, bytes):
        return ["blob", base64.b64encode(value).decode("ascii")]
    if isinstance(value, int):
        return ["integer", value]
    if isinstance(value, float):
        return ["real", repr(value)]
    if isinstance(value, str):
        return ["text", value]
    raise TypeError(f"Unsupported SQLite value type: {type(value)!r}")


def _decode(value: list) -> object:
    kind, payload = value
    if kind == "null":
        return None
    if kind == "blob":
        return base64.b64decode(payload)
    if kind == "real":
        return float(payload)
    return payload


def _serialize(row: dict[str, object]) -> str:
    return json.dumps({k: _value(v) for k, v in row.items()},
                      sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _deserialize(raw: str) -> dict[str, object]:
    return {k: _decode(v) for k, v in json.loads(raw).items()}


def _identity(row: dict[str, object], keys: Sequence[str]) -> str:
    return json.dumps([_value(row[k]) for k in keys],
                      separators=(",", ":"), ensure_ascii=False)


@contextmanager
def _source_snapshot(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Use SQLite backup API to include committed WAL contents consistently.

    Does not copy only the .sqlite file: WAL-mode databases may have pending
    committed pages in an adjacent -wal file. The source is opened mode=ro.
    """
    with tempfile.TemporaryDirectory(prefix="kalden-versioning-") as folder:
        target = Path(folder) / "snapshot.sqlite"
        uri = db_path.resolve().as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=10) as source:
            with sqlite3.connect(target) as backup:
                source.backup(backup, pages=256, sleep=0.05)
        with sqlite3.connect(f"{target.as_uri()}?mode=ro", uri=True) as snapshot:
            snapshot.row_factory = sqlite3.Row
            yield snapshot


@dataclass(frozen=True)
class Change:
    checkpoint_id: int
    table: str
    row_key: str
    operation: str  # INSERT, DELETE, UPDATE
    column: str | None
    old_value: object
    new_value: object


class MikePlusVersioning:
    """Checkpoint-based field-level audit, separate from the MIKE+ database.

    Parameters
    ----------
    db_path : path to an actual SQLite file (resolve .mupp via MPlusModel first)
    history_path : external history SQLite; default is .kalden/<name>.history.sqlite
    tables : exact table names to audit. Default is intentionally conservative.
    """

    def __init__(self, db_path: str | Path, *, history_path: str | Path | None = None,
                 tables: Sequence[str] = DEFAULT_TABLES):
        self.db_path = Path(db_path).resolve()
        if not self.db_path.is_file():
            raise FileNotFoundError(self.db_path)
        self.history_path = Path(history_path).resolve() if history_path else (
            self.db_path.parent / ".kalden" / f"{self.db_path.stem}.history.sqlite"
        )
        if self.history_path == self.db_path:
            raise ValueError("History database must not be the MIKE+ source database")
        self.tables = tuple(dict.fromkeys(tables))
        if not self.tables:
            raise ValueError("At least one table must be selected")

    def _store(self) -> sqlite3.Connection:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.history_path, timeout=30)
        db.execute("PRAGMA foreign_keys=ON")
        db.executescript("""
            CREATE TABLE IF NOT EXISTS configuration (
                setting TEXT PRIMARY KEY, value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                checked_at TEXT NOT NULL,
                note TEXT NOT NULL DEFAULT '',
                changes INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS row_state (
                table_name TEXT NOT NULL,
                row_key TEXT NOT NULL,
                digest TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY(table_name, row_key)
            );
            CREATE TABLE IF NOT EXISTS changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                checkpoint_id INTEGER NOT NULL REFERENCES checkpoints(id),
                table_name TEXT NOT NULL,
                row_key TEXT NOT NULL,
                operation TEXT NOT NULL,
                column_name TEXT,
                old_value TEXT,
                new_value TEXT
            );
            CREATE INDEX IF NOT EXISTS ix_changes_asset
                ON changes(table_name, row_key);
            CREATE INDEX IF NOT EXISTS ix_changes_checkpoint
                ON changes(checkpoint_id);
        """)
        return db

    @staticmethod
    def _key_columns(source: sqlite3.Connection, table: str) -> tuple[str, ...]:
        info = source.execute(f"PRAGMA table_info({_q(table)})").fetchall()
        if not info:
            raise ValueError(f"Table {table!r} not present in database")
        pk = sorted(((r[5], r[1]) for r in info if r[5]), key=lambda x: x[0])
        if pk:
            return tuple(name for _, name in pk)
        columns = {r[1] for r in info}
        if "MUID" in columns:
            return ("MUID",)  # duplicate identifiers checked below
        raise ValueError(f"No primary key / MUID in {table!r}; specify another table")

    def check(self, note: str = "") -> list[Change]:
        """Compare against preceding checkpoint and atomically update history.

        First check creates baseline state without change events. A failed check
        rolls back the history transaction; no partial checkpoint is committed.
        """
        output: list[Change] = []
        with _source_snapshot(self.db_path) as source, self._store() as store:
            try:
                store.execute("BEGIN IMMEDIATE")
                config = json.dumps(self.tables)
                previous = store.execute(
                    "SELECT value FROM configuration WHERE setting='tables'"
                ).fetchone()
                if previous is not None and previous[0] != config:
                    raise ValueError("Tracked tables changed; use a new history_path")
                if previous is None:
                    store.execute("INSERT INTO configuration VALUES ('tables', ?)", (config,))
                first = store.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == 0
                cursor = store.execute(
                    "INSERT INTO checkpoints(checked_at, note, changes) VALUES (?, ?, 0)",
                    (datetime.now(timezone.utc).isoformat(), note),
                )
                checkpoint_id = cursor.lastrowid
                for table in self.tables:
                    keys = self._key_columns(source, table)
                    stored = {
                        r[0]: (r[1], r[2]) for r in store.execute(
                            "SELECT row_key, digest, payload FROM row_state WHERE table_name=?",
                            (table,),
                        )
                    }
                    seen: set[str] = set()
                    for fetched in source.execute(f"SELECT * FROM {_q(table)}"):
                        row = dict(fetched)
                        key = _identity(row, keys)
                        if key in seen:
                            raise ValueError(f"Duplicate key in {table}: {key}")
                        seen.add(key)
                        payload = _serialize(row)
                        digest = sha256(payload.encode("utf-8")).hexdigest()
                        previous_state = stored.get(key)
                        if previous_state is None:
                            if not first:
                                output.append(Change(checkpoint_id, table, key, "INSERT",
                                                     None, None, row))
                        elif previous_state[0] != digest:
                            old = _deserialize(previous_state[1])
                            for column in sorted(set(old) | set(row)):
                                if old.get(column) != row.get(column) or (
                                    (column in old) != (column in row)
                                ):
                                    output.append(Change(checkpoint_id, table, key,
                                                         "UPDATE", column, old.get(column),
                                                         row.get(column)))
                        store.execute("""INSERT INTO row_state(table_name,row_key,digest,payload)
                                         VALUES (?,?,?,?)
                                         ON CONFLICT(table_name,row_key) DO UPDATE SET
                                         digest=excluded.digest,payload=excluded.payload""",
                                      (table, key, digest, payload))
                    for key in stored.keys() - seen:
                        if not first:
                            output.append(Change(checkpoint_id, table, key, "DELETE",
                                                 None, _deserialize(stored[key][1]), None))
                        store.execute("DELETE FROM row_state WHERE table_name=? AND row_key=?",
                                      (table, key))
                for change in output:
                    store.execute("""INSERT INTO changes
                        (checkpoint_id,table_name,row_key,operation,column_name,old_value,new_value)
                        VALUES (?,?,?,?,?,?,?)""",
                        (change.checkpoint_id, change.table, change.row_key,
                         change.operation, change.column,
                         json.dumps(_value(change.old_value)) if change.column else (
                             _serialize(change.old_value) if change.old_value is not None else None),
                         json.dumps(_value(change.new_value)) if change.column else (
                             _serialize(change.new_value) if change.new_value is not None else None))
                    )
                store.execute("UPDATE checkpoints SET changes=? WHERE id=?",
                              (len(output), checkpoint_id))
                store.commit()
            except Exception:
                store.rollback()
                raise
        return output

    def checkpoints(self) -> list[dict]:
        with self._store() as store:
            store.row_factory = sqlite3.Row
            return [dict(r) for r in store.execute(
                "SELECT * FROM checkpoints ORDER BY id DESC")]

    def changes(self, *, checkpoint_id: int | None = None,
                table: str | None = None, row_key: str | None = None) -> list[Change]:
        filters, params = [], []
        for column, value in (("checkpoint_id", checkpoint_id),
                              ("table_name", table), ("row_key", row_key)):
            if value is not None:
                filters.append(f"{column}=?")
                params.append(value)
        query = "SELECT checkpoint_id,table_name,row_key,operation,column_name,old_value,new_value FROM changes"
        if filters:
            query += " WHERE " + " AND ".join(filters)
        query += " ORDER BY id"
        with self._store() as store:
            result = []
            for cp, t, k, op, col, old, new in store.execute(query, params):
                old_value = (_decode(json.loads(old)) if col else _deserialize(old)) if old is not None else None
                new_value = (_decode(json.loads(new)) if col else _deserialize(new)) if new is not None else None
                result.append(Change(cp, t, k, op, col, old_value, new_value))
            return result
