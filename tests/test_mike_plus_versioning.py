import sqlite3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / 'source'))
from kalden.core.mike.mike_plus_versioning import MikePlusVersioning


def test_baseline_changes_and_history(tmp_path):
    source = tmp_path / 'model.sqlite'
    with sqlite3.connect(source) as db:
        db.execute('CREATE TABLE msm_Node(MUID TEXT PRIMARY KEY, GroundLevel REAL, Description TEXT)')
        db.execute("INSERT INTO msm_Node VALUES ('N1', 100.0, 'original')")
        db.execute("INSERT INTO msm_Node VALUES ('N2', 200.0, NULL)")
    history = MikePlusVersioning(source, tables=('msm_Node',))
    assert history.check('baseline') == []
    with sqlite3.connect(source) as db:
        db.execute("UPDATE msm_Node SET GroundLevel=101.0 WHERE MUID='N1'")
        db.execute("DELETE FROM msm_Node WHERE MUID='N2'")
        db.execute("INSERT INTO msm_Node VALUES ('N3', 300.0, 'new')")
    changes = history.check('revised')
    assert {(c.operation, c.column) for c in changes} == {
        ('UPDATE', 'GroundLevel'), ('DELETE', None), ('INSERT', None)
    }
    assert len(history.changes(checkpoint_id=2)) == 3
    assert len(history.check('no changes')) == 0
    assert history.checkpoints()[0]['changes'] == 0


def test_wal_visibility(tmp_path):
    source = tmp_path / 'wal.sqlite'
    with sqlite3.connect(source) as db:
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('CREATE TABLE msm_Node(MUID TEXT PRIMARY KEY, GroundLevel REAL)')
        db.execute("INSERT INTO msm_Node VALUES('N1', 1.0)")
        db.commit()
        history = MikePlusVersioning(source, tables=('msm_Node',))
        history.check()
        db.execute("UPDATE msm_Node SET GroundLevel=2.0 WHERE MUID='N1'")
        db.commit()
        assert history.check()[0].new_value == 2.0


def test_rolls_back_when_table_missing(tmp_path):
    source = tmp_path / 'bad.sqlite'
    with sqlite3.connect(source) as db:
        db.execute('CREATE TABLE msm_Node(MUID TEXT PRIMARY KEY)')
    history = MikePlusVersioning(source, tables=('msm_Node', 'msm_Link'))
    try:
        history.check()
        assert False
    except ValueError:
        pass
    assert history.checkpoints() == []
