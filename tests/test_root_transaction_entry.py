"""Independent review counterexamples over synthetic SQLite fixtures."""
import sqlite3
import pytest
from hermes_lcm.assertion_store import AssertionStoreError
from .test_repair_boundaries import source_fixture, InterruptAfterSQL, SyntheticCancellation


def operation(store, snapshot, candidates, relation, name):
    if name == 'publish':
        return store.publish_source(snapshot, candidates, relations=[relation])
    if name == 'invalidate':
        return store.invalidate_source(snapshot.store_id, reason='root-synthetic')
    return store.remove_version('assertions-v1')


@pytest.mark.parametrize('name', ['publish', 'invalidate', 'remove'])
@pytest.mark.parametrize('interruption', [KeyboardInterrupt, SystemExit, SyntheticCancellation])
def test_begin_interruption_rolls_back_owned_transaction_and_allows_retry(tmp_path, name, interruption):
    _, messages, store, snapshot, candidates, relation = source_fixture(tmp_path)
    try:
        if name != 'publish':
            store.publish_source(snapshot, candidates, relations=[relation])
        store._conn = InterruptAfterSQL(store._conn, 'BEGIN IMMEDIATE', interruption)
        with pytest.raises(interruption):
            operation(store, snapshot, candidates, relation, name)
        assert store.connection.in_transaction is False
        operation(store, snapshot, candidates, relation, name)
        assert store.connection.in_transaction is False
    finally:
        store.close()
        messages.close()


@pytest.mark.parametrize('name', ['publish', 'invalidate', 'remove'])
def test_rejected_nested_write_preserves_callers_open_transaction(tmp_path, name):
    _, messages, store, snapshot, candidates, relation = source_fixture(tmp_path)
    try:
        store.connection.execute('CREATE TABLE root_fixture(value TEXT)')
        store.connection.commit()
        store.connection.execute('BEGIN IMMEDIATE')
        store.connection.execute("INSERT INTO root_fixture VALUES ('pending caller work')")
        with pytest.raises((AssertionStoreError, sqlite3.OperationalError)):
            operation(store, snapshot, candidates, relation, name)
        assert store.connection.in_transaction is True
        assert store.connection.execute('SELECT value FROM root_fixture').fetchall()[0][0] == 'pending caller work'
        store.connection.rollback()
    finally:
        store.close()
        messages.close()


@pytest.mark.parametrize('name,statement', [('invalidate', 'UPDATE LCM_ASSERTION_SOURCES'), ('remove', 'DELETE FROM LCM_ASSERTION_SOURCES')])
@pytest.mark.parametrize('interruption', [KeyboardInterrupt, SystemExit, SyntheticCancellation])
def test_maintenance_rolls_back_immediately_without_relying_on_close(tmp_path, name, statement, interruption):
    _, messages, store, snapshot, candidates, relation = source_fixture(tmp_path)
    try:
        store.publish_source(snapshot, candidates, relations=[relation])
        store._conn = InterruptAfterSQL(store._conn, statement, interruption)
        with pytest.raises(interruption):
            operation(store, snapshot, candidates, relation, name)
        assert store.connection.in_transaction is False
        assert len(store.query_assertions()) == 2
        assert len(store.query_relations()) == 1
        operation(store, snapshot, candidates, relation, name)
        assert store.connection.in_transaction is False
    finally:
        store.close()
        messages.close()
