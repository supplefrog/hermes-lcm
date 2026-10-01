"""Synthetic regressions for explicit lifecycle identity and atomic publication."""
import sqlite3
import time
import threading

import pytest

from hermes_lcm.assertion_store import AssertionCandidate, AssertionRelationCandidate, AssertionStore
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.store import MessageStore


def make_engine(tmp_path):
    return LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / 'synthetic-lcm.db'),
        temporal_rollups_enabled=False,
        assertions_enabled=False,
        assertion_extraction_enabled=False,
        extraction_enabled=False,
    ), hermes_home=str(tmp_path / 'synthetic-home'))


@pytest.mark.parametrize('old_already_ingested', [False, True])
@pytest.mark.parametrize('foreground_rows', [0, 3])
def test_delayed_ordinary_end_preserves_ended_identity_and_foreground_cursor(
    tmp_path, old_already_ingested, foreground_rows,
):
    instance = make_engine(tmp_path)
    old = [{'role':'user','content':'Synthetic A question'}]
    ended = old + [{'role':'assistant','content':'Synthetic A final'}]
    foreground = [{'role':'user','content':f'Synthetic B message {i}'} for i in range(foreground_rows)]
    try:
        instance.on_session_start('ordinary-a', platform='cli', conversation_id='conversation-a', context_length=10000)
        if old_already_ingested:
            instance.ingest(old)
        instance.on_session_start('ordinary-b', platform='discord', conversation_id='conversation-b', context_length=10000)
        if foreground:
            instance.ingest(foreground)
        cursor_before = instance._ingest_cursor
        instance.on_session_end('ordinary-a', ended)
        rows_a = instance._store.get_range('ordinary-a', conversation_id='conversation-a')
        rows_b = instance._store.get_range('ordinary-b', conversation_id='conversation-b')
        assert [(r['role'],r['content'],r['source']) for r in rows_a] == [
            ('user','Synthetic A question','cli'),('assistant','Synthetic A final','cli')]
        assert [r['content'] for r in rows_b] == [m['content'] for m in foreground]
        assert (instance._session_id, instance._conversation_id, instance._ingest_cursor) == (
            'ordinary-b','conversation-b',cursor_before)
        ended_state = instance._lifecycle.get_by_conversation('conversation-a')
        foreground_state = instance._lifecycle.get_by_conversation('conversation-b')
        assert ended_state.last_finalized_session_id == 'ordinary-a'
        assert foreground_state.current_session_id == 'ordinary-b'
        # Repeated callbacks must not duplicate A or corrupt B's next ingest.
        instance.on_session_end('ordinary-a', ended)
        assert len(instance._store.get_range('ordinary-a', conversation_id='conversation-a')) == 2
        instance.ingest(foreground + [{'role':'assistant','content':'Synthetic B final'}])
        assert [r['content'] for r in instance._store.get_range('ordinary-b', conversation_id='conversation-b')] == [
            *[m['content'] for m in foreground], 'Synthetic B final']
    finally:
        instance.shutdown()


def test_unknown_ended_identity_does_not_write_or_finalize_current_session(tmp_path, caplog):
    instance = make_engine(tmp_path)
    try:
        instance.on_session_start('known', platform='cli', conversation_id='known-conversation', context_length=10000)
        instance.on_session_end('never-bound', [{'role':'assistant','content':'Unknown identity synthetic payload'}])
        assert instance._store.get_range('known') == []
        assert instance._store.get_range('never-bound') == []
        assert instance._lifecycle.get_by_conversation('known-conversation').current_session_id == 'known'
        assert 'identity' in caplog.text.lower() or 'off-current' in caplog.text.lower()
    finally:
        instance.shutdown()


def test_mismatched_delayed_snapshot_does_not_append_or_finalize_foreground(tmp_path):
    instance = make_engine(tmp_path)
    try:
        instance.on_session_start('ordinary-a', platform='cli', conversation_id='conversation-a', context_length=10000)
        instance.ingest([{'role':'user','content':'Recorded A opener'}])
        instance.on_session_start('ordinary-b', platform='cli', conversation_id='conversation-b', context_length=10000)
        instance.on_session_end('ordinary-a', [
            {'role':'user','content':'Different unverified opener'},
            {'role':'assistant','content':'Unverified late suffix'},
        ])
        assert [r['content'] for r in instance._store.get_range('ordinary-a')] == ['Recorded A opener']
        assert instance._store.get_range('ordinary-b') == []
        assert instance._lifecycle.get_by_conversation('conversation-b').current_session_id == 'ordinary-b'
    finally:
        instance.shutdown()


def test_delayed_end_remains_bounded_under_writer_lock_and_can_retry(tmp_path, caplog):
    instance = make_engine(tmp_path)
    ended = [{'role':'user','content':'Recorded A opener'}, {'role':'assistant','content':'A final'}]
    try:
        instance.on_session_start('ordinary-a', platform='cli', conversation_id='conversation-a', context_length=10000)
        instance.ingest(ended[:1])
        instance.on_session_start('ordinary-b', platform='cli', conversation_id='conversation-b', context_length=10000)
        with sqlite3.connect(tmp_path / 'synthetic-lcm.db') as blocker:
            blocker.execute('BEGIN IMMEDIATE')
            started = time.monotonic()
            instance.on_session_end('ordinary-a', ended)
            elapsed = time.monotonic() - started
            blocker.rollback()
        assert elapsed < 3.0
        assert 'SQLite lock' in caplog.text
        assert instance._store.get_range('ordinary-b') == []
        instance.on_session_end('ordinary-a', ended)
        assert [r['content'] for r in instance._store.get_range('ordinary-a')] == ['Recorded A opener','A final']
        assert instance._lifecycle.get_by_conversation('conversation-b').current_session_id == 'ordinary-b'
    finally:
        instance.shutdown()


class SyntheticCancellation(BaseException):
    pass


class InterruptAfterSQL:
    def __init__(self, connection, needle, interruption):
        self.connection = connection
        self.needle = needle
        self.interruption = interruption
        self.fired = False

    def execute(self, sql, *args, **kwargs):
        result = self.connection.execute(sql, *args, **kwargs)
        if not self.fired and self.needle in ' '.join(sql.split()).upper():
            self.fired = True
            raise self.interruption()
        return result

    def __getattr__(self, name):
        return getattr(self.connection, name)


def source_fixture(tmp_path):
    db_path = tmp_path / 'synthetic-assertions.db'
    messages = MessageStore(db_path)
    assertions = AssertionStore(db_path)
    content = 'Synthetic Alice likes tea and water.'
    store_id = messages.append('synthetic-session', {'role':'user','content':content}, source='cli')
    snapshot = assertions.snapshot_source(store_id)
    candidates = []
    for value in ['tea','water']:
        start = content.index(value)
        candidates.append(AssertionCandidate(start,start+len(value),'Alice','likes',object_value=value))
    ids = [assertions.assertion_id_for(snapshot,c) for c in candidates]
    relation = AssertionRelationCandidate(0,len(content),ids[0],'narrows',ids[1])
    return db_path,messages,assertions,snapshot,candidates,relation


@pytest.mark.parametrize('interruption', [KeyboardInterrupt, SystemExit, SyntheticCancellation])
@pytest.mark.parametrize('statement', [
    'INSERT INTO LCM_ASSERTION_SOURCES(',
    'INSERT INTO LCM_ASSERTIONS(',
    'INSERT INTO LCM_ASSERTION_RELATIONS(',
])
def test_interrupted_publication_remains_unpublished_after_close(tmp_path, interruption, statement):
    db_path,messages,assertions,snapshot,candidates,relation = source_fixture(tmp_path)
    assertions._conn = InterruptAfterSQL(assertions._conn, statement, interruption)
    try:
        with pytest.raises(interruption):
            assertions.publish_source(snapshot,candidates,relations=[relation])
        transaction_left_open = assertions.connection.in_transaction
    finally:
        assertions.close()
        messages.close()
    with sqlite3.connect(db_path) as reader:
        counts = {table:reader.execute('SELECT COUNT(*) FROM '+table).fetchone()[0]
                  for table in ['messages','lcm_assertion_sources','lcm_assertions','lcm_assertion_relations']}
    assert counts == {'messages':1,'lcm_assertion_sources':0,'lcm_assertions':0,'lcm_assertion_relations':0}
    assert transaction_left_open is False


@pytest.mark.parametrize('operation,statement', [
    ('invalidate','UPDATE LCM_ASSERTION_SOURCES'),
    ('remove','DELETE FROM LCM_ASSERTION_SOURCES'),
])
def test_interrupted_derived_maintenance_does_not_publish_on_close(tmp_path, operation, statement):
    db_path,messages,assertions,snapshot,candidates,relation = source_fixture(tmp_path)
    assertions.publish_source(snapshot,candidates,relations=[relation])
    assertions._conn = InterruptAfterSQL(assertions._conn, statement, KeyboardInterrupt)
    try:
        with pytest.raises(KeyboardInterrupt):
            if operation == 'invalidate':
                assertions.invalidate_source(snapshot.store_id, reason='synthetic-test')
            else:
                assertions.remove_version('assertions-v1')
    finally:
        assertions.close()
        messages.close()
    reopened = AssertionStore(db_path, read_only=True)
    try:
        assert len(reopened.query_assertions()) == 2
        assert len(reopened.query_relations()) == 1
    finally:
        reopened.close()


def test_close_rolls_back_unfinished_transaction_without_erasing_completed_publication(tmp_path):
    db_path,messages,assertions,snapshot,candidates,relation = source_fixture(tmp_path)
    assertions.publish_source(snapshot,candidates,relations=[relation])
    assertions.connection.execute('BEGIN IMMEDIATE')
    assertions.connection.execute('DELETE FROM lcm_assertion_sources')
    assertions.close()
    messages.close()
    reopened = AssertionStore(db_path, read_only=True)
    try:
        assert len(reopened.query_assertions()) == 2
        assert len(reopened.query_relations()) == 1
    finally:
        reopened.close()


@pytest.mark.parametrize('read_operation', ['assertions','relations','receipt','plan','snapshot'])
def test_reads_cannot_observe_publication_that_is_rolled_back(tmp_path, read_operation):
    _db_path,messages,assertions,snapshot,candidates,relation = source_fixture(tmp_path)
    inserted = threading.Event()
    release = threading.Event()
    reader_started = threading.Event()
    reader_at_lock = threading.Event()
    reader_finished = threading.Event()
    failures = []
    observed = []
    real = assertions._conn
    real_lock = assertions._write_lock

    class ObservedLock:
        def __enter__(self):
            if threading.current_thread() is read_thread:
                reader_at_lock.set()
            return real_lock.__enter__()
        def __exit__(self, *args):
            return real_lock.__exit__(*args)

    assertions._write_lock = ObservedLock()
    pause_statement = 'INSERT INTO LCM_ASSERTION_RELATIONS(' if read_operation == 'relations' else 'INSERT INTO LCM_ASSERTIONS('

    class PausedConnection:
        def execute(self, sql, *args, **kwargs):
            result = real.execute(sql, *args, **kwargs)
            if pause_statement in ' '.join(sql.split()).upper() and not inserted.is_set():
                inserted.set()
                if not release.wait(timeout=5):
                    raise RuntimeError('synthetic publication release timed out')
                raise KeyboardInterrupt()
            return result
        def __getattr__(self,name):
            return getattr(real,name)

    assertions._conn = PausedConnection()
    def writer():
        try:
            assertions.publish_source(snapshot,candidates,relations=[relation])
        except BaseException as exc:
            failures.append(type(exc).__name__)
    def reader():
        reader_started.set()
        try:
            if read_operation == 'assertions':
                observed.append(assertions.query_assertions())
            elif read_operation == 'relations':
                observed.append(assertions.query_relations())
            elif read_operation == 'receipt':
                observed.append(assertions.has_current_receipt(snapshot))
            elif read_operation == 'snapshot':
                observed.append(assertions.snapshot_source(snapshot.store_id))
            else:
                observed.append(assertions.plan_rebuild().pending_count)
        except BaseException as exc:
            failures.append(type(exc).__name__)
        finally:
            reader_finished.set()

    write_thread = threading.Thread(target=writer, daemon=True)
    read_thread = threading.Thread(target=reader, daemon=True)
    try:
        write_thread.start()
        assert inserted.wait(timeout=5)
        read_thread.start()
        assert reader_started.wait(timeout=5)
        assert reader_at_lock.wait(timeout=5)
        assert not reader_finished.wait(timeout=0.15)
    finally:
        release.set()
        write_thread.join(timeout=5)
        if read_thread.ident is not None:
            read_thread.join(timeout=5)
        assert not write_thread.is_alive() and not read_thread.is_alive(), 'Synthetic workers did not settle; skip lock-taking cleanup'
        assertions.close()
        messages.close()
    assert not write_thread.is_alive() and not read_thread.is_alive()
    assert failures == ['KeyboardInterrupt']
    expected = {'assertions':[[]], 'relations':[[]], 'receipt':[False], 'plan':[1], 'snapshot':[snapshot]}
    assert observed == expected[read_operation]


@pytest.mark.parametrize('interruption', [KeyboardInterrupt, SystemExit])
def test_interruption_after_commit_preserves_complete_publication_and_original_signal(tmp_path, interruption):
    db_path,messages,assertions,snapshot,candidates,relation = source_fixture(tmp_path)
    assertions._conn = InterruptAfterSQL(assertions._conn, 'COMMIT', interruption)
    try:
        with pytest.raises(interruption):
            assertions.publish_source(snapshot,candidates,relations=[relation])
    finally:
        assertions.close()
        messages.close()
    reopened = AssertionStore(db_path, read_only=True)
    try:
        assert len(reopened.query_assertions()) == 2
        assert len(reopened.query_relations()) == 1
    finally:
        reopened.close()
