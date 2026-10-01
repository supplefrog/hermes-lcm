"""Final-review session boundaries over actual native stores and synthetic data."""
from dataclasses import asdict
import sqlite3
import time

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.store import MessageStore


class SyntheticCancel(BaseException):
    pass


def engine(tmp_path, *, ignored=False):
    instance=LCMEngine(config=LCMConfig(database_path=str(tmp_path/'engine.db'),
        temporal_rollups_enabled=False, assertions_enabled=False,
        assertion_extraction_enabled=False, extraction_enabled=False,
        ignore_message_patterns=['^IGNORE_SYNTHETIC$'] if ignored else []),
        hermes_home=str(tmp_path/'home'))
    if ignored:
        # The inspected Hermes venv lacks the optional regex wheel. Exercise
        # the real native filtering/ingest paths with a constant-time literal
        # matcher implementing the same timeout-aware compiled-pattern API.
        class LiteralPattern:
            pattern='^IGNORE_SYNTHETIC$'
            def search(self,text,*,timeout):
                assert timeout > 0
                return text=='IGNORE_SYNTHETIC'
        instance._compiled_ignore_message_patterns=[LiteralPattern()]
    return instance


def bind_pair(obj, *, shared=False, rows=None):
    original=rows if rows is not None else [{'role':'user','content':'A opener'}]
    obj.on_session_start('ended-a',platform='cli',conversation_id='conversation-a',context_length=10000)
    if original:
        obj.ingest(original)
    obj.on_session_start('active-b',platform='discord',conversation_id='conversation-a' if shared else 'conversation-b',context_length=10000)
    return original


def binding(obj):
    return (obj._session_id,obj._conversation_id,obj._ingest_cursor,obj._last_compacted_store_id)


def contents(obj):
    return [r['content'] for r in obj._store.get_range('ended-a',conversation_id='conversation-a')]


def test_prefix_reads_use_short_busy_timeout_and_restore(tmp_path,monkeypatch):
    obj=engine(tmp_path)
    try:
        original=bind_pair(obj)
        obj._store._conn.execute('PRAGMA busy_timeout=1500')
        native=obj._store.get_range
        observed=[]
        def read(*args,**kwargs):
            observed.append(obj._store._conn.execute('PRAGMA busy_timeout').fetchone()[0])
            return native(*args,**kwargs)
        monkeypatch.setattr(obj._store,'get_range',read)
        obj.on_session_end('ended-a',original+[{'role':'assistant','content':'A final'}])
        assert observed and set(observed)=={50}
        assert obj._store._conn.execute('PRAGMA busy_timeout').fetchone()[0]==1500
    finally:
        obj.shutdown()


def test_actual_exclusive_read_lock_warns_quickly_and_retry_works(tmp_path,caplog):
    obj=engine(tmp_path)
    old_store=None
    try:
        original=bind_pair(obj)
        # A separate native MessageStore permits rollback journal mode without
        # interference from the engine's other WAL connections to engine.db.
        old_store=obj._store
        obj._store=MessageStore(tmp_path/'read-lock.db')
        obj._store.append_batch('ended-a',original,source='cli',conversation_id='conversation-a')
        assert obj._store._conn.execute('PRAGMA journal_mode=DELETE').fetchone()[0]=='delete'
        obj._store._conn.execute('PRAGMA busy_timeout=1500')
        ended=original+[{'role':'assistant','content':'A final'}]
        before=binding(obj)
        with sqlite3.connect(tmp_path/'read-lock.db') as blocker:
            blocker.execute('BEGIN EXCLUSIVE')
            start=time.monotonic()
            obj.on_session_end('ended-a',ended)
            elapsed=time.monotonic()-start
            blocker.rollback()
        assert elapsed < 0.75, f'Prefix read waited {elapsed:.3f}s with normal timeout'
        assert 'SQLite lock' in caplog.text
        assert obj._store._conn.execute('PRAGMA busy_timeout').fetchone()[0]==1500
        assert binding(obj)==before
        assert contents(obj)==['A opener']
        obj.on_session_end('ended-a',ended)
        assert contents(obj)==['A opener','A final']
    finally:
        obj.shutdown()
        if old_store:old_store.close()


@pytest.mark.parametrize('ignored_position',[0,1,2])
def test_filtered_host_prefix_repeated_end_does_not_duplicate(tmp_path,ignored_position):
    obj=engine(tmp_path,ignored=True)
    u={'role':'user','content':'A opener'}
    x={'role':'assistant','content':'IGNORE_SYNTHETIC'}
    f={'role':'assistant','content':'A final'}
    end=[u,f]
    end.insert(ignored_position,x)
    original=end[:end.index(f)]
    try:
        bind_pair(obj,rows=original)
        before=binding(obj)
        obj.on_session_end('ended-a',end)
        obj.on_session_end('ended-a',end)
        assert contents(obj)==['A opener','A final']
        assert binding(obj)==before
    finally:
        obj.shutdown()


@pytest.mark.parametrize('point',['prefix','append','finalize'])
@pytest.mark.parametrize('locked',[False,True])
def test_database_errors_are_warned_only_for_locks(tmp_path,monkeypatch,caplog,point,locked):
    obj=engine(tmp_path)
    try:
        original=bind_pair(obj)
        before=binding(obj)
        failure=sqlite3.OperationalError('database is locked') if locked else sqlite3.DatabaseError('synthetic database failure')
        owner,name={'prefix':(obj._store,'get_range'),'append':(obj._store,'_append_protected_batch'),
                    'finalize':(obj._lifecycle,'finalize_session')}[point]
        native=getattr(owner,name)
        def fail(*args,**kwargs):raise failure
        monkeypatch.setattr(owner,name,fail)
        ended=original+[{'role':'assistant','content':'A final'}]
        if locked:
            obj.on_session_end('ended-a',ended)
            assert 'SQLite lock' in caplog.text
        else:
            with pytest.raises(sqlite3.DatabaseError,match='synthetic database failure'):
                obj.on_session_end('ended-a',ended)
        monkeypatch.setattr(owner,name,native)
        assert binding(obj)==before
        obj.on_session_end('ended-a',ended)
        assert contents(obj)==['A opener','A final']
    finally:
        obj.shutdown()


@pytest.mark.parametrize('stored',[False,True])
def test_known_empty_delayed_end_finalizes_without_changing_rows(tmp_path,stored):
    obj=engine(tmp_path)
    try:
        bind_pair(obj,rows=[{'role':'user','content':'A opener'}] if stored else [])
        before=binding(obj)
        rows=contents(obj)
        obj.on_session_end('ended-a',[])
        state=obj._lifecycle.get_by_conversation('conversation-a')
        assert state.last_finalized_session_id=='ended-a'
        assert state.current_session_id is None
        assert contents(obj)==rows
        assert binding(obj)==before
    finally:obj.shutdown()


@pytest.mark.parametrize('newer_finished',[False,True])
def test_late_shared_conversation_end_preserves_newer_lifecycle_and_frontier(tmp_path,newer_finished):
    obj=engine(tmp_path)
    try:
        original=bind_pair(obj,shared=True)
        obj._lifecycle.advance_frontier('conversation-a','active-b',99)
        if newer_finished:
            obj._lifecycle.finalize_session('conversation-a','active-b',frontier_store_id=99)
        before_state=asdict(obj._lifecycle.get_by_conversation('conversation-a'))
        before=binding(obj)
        obj.on_session_end('ended-a',original+[{'role':'assistant','content':'A final'}])
        assert asdict(obj._lifecycle.get_by_conversation('conversation-a'))==before_state
        assert binding(obj)==before
        assert contents(obj)==['A opener','A final']
    finally:obj.shutdown()


class InterruptConnection:
    def __init__(self,connection,needle,signal):
        self.connection,self.needle,self.signal=connection,needle,signal
        self.fired=False
    def execute(self,sql,*args,**kwargs):
        value=self.connection.execute(sql,*args,**kwargs)
        if not self.fired and self.needle in ' '.join(sql.split()).upper():
            self.fired=True
            raise self.signal()
        return value
    def __enter__(self):
        self.connection.__enter__()
        return self
    def __exit__(self,*args):return self.connection.__exit__(*args)
    def __getattr__(self,name):return getattr(self.connection,name)


@pytest.mark.parametrize('point',['prefix','append','finalize'])
@pytest.mark.parametrize('signal',[KeyboardInterrupt,SystemExit,SyntheticCancel])
def test_cancel_at_native_sql_boundaries_preserves_binding_and_retry(tmp_path,caplog,point,signal):
    obj=engine(tmp_path)
    try:
        original=bind_pair(obj)
        before=binding(obj)
        store_conn=obj._store._conn
        lifecycle_conn=obj._lifecycle._conn
        store_conn.execute('PRAGMA busy_timeout=1500')
        lifecycle_conn.execute('PRAGMA busy_timeout=1800')
        if point=='finalize':
            obj._lifecycle._conn=InterruptConnection(lifecycle_conn,'UPDATE LCM_LIFECYCLE_STATE',signal)
        else:
            obj._store._conn=InterruptConnection(store_conn,'SELECT' if point=='prefix' else 'INSERT INTO MESSAGES',signal)
        ended=original+[{'role':'assistant','content':'A final 1'},{'role':'assistant','content':'A final 2'}]
        if signal is KeyboardInterrupt:
            try:obj.on_session_end('ended-a',ended)
            except KeyboardInterrupt:pytest.fail('Session-end prefix interruption escaped best-effort policy')
            assert 'interrupt' in caplog.text.lower()
        else:
            with pytest.raises(signal):obj.on_session_end('ended-a',ended)
        assert binding(obj)==before
        assert store_conn.execute('PRAGMA busy_timeout').fetchone()[0]==1500
        assert lifecycle_conn.execute('PRAGMA busy_timeout').fetchone()[0]==1800
        assert store_conn.in_transaction is False
        # Native MessageStore's connection context rolls back an interrupted
        # append batch. Lifecycle uses autocommit; a signal after its UPDATE
        # may leave the complete finalization committed, which retry preserves.
        assert contents(obj)==(['A opener','A final 1','A final 2'] if point=='finalize' else ['A opener'])
        obj.on_session_end('ended-a',ended)
        assert contents(obj)==['A opener','A final 1','A final 2']
        assert binding(obj)==before
    finally:obj.shutdown()
