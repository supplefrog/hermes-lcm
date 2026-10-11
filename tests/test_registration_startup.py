"""Registration must survive slow storage without losing the context engine."""
import importlib.util
from pathlib import Path
import sqlite3
import sys

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def load_plugin():
    root = Path(__file__).resolve().parents[1]
    name = "hermes_lcm_registration_startup"
    spec = importlib.util.spec_from_file_location(name, root / "__init__.py", submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_native_registration_survives_database_writer(tmp_path, monkeypatch):
    plugins = pytest.importorskip("hermes_cli.plugins")
    from hermes_cli.plugins_manifest import PluginManifest
    # This plugin has its own tools.py; keep the actual host's tools package
    # ahead of the checkout when exercising the native registration seam.
    monkeypatch.syspath_prepend(str(Path(plugins.__file__).resolve().parents[1]))

    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("plugins:\n  enabled: []\n  load_timeout_seconds: 10\n", encoding="utf-8")
    db_path = home / "lcm.db"
    initial = LCMEngine(config=LCMConfig(database_path=str(db_path)), hermes_home=str(home))
    initial.shutdown_all_instances()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("LCM_DATABASE_PATH", str(db_path))
    manager = plugins.PluginManager(scope_key=str(home))
    root = Path(__file__).resolve().parents[1]
    manifest = PluginManifest(name="hermes-lcm", version="0.21.0", description="LCM startup regression", source="user", path=str(root))
    writer = sqlite3.connect(db_path)
    writer.execute("BEGIN IMMEDIATE")
    try:
        manager._load_plugin(manifest)
        loaded = manager._plugins["hermes-lcm"]
        assert loaded.enabled, loaded.error
        assert loaded.error is None
        assert manager._context_engine.name == "lcm"
    finally:
        writer.rollback()
        writer.close()
    try:
        clone = manager._context_engine.clone_for_agent()
        clone.on_session_start("registration-regression")
        clone.ingest([{"role": "user", "content": "registration recovery marker"}])
        assert clone._store.get_session_messages("registration-regression")[0]["content"] == "registration recovery marker"
        assert clone.get_tool_schemas()
    finally:
        manager.unload()


def test_unload_before_first_use_creates_no_storage(tmp_path, monkeypatch):
    module = load_plugin()
    db_path = tmp_path / "unused.db"
    monkeypatch.setenv("LCM_DATABASE_PATH", str(db_path))
    callbacks = []

    class Context:
        def register_context_engine(self, engine):
            self.engine = engine

        def on_unload(self, callback):
            callbacks.append(callback)

    ctx = Context()
    module.register(ctx)
    assert not db_path.exists()
    callbacks[0]()
    assert not db_path.exists()
    with pytest.raises(RuntimeError, match="unloading"):
        ctx.engine.clone_for_agent()


def test_failed_storage_activation_closes_partial_handles_and_can_retry(tmp_path, monkeypatch):
    from hermes_lcm import engine as engine_module

    real_store = engine_module.MessageStore
    real_dag = engine_module.SummaryDAG
    stores = []
    attempts = []

    def record_store(*args, **kwargs):
        store = real_store(*args, **kwargs)
        stores.append(store)
        return store

    def fail_first_dag(*args, **kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            raise RuntimeError("startup failure")
        return real_dag(*args, **kwargs)

    monkeypatch.setattr(engine_module, "MessageStore", record_store)
    monkeypatch.setattr(engine_module, "SummaryDAG", fail_first_dag)
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "retry.db")), defer_storage=True)
    with pytest.raises(RuntimeError, match="startup failure"):
        _ = engine._store
    assert len(stores) == 1
    assert stores[0]._conn is None
    try:
        engine.on_session_start("retry-session")
        engine.ingest([{"role": "user", "content": "retry marker"}])
        assert engine._store.get_session_messages("retry-session")[0]["content"] == "retry marker"
        assert len(stores) == 2
    finally:
        engine.shutdown_all_instances()


def test_first_activation_uses_requested_profile(tmp_path):
    original_home = tmp_path / "original"
    active_home = tmp_path / "active"
    engine = LCMEngine(config=LCMConfig(), hermes_home=str(original_home), defer_storage=True)
    try:
        engine.on_session_start("profile-session", hermes_home=str(active_home))
        assert not (original_home / "lcm.db").exists()
        assert engine._store.db_path == active_home / "lcm.db"
        engine.ingest([{"role": "user", "content": "active profile marker"}])
        assert engine._store.get_session_messages("profile-session")[0]["content"] == "active profile marker"
    finally:
        engine.shutdown_all_instances()
