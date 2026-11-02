"""Offline installed entrypoint and real Core hook lifecycle qualification."""
import asyncio
import importlib
import importlib.metadata
import socket
from pathlib import Path

import pytest
import httpx

core = pytest.importorskip('amplifier_core')
from amplifier_core.validation import HookValidator

import amplifier_module_hooks_local_decisions as hook
from amplifier_module_provider_local_decisions import LocalDecisionsProvider
from amplifier_module_provider_local_decisions.catalog import catalog


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail('Offline lifecycle test attempted a network connection')
    monkeypatch.setattr(socket.socket, 'connect', blocked)
    monkeypatch.setattr(socket, 'getaddrinfo', blocked)


def account(identity, enabled=True):
    return {'_decision_account': {'binding_id': identity}, 'api_key': 'synthetic-unused',
            'decisions': {'enabled': enabled, 'backends': {
                identity: {'enabled': True, 'model': 'lightning-int8'}}}}


def test_real_same_wheel_hook_entrypoint_and_catalog():
    distribution = importlib.metadata.distribution('amplifier-module-provider-local-decisions')
    entry = next(item for item in distribution.entry_points
                 if item.group == 'amplifier.modules' and item.name == 'hooks-local-decisions')
    mounted = entry.load()
    assert mounted.__module__ == 'amplifier_module_hooks_local_decisions'
    assert importlib.import_module(mounted.__module__).__amplifier_module_type__ == 'hook'
    assert catalog()['runtime'] == {'kind': 'hook', 'module': 'hooks-local-decisions'}
    assert catalog()['provider_module'] == 'provider-local-decisions'


@pytest.mark.parametrize('enabled', [True, False])
async def test_core_validator_uses_real_hook_package(enabled):
    result = await HookValidator().validate(Path(hook.__file__).parent, config=account('validation', enabled))
    assert result.passed, result.summary()


@pytest.mark.parametrize('mode', ['ready', 'disabled', 'unavailable', 'cancelled', 'receipt_failure'])
async def test_actual_core_session_hook_mount_and_cleanup(monkeypatch, mode):
    # These two real modules are supplied by the qualification host, not runtime
    # dependencies of the decision-only wheel. No model or tool is executed.
    for package in ('amplifier_module_loop_streaming', 'amplifier_module_context_simple'):
        pytest.importorskip(package)
    rows = [account('first', mode != 'disabled'), account('second', mode != 'disabled')]
    if mode == 'unavailable':
        rows[1]['decisions']['backends']['bad'] = {'enabled': True, 'model': 'lightning-int8'}
    session = core.AmplifierSession({
        'session': {'orchestrator': 'loop-streaming', 'context': 'context-simple'},
        'hooks': [{'module': 'hooks-local-decisions', 'config': row} for row in rows],
    })
    created, receipts, failures, transports = [], [], [], []
    original = LocalDecisionsProvider.decision_backend
    original_init = LocalDecisionsProvider.__init__

    def account_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Exercise closure of a real account-owned transport without HTTP.
        def no_http(request):
            pytest.fail('Lifecycle mounting must not send inference requests')
        self._client = httpx.AsyncClient(transport=httpx.MockTransport(no_http))
        transports.append(self._client)

    monkeypatch.setattr(LocalDecisionsProvider, '__init__', account_init)

    def backend(self, config, coordinator=None):
        if mode == 'unavailable' and len(created) == 2:
            raise ValueError('synthetic factory failure after partial account mount')
        result = original(self, config, coordinator)
        created.append(result)
        return result

    monkeypatch.setattr(LocalDecisionsProvider, 'decision_backend', backend)

    async def report(binding, module, outcome, error):
        assert module == 'hooks-local-decisions'
        receipts.append((binding, outcome))
        if binding == 'second' and outcome == 'ready':
            if mode == 'cancelled':
                raise asyncio.CancelledError()
            if mode == 'receipt_failure':
                raise RuntimeError('synthetic host rejection')
        return outcome == 'unavailable'

    async def failed(event, data):
        failures.append(data)
        return core.HookResult()

    session.coordinator.register_capability('decision.account_lifecycle', report)
    session.coordinator.hooks.register('module:load_failed', failed, name='offline-qualification')
    try:
        if mode == 'cancelled':
            # Core2.0.2's Rust boundary wraps propagated cancellation.
            # It must still abort initialization, never permit fallback.
            with pytest.raises(RuntimeError, match='initialization failed: CancelledError'):
                await session.initialize()
        else:
            await session.initialize()
        registered = session.coordinator.get_capability('decision.backends') or {}
        assert not session.coordinator.get('providers')
        if mode == 'ready':
            assert set(registered) == {'first', 'second'}
            assert receipts == [('first', 'ready'), ('second', 'ready')] and not failures
        elif mode == 'disabled':
            assert not registered and not created and not transports and not failures
            assert receipts == [('first', 'disabled'), ('second', 'disabled')]
        else:
            assert set(registered) == {'first'}
            assert created[1].closed and not created[0].closed
            assert transports[1].is_closed and not transports[0].is_closed
            outcome = {'unavailable': 'unavailable', 'cancelled': 'cancelled',
                       'receipt_failure': 'unsafe'}[mode]
            assert receipts[-1] == ('second', outcome)
            assert bool(failures) is (mode == 'receipt_failure')
    finally:
        await session.cleanup()
    assert not session.coordinator.get_capability('decision.backends')
    assert all(backend.closed for backend in created)
    assert all(transport.is_closed for transport in transports)
