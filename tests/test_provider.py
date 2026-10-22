import asyncio
import copy
import json
from types import SimpleNamespace
import httpx
import pytest
from amplifier_decisions import DecisionRequest, Question
from amplifier_module_provider_local_decisions import LocalDecisionsProvider, mount
from amplifier_module_provider_local_decisions.catalog import catalog, validate, MODELS, LOCAL_COST

REQUEST = DecisionRequest({'page': 'Synthetic'}, (Question('next', 'choice', 'Choose', {'save': 'Save', 'cancel': 'Cancel'}),), allow_external_state=True)
PAYLOAD = {'model': 'lightning-int8', 'checkpoint': MODELS['lightning-int8']['checkpoint'], 'precision': 'int8',
    'usage': {'input_tokens': 5, 'output_tokens': 0},
    'answers': {'next': {'type': 'choice', 'choice': 'save', 'probabilities': {'save': .9, 'cancel': .1}, 'confidence': .9}}}


async def test_real_protocol_receipts_and_configured_auth():
    seen = []
    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=PAYLOAD, headers={'x-request-id': 'owned'})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = LocalDecisionsProvider(api_key='private-token', config={'base_url': 'http://127.0.0.1:18990/v1/systemone'}, client=client)
        result = await provider.decision_backend({'model': 'lightning-int8'}).decide(REQUEST)
        assert result.answers['next'].value == 'save'
        assert result.checkpoint == MODELS['lightning-int8']['checkpoint'] and result.cost == LOCAL_COST
        assert result.request_id == 'owned' and result.usage == {'input_tokens': 5, 'output_tokens': 0}
        assert seen[0].headers['authorization'] == 'Bearer private-token'
        assert json.loads(seen[0].content)['model'] == 'lightning-int8'
        await provider.close()
        assert not client.is_closed and 'private-token' not in repr(result)


@pytest.mark.parametrize('field,value', [('model', 'other'), ('checkpoint', 'other'), ('precision', 'nf4')])
async def test_service_cannot_silently_substitute_model(field, value):
    payload = copy.deepcopy(PAYLOAD); payload[field] = value
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))) as client:
        result = await LocalDecisionsProvider(api_key='x', client=client).decision_backend({'model': 'lightning-int8'}).decide(REQUEST)
        assert result.answers['next'].reason == 'model_identity_mismatch'


@pytest.mark.parametrize('status', [302, 400, 401, 413, 422, 429, 500, 503])
async def test_failures_redacted_and_not_retried(status):
    seen = []
    def handler(request):
        seen.append(request); return httpx.Response(status, text='PRIVATE-EVIDENCE', headers={'location': 'https://other.test'})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await LocalDecisionsProvider(api_key='x', client=client).decision_backend({'model': 'lightning-int8'}).decide(REQUEST)
        assert len(seen) == 1 and 'PRIVATE' not in repr(result)
        assert result.answers['next'].status in {'invalid', 'unavailable'}


async def test_disabled_and_no_transfer_consent_are_inert():
    coordinator = SimpleNamespace(caps={})
    coordinator.get_capability = lambda key: coordinator.caps.get(key)
    coordinator.register_capability = lambda key, value: coordinator.caps.__setitem__(key, value)
    assert await mount(coordinator, {'api_key': 'x'}) is None
    assert not coordinator.caps
    provider = LocalDecisionsProvider(api_key='x')
    result = await provider.decision_backend({'model': 'lightning-int8'}).decide(DecisionRequest('x', REQUEST.questions))
    assert result.answers['next'].status == 'invalid' and provider._client is None
    cleanup = await mount(coordinator, {'decisions': {'enabled': True, 'backends': {
        'first': {'enabled': True, 'model': 'lightning-int8'}, 'second': {'enabled': True, 'model': 'd3-mini-int8'}}}})
    assert set(coordinator.caps['decision.backends']) == {'first', 'second'}
    await cleanup()
    assert coordinator.caps['decision.backends'] == {}


async def test_cancellation_and_timeout():
    async def slow(request): await asyncio.sleep(10)
    async with httpx.AsyncClient(transport=httpx.MockTransport(slow)) as client:
        backend = LocalDecisionsProvider(api_key='x', client=client).decision_backend({'model': 'd3-mini-int8'})
        result = await backend.decide(DecisionRequest('x', REQUEST.questions, timeout_seconds=.01, allow_external_state=True))
        assert result.answers['next'].reason == 'timeout'
        task = asyncio.create_task(backend.decide(REQUEST)); await asyncio.sleep(.01); task.cancel()
        with pytest.raises(asyncio.CancelledError): await task


@pytest.mark.parametrize('endpoint', ['http://10.0.0.1/v1/systemone', 'http://localhost.test/', 'https://user:secret@test/', 'https://test/?key=secret', 'https://test/#fragment'])
def test_endpoint_boundary(endpoint):
    with pytest.raises(ValueError): validate({'model': 'lightning-int8', 'endpoint': endpoint})


def test_discovery_is_offline_and_lightweight():
    import subprocess, sys
    subprocess.run([sys.executable, '-c', 'from amplifier_module_provider_local_decisions.catalog import catalog; import sys; catalog(); assert "torch" not in sys.modules'], check=True)
    assert catalog()['backends'][0]['cost']['api_cost_usd'] == 0
    for endpoint in ('http://127.0.0.1:18990/v1/systemone', 'http://[::1]:18990/v1/systemone', 'https://local.test/v1/systemone'):
        assert validate({'model': 'd3-mini-int8', 'endpoint': endpoint})['endpoint'] == endpoint
    with pytest.raises(ValueError): validate({'model': 'lightning-int8', 'api_key': 'no'})
