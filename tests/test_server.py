import asyncio
import threading
from types import SimpleNamespace
import httpx
import pytest
from amplifier_module_provider_local_decisions.runtime.server import create_app

TOKEN = 'synthetic-service-token-not-a-secret-12345'
PAYLOAD = {'model': 'fixture', 'state': 'Choose', 'questions': {'next': {'type': 'choice', 'instructions': 'Save', 'criteria': {'a': 'Save', 'b': 'Cancel'}}}}
ANSWER = {'model': 'fixture', 'answers': {'next': {'type': 'choice', 'choice': 'a', 'probabilities': {'a': .9, 'b': .1}, 'confidence': .9}}}


async def test_authenticated_bounded_typed_requests():
    seen = []
    def infer(payload): seen.append(payload); return ANSWER
    app = create_app({'fixture': SimpleNamespace(decide=infer)}, TOKEN)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://owned') as client:
            assert (await client.post('/v1/systemone', json=PAYLOAD)).status_code == 401
            client.headers['Authorization'] = 'Bearer ' + TOKEN
            assert (await client.post('/v1/systemone', json={**PAYLOAD, 'images': ['https://private/']})).status_code == 422
            assert (await client.post('/v1/systemone', json={**PAYLOAD, 'state': 'a' * 70000})).status_code == 422
            assert (await client.post('/v1/systemone', json={**PAYLOAD, 'state': 'a' * 140000})).status_code == 413
            assert (await client.post('/v1/systemone', json={**PAYLOAD, 'model': 'unloaded'})).status_code == 422
            response = await client.post('/v1/systemone', json=PAYLOAD)
            assert response.status_code == 200 and response.json()['cost']['api_cost_usd'] == 0
            assert response.headers['x-request-id'] and seen == [PAYLOAD]


async def test_timeout_retains_busy_gate_until_inference_finishes():
    started, release = threading.Event(), threading.Event()
    def infer(payload): started.set(); release.wait(2); return ANSWER
    app = create_app({'fixture': SimpleNamespace(decide=infer)}, TOKEN, inference_timeout=.02)
    try:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://owned', headers={'Authorization': 'Bearer ' + TOKEN}) as client:
                assert (await client.post('/v1/systemone', json=PAYLOAD)).status_code == 504
                assert started.is_set()
                assert (await client.post('/v1/systemone', json=PAYLOAD)).status_code == 503
                release.set()
                await asyncio.sleep(.02)
                assert (await client.post('/v1/systemone', json=PAYLOAD)).status_code == 200
    finally: release.set()


async def test_invalid_model_output_never_becomes_answer():
    app = create_app({'fixture': SimpleNamespace(decide=lambda _: {'answers': {}})}, TOKEN)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://owned', headers={'Authorization': 'Bearer ' + TOKEN}) as client:
            result = await client.post('/v1/systemone', json=PAYLOAD)
            assert result.status_code == 503 and result.json() == {'error': 'inference_failed'}
