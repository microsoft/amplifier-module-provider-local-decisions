"""Authenticated, bounded, single-inference service; never executes UI actions."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import hmac
import json
import threading
import uuid

from amplifier_decisions import Capabilities, DecisionRequest, Question
from amplifier_decisions.systemone import SystemOneCodec
from starlette.applications import Starlette
from starlette.requests import ClientDisconnect
from starlette.responses import JSONResponse
from starlette.routing import Route
from ..catalog import LOCAL_COST


def validate_request(payload, models):
    if not isinstance(payload, dict) or set(payload) != {'model', 'state', 'questions'} or payload['model'] not in models:
        raise ValueError('Invalid model or envelope')
    if not isinstance(payload['questions'], dict): raise ValueError('Invalid questions')
    questions = []
    for name, row in payload['questions'].items():
        if not isinstance(row, dict) or set(row) - {'type', 'instructions', 'criteria'}:
            raise ValueError('Invalid question')
        kind = {'noul': 'predicate', 'choice': 'choice', 'score': 'score'}[row['type']]
        criteria = row.get('criteria')
        if (kind == 'choice' and not isinstance(criteria, dict)
                or kind == 'score' and not isinstance(criteria, list)
                or kind == 'predicate' and criteria is not None):
            raise ValueError('Invalid criteria')
        questions.append(Question(name, kind, row['instructions'],
            criteria if kind == 'choice' else {}, tuple(criteria) if kind == 'score' else ()))
    request = DecisionRequest(payload['state'], tuple(questions), allow_external_state=True)
    codec = SystemOneCodec(payload['model'])
    codec.capabilities = Capabilities(max_questions=8)
    codec._body(request)
    return request, codec


def create_app(engines, token, *, inference_timeout=120):
    if not isinstance(token, str) or len(token) < 32: raise ValueError('Use a private service token of at least 32 characters')
    pool, gate = ThreadPoolExecutor(max_workers=1), threading.BoundedSemaphore(1)
    @asynccontextmanager
    async def lifespan(app):
        yield
        pool.shutdown(wait=True, cancel_futures=True)
    def authorized(request):
        return hmac.compare_digest(request.headers.get('authorization', '').encode(), ('Bearer ' + token).encode())
    async def health(request):
        return JSONResponse({'ready': True})
    async def models(request):
        if not authorized(request): return JSONResponse({'error': 'unauthorized'}, status_code=401)
        return JSONResponse({'models': list(engines), 'cost': LOCAL_COST, 'images': False})
    async def decide(request):
        if not authorized(request): return JSONResponse({'error': 'unauthorized'}, status_code=401)
        if request.headers.get('content-type', '').split(';')[0] != 'application/json':
            return JSONResponse({'error': 'json_required'}, status_code=415)
        try:
            data = bytearray()
            async with asyncio.timeout(10):
                async for chunk in request.stream():
                    data.extend(chunk)
                    if len(data) > 131072: return JSONResponse({'error': 'request_limit'}, status_code=413)
            payload = json.loads(data)
            typed, codec = validate_request(payload, engines)
        except (ValueError, TypeError, KeyError, RecursionError):
            return JSONResponse({'error': 'invalid_request'}, status_code=422)
        except (TimeoutError, ClientDisconnect):
            return JSONResponse({'error': 'request_timeout'}, status_code=408)
        if not gate.acquire(blocking=False):
            return JSONResponse({'error': 'busy'}, status_code=503)
        def infer():
            try:
                value = engines[payload['model']].decide(payload)
                answers = value.get('answers', {})
                if set(answers) != {q.name for q in typed.questions} or any(codec._answer(q, answers[q.name]).status != 'answered' for q in typed.questions):
                    raise ValueError('Invalid model output')
                return value
            finally:
                gate.release()
        # Cancellation/timeout does not release the gate while GPU work is active.
        future = asyncio.get_running_loop().run_in_executor(pool, infer)
        future.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
        try:
            value = await asyncio.wait_for(asyncio.shield(future), inference_timeout)
            return JSONResponse({**value, 'cost': LOCAL_COST}, headers={'x-request-id': uuid.uuid4().hex})
        except TimeoutError:
            return JSONResponse({'error': 'inference_timeout'}, status_code=504)
        except Exception:
            return JSONResponse({'error': 'inference_failed'}, status_code=503)
    return Starlette(routes=[Route('/health', health), Route('/v1/models', models),
                            Route('/v1/systemone', decide, methods=['POST'])], lifespan=lifespan)
