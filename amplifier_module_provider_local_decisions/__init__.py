"""Lightweight decision-only provider for an explicitly configured local service."""
__amplifier_module_type__ = 'provider'

import os
import ssl
import httpx
from .catalog import MODELS, LOCAL_COST, endpoint, validate


class LocalDecisionsProvider:
    name = 'local-decisions'

    def __init__(self, api_key=None, config=None, *, client=None):
        self.config = config or {}
        self._api_key = api_key or self.config.get('api_key') or os.environ.get('LOCAL_DECISIONS_API_KEY')
        self._client, self._owned_client = client, client is None

    @property
    def client(self):
        if self._client is None:
            ca_file = self.config.get('tls_ca_file')
            verify = ssl.create_default_context(cafile=ca_file) if ca_file else True
            self._client = httpx.AsyncClient(follow_redirects=False, trust_env=False, verify=verify)
        return self._client

    @staticmethod
    def get_info():
        from amplifier_core import ProviderInfo, ConfigField
        return ProviderInfo(id='local-decisions', display_name='Local decision models',
            credential_env_vars=['LOCAL_DECISIONS_API_KEY'], capabilities=['decisions'],
            defaults={'chat_enabled': False}, config_fields=[
                ConfigField(id='api_key', display_name='Local service token', field_type='secret',
                            prompt='Token for your local inference service', env_var='LOCAL_DECISIONS_API_KEY'),
                ConfigField(id='base_url', display_name='Decision endpoint', field_type='text',
                            prompt='Local service URL ending in /v1/systemone'),
                ConfigField(id='tls_ca_file', display_name='Private CA file (optional)', field_type='text',
                            prompt='Path on the Amplifier host to your private service CA', required=False)])

    async def list_models(self):
        return []  # These models cannot complete ordinary conversations.

    def decision_backend(self, config, coordinator=None):
        from amplifier_decisions import Capabilities
        from amplifier_decisions.runtime import SystemOneBackend, DecisionFailure
        from amplifier_decisions.response import read_json
        selected = validate({key: config[key] for key in ('model', 'endpoint') if key in config})
        model = selected['model']
        target = endpoint(selected.get('endpoint') or self.config.get('base_url') or 'http://127.0.0.1:18990/v1/systemone')

        async def send(content, timeout):
            if not self._api_key:
                raise DecisionFailure('unavailable', 'missing_service_token')
            try:
                async with self.client.stream('POST', target, content=content, timeout=timeout,
                    headers={'Authorization': 'Bearer '+self._api_key, 'Content-Type': 'application/json'}) as response:
                    payload, request_id = await read_json(response)
            except httpx.TimeoutException:
                raise DecisionFailure('unavailable', 'timeout') from None
            except (httpx.HTTPError, OSError):
                raise DecisionFailure('unavailable', 'transport_error') from None
            if (not isinstance(payload, dict) or payload.get('model') != model
                    or payload.get('checkpoint') != MODELS[model]['checkpoint']
                    or payload.get('precision') != MODELS[model]['precision']):
                raise DecisionFailure('invalid', 'model_identity_mismatch')
            return payload, request_id

        return SystemOneBackend(self.name, model, send, coordinator=coordinator,
            capabilities=Capabilities(max_questions=8), checkpoint=MODELS[model]['checkpoint'], cost=LOCAL_COST)

    async def close(self):
        if self._owned_client and self._client is not None:
            await self._client.aclose()
            self._client = None


async def mount(coordinator, config=None):
    config = config or {}
    if (config.get('decisions') or {}).get('enabled') is not True:
        return None
    from amplifier_decisions.runtime import register
    from amplifier_decisions.config import configured_backends
    provider, removals = LocalDecisionsProvider(config=config), []
    try:
        for identity, row in configured_backends(config, 'local-decisions').items():
            removals.append(register(coordinator, identity, provider.decision_backend(row, coordinator)))
    except Exception:
        for remove in removals: await remove()
        await provider.close()
        raise
    async def cleanup():
        for remove in removals: await remove()
        await provider.close()
    return cleanup
