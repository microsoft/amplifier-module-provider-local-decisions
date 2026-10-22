"""Offline metadata: no Torch import, downloads, account access or inference."""
import ipaddress
from urllib.parse import urlsplit

MODELS = {
    'lightning-int8': {'key': 'lightning', 'precision': 'int8', 'label': 'H2O Lightning 4B · INT8',
        'checkpoint': 'd922608ae0abcd8de27948f004ecbb8f0f2334f9'},
    'd3-mini-int8': {'key': 'd3-mini', 'precision': 'int8', 'label': 'd3-mini · INT8 (experimental)',
        'checkpoint': 'bd0090dd758d32ba59ef5c749910a8e87fc4391f'},
    'lightning-bf16': {'key': 'lightning', 'precision': 'bf16', 'label': 'H2O Lightning 4B · BF16',
        'checkpoint': 'd922608ae0abcd8de27948f004ecbb8f0f2334f9'},
    'd3-mini-bf16': {'key': 'd3-mini', 'precision': 'bf16', 'label': 'd3-mini · BF16 (experimental)',
        'checkpoint': 'bd0090dd758d32ba59ef5c749910a8e87fc4391f'},
}
LOCAL_COST = {'api_cost_usd': 0, 'basis': 'self_hosted', 'infrastructure_included': False}


def endpoint(value):
    if not isinstance(value, str):
        raise ValueError('Choose a local decision service endpoint')
    url = urlsplit(value)
    try:
        _ = url.port
    except ValueError:
        raise ValueError('Invalid service port') from None
    try:
        loopback = ipaddress.ip_address(url.hostname or '').is_loopback
    except ValueError:
        loopback = False
    if (not url.hostname or url.username or url.password or url.query or url.fragment
            or url.scheme not in {'https', 'http'} or (url.scheme == 'http' and not loopback)):
        raise ValueError('Use HTTPS, or HTTP with a literal loopback address; omit credentials and query parameters')
    return value.rstrip('/')


def validate(config):
    if not isinstance(config, dict) or set(config) - {'model', 'endpoint'}:
        raise ValueError('Only model and endpoint belong in decision preferences')
    if config.get('model') not in MODELS:
        raise ValueError('Choose an installed Lightning or d3-mini precision variant')
    result = {'model': config['model']}
    if config.get('endpoint'):
        result['endpoint'] = endpoint(config['endpoint'])
    return result


def catalog():
    return {'schema_version': 2, 'provider_module': 'provider-local-decisions', 'validate': validate,
        'backends': [{'kind': 'local-decisions', 'label': 'Local decisions',
            'decision_only': True, 'connection': {'auth': 'key', 'endpoint': True,
                'endpoint_placeholder': 'http://127.0.0.1:18990/v1/systemone',
                'description': 'Connect an installed local decision service with its address and access token.'},
            'default_model': 'lightning-int8', 'models': list(MODELS), 'fixed_models': True,
            'model_labels': {key: row['label'] for key, row in MODELS.items()},
            'endpoint_optional': True, 'configuration_key': 'decisions', 'execution': 'self_hosted',
            'cost': LOCAL_COST,
            'cost_note': 'Local decisions have no API charge. Planning, text writing and ordinary-model fallback may still incur charges. Hardware and electricity are not included.'}]}
