"""Explicit operator entry point. Normal provider installation starts nothing."""
import argparse
import ipaddress
import json
import os
from pathlib import Path
from ..catalog import MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['download', 'serve'])
    parser.add_argument('--model', action='append', required=True, choices=list(MODELS))
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0', choices=['cuda:0', 'cpu'])
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=18990)
    parser.add_argument('--token-file', type=Path)
    parser.add_argument('--tls-cert', type=Path)
    parser.add_argument('--tls-key', type=Path)
    parser.add_argument('--startup-receipt', type=Path)
    args = parser.parse_args()
    keys = {MODELS[identity]['key'] for identity in args.model}
    from .checkpoints import download, verify
    if args.command == 'download':
        download(args.directory, keys)
        return
    if not args.token_file: parser.error('serve requires --token-file')
    if not 1 <= args.threads <= 64: parser.error('threads must be between 1 and 64')
    if not ipaddress.ip_address(args.host).is_loopback and not (args.tls_cert and args.tls_key):
        parser.error('Non-loopback listeners require TLS')
    token = args.token_file.read_text().strip()
    if args.token_file.stat().st_mode & 0o077: parser.error('Service token file must be private (0600)')
    verify(args.directory, keys)
    os.environ.update(HF_HUB_OFFLINE='1', HF_HUB_DISABLE_PROGRESS_BARS='1', TOKENIZERS_PARALLELISM='false',
                      OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261019)
    if args.device.startswith('cuda'): torch.backends.cuda.enable_cudnn_sdp(False)
    else:
        import transformers.utils.import_utils as imports
        import transformers.utils as utils
        for module in (imports, utils):
            for name in ('is_flash_linear_attention_available', 'is_causal_conv1d_available'):
                if hasattr(module, name): setattr(module, name, lambda: False)
    from .engine import Engine
    engines = {identity: Engine(identity, args.directory, args.device) for identity in dict.fromkeys(args.model)}
    if args.startup_receipt:
        args.startup_receipt.write_text(json.dumps({'device': args.device, 'torch': torch.__version__, 'models': {
            identity: {**engine.spec, 'startup_seconds': engine.startup_seconds, 'parity_probes': engine.probes}
            for identity, engine in engines.items()}}, indent=2) + '\n')
    from .server import create_app
    import uvicorn
    uvicorn.run(create_app(engines, token), host=args.host, port=args.port, access_log=False,
        ssl_certfile=str(args.tls_cert) if args.tls_cert else None,
        ssl_keyfile=str(args.tls_key) if args.tls_key else None, limit_concurrency=16)


if __name__ == '__main__': main()
