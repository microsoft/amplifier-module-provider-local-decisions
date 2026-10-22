"""Explicit downloads of public immutable checkpoints, verified before import."""
import hashlib
import json
from pathlib import Path
import urllib.request


def manifest():
    return {row['key']: row for row in json.loads(Path(__file__).with_name('checkpoints.json').read_text())}


def valid(path, row):
    if not path.is_file() or path.stat().st_size != row['size']:
        return False
    with path.open('rb') as source:
        if row.get('lfs'):
            return hashlib.file_digest(source, 'sha256').hexdigest() == row['lfs']['sha256']
        digest = hashlib.sha1(f"blob {row['size']}\0".encode())
        while chunk := source.read(1024 * 1024): digest.update(chunk)
        return digest.hexdigest() == row['blobId']


def required(keys):
    for key, model in manifest().items():
        for row in model['downloads']:
            # Lightning's attention-pattern port uses the pinned d3 PromptEncoder.
            if key in keys or (key == 'd3-mini' and row['rfilename'] == 'd3_runtime.py'):
                yield key, model, row


def verify(directory, keys):
    for key, _, row in required(keys):
        if not valid(Path(directory) / key / row['rfilename'], row):
            raise ValueError(f"Missing or modified pinned checkpoint file: {key}/{row['rfilename']}")


def download(directory, keys):
    for key, model, row in required(keys):
        target = Path(directory) / key / row['rfilename']
        target.parent.mkdir(parents=True, exist_ok=True)
        if valid(target, row): continue
        partial = target.with_name(target.name + '.part')
        url = f"https://huggingface.co/{model['repository']}/resolve/{model['revision']}/{row['rfilename']}?download=true"
        try:
            with urllib.request.urlopen(url, timeout=120) as source, partial.open('wb') as sink:
                while chunk := source.read(4 * 1024 * 1024): sink.write(chunk)
            if not valid(partial, row): raise ValueError('Checkpoint hash mismatch')
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)
        print(f"Verified {key}/{row['rfilename']}", flush=True)
