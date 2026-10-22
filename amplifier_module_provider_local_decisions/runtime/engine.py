"""Pinned reference inference. Imported only by the optional service process.

Lightning uses the publisher's prompt, FP32 label rows and temperatures with a
Transformers port of its two-way pooled readout. d3-mini uses its publisher's
runtime. This is not an optimized vLLM server. No text generation or image URL
fetching is exposed. Publisher source files are hash-verified before import.
"""
import json
import os
from pathlib import Path
import sys
import time

from ..catalog import MODELS


class Engine:
    def __init__(self, identity, directory, device='cuda:0'):
        import torch
        from transformers import BitsAndBytesConfig
        self.torch = torch
        self.identity, self.spec, self.device = identity, MODELS[identity], device
        self.path = Path(directory) / self.spec['key']
        self.probes = []
        if self.spec['precision'] == 'int8' and not device.startswith('cuda'):
            raise ValueError('INT8 requires CUDA in this qualified runtime; use BF16 for CPU')
        kwargs = {'quantization_config': BitsAndBytesConfig(load_in_8bit=True)} if self.spec['precision'] == 'int8' else {}
        sys.path.insert(0, str(Path(directory) / 'd3-mini'))
        os.environ['D3_FAST'] = '0'
        started = time.perf_counter()
        if self.spec['key'] == 'd3-mini':
            from d3_runtime import D3, backbone_class
            if kwargs:
                # The published loader has no quantization argument. Intercept only
                # its backbone load during this single-threaded startup phase.
                from unittest.mock import patch
                cls = backbone_class('qwen3_5')
                original = cls.from_pretrained
                with patch.object(cls, 'from_pretrained', side_effect=lambda *a, **kw: original(*a, **kw, **kwargs)):
                    self.model = D3.from_pretrained(self.path, device=device, batch_size=1, verify='fast', local_files_only=True)
            else:
                self.model = D3.from_pretrained(self.path, device=device, batch_size=1, verify='fast', local_files_only=True)
            self.model.max_length = 32000
        else:
            from transformers import AutoTokenizer, Qwen3_5Model
            from safetensors.torch import load_file
            from safetensors import safe_open
            from d3_runtime import PromptEncoder
            sys.path.insert(0, str(self.path))
            import h2o_lightning_shim as shim
            self.shim = shim
            self.cfg = json.loads((self.path / 'serve_config.json').read_text())
            self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
            self.model = Qwen3_5Model.from_pretrained(self.path, dtype=torch.bfloat16,
                device_map={'': device}, attn_implementation='sdpa', local_files_only=True, **kwargs).eval().requires_grad_(False)
            self.encoder = PromptEncoder(self.model.language_model)
            tensors = load_file(self.path / 'label_rows.lightning')
            self.readout = tensors['W'].to(device=device, dtype=torch.float32)
            token_ids = tensors['token_ids'].tolist()
            with safe_open(self.path / 'label_rows.lightning', framework='pt') as source: meta = source.metadata()
            groups = json.loads(meta['groups']) if 'groups' in meta else [[label, [tid]] for label, tid in zip(json.loads(meta['label_order']), token_ids)]
            self.label_groups = {label: [token_ids.index(tid) for tid in ids] for label, ids in groups}
            self.softcap = float(meta['softcap']) if meta.get('softcap') not in (None, 'none', '') else 0
            for probe, reference in zip(shim.TWO_WAY_PROBES, self.cfg['two_way']['probe_reference']):
                for qid, question in probe['questions'].items():
                    with torch.inference_mode(): answer, _ = self._lightning(probe['state'], question, temperature=1)
                    delta = max(abs(answer['probabilities'][key] - p) for key, p in reference[qid].items())
                    self.probes.append({'question': qid, 'maximum_absolute_delta': delta})
            if max(p['maximum_absolute_delta'] for p in self.probes) > self.cfg['two_way']['probe_tolerance']:
                raise ValueError('Lightning failed publisher parity checks; service was not started')
        self.startup_seconds = time.perf_counter() - started

    def _lightning(self, state, question, temperature=None):
        torch, shim, cfg = self.torch, self.shim, self.cfg
        prompt_config = cfg['prompt']
        names, descriptions = shim.options_of(question)
        labels = cfg['labels'][:len(names)]
        user = shim.user_turn(shim.render_state(state, prompt_config['state_compact']),
            shim.render_instr(question['instructions']), names, descriptions, labels, prompt_config['layout'])
        prompt = shim.render_chat(prompt_config['system'], user, prompt_config['prefill'], prompt_config['chat_template'])
        encoded = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt').to(self.device)
        tokens = encoded['input_ids'].numel()
        if tokens > 32000: raise ValueError('Context limit; inputs were not truncated')
        embedding = self.model.get_input_embeddings()(encoded['input_ids'])
        hidden = self.encoder(embedding, encoded['attention_mask'])[:, -1, :].float()
        logits = (hidden @ self.readout.T)[0]
        if self.softcap: logits = self.softcap * torch.tanh(logits / self.softcap)
        grouped = torch.stack([torch.logsumexp(logits[self.label_groups[label]], dim=0) for label in labels])
        temperature = temperature if temperature is not None else cfg['two_way']['temperature'][question['type']]
        probabilities = dict(zip(names, (grouped / temperature).softmax(-1).cpu().tolist()))
        winner = max(probabilities, key=probabilities.get)
        answer = {'type': question['type'], 'probabilities': probabilities, 'confidence': probabilities[winner]}
        if question['type'] == 'noul': answer['noul'] = probabilities['true']
        elif question['type'] == 'choice': answer['choice'] = winner
        else: answer['score'] = sum(int(key) * p for key, p in probabilities.items())
        return answer, tokens

    def decide(self, request):
        with self.torch.inference_mode():
            if self.spec['key'] == 'd3-mini':
                result = self.model.system_one(state=request['state'], questions=request['questions'])
            else:
                answers, tokens = {}, 0
                for qid, question in request['questions'].items():
                    answers[qid], count = self._lightning(request['state'], question)
                    tokens += count
                result = {'answers': answers, 'usage': {'input_tokens': tokens, 'output_tokens': 0}}
        return {**result, 'model': self.identity, 'checkpoint': self.spec['checkpoint'], 'precision': self.spec['precision']}
