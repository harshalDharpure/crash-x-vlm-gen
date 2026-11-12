"""Fixed architecture ablations with isolated artifacts and resumable predictions.

No test metric is used to choose configurations or training checkpoints.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def infer(adapter, split, destination, frames, side):
    from crashx.data.dataset import load_jsonl, load_video_frames_cv2
    from crashx.models.vru_baseline import VRUBaselineModel
    from crashx.prompts import USER_PROMPT
    rows = load_jsonl(split)
    outputs = json.loads(destination.read_text(encoding="utf-8")) if destination.exists() else []
    ids = [o['video_id'] for o in outputs]
    expected = [r['video_id'] for r in rows]
    if ids != expected[:len(ids)]:
        raise ValueError(f'Prediction IDs do not match split: {destination}')
    if len(outputs) == len(rows):
        return
    model = VRUBaselineModel(lora_path=adapter, max_side=side).load()
    for rec in rows[len(outputs):]:
        # Only video pixels and a fixed prompt enter generation; labels are stored afterwards.
        text = model.generate_greedy(load_video_frames_cv2(rec['video_path'], frames),
                                     USER_PROMPT, max_new_tokens=512)
        outputs.append({**{k: rec[k] for k in ('video_id', 'severity', 'impact', 'start_sec',
                          'end_sec', 'vehicles', 'weather', 'explanation')},
                        'prediction': text, 'reference': rec['messages'][-1]['content']})
        atomic_json(destination, outputs)
        LOG.info('%s %s/%s', destination.name, len(outputs), len(rows))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=ROOT / 'results/architecture_v4')
    p.add_argument('--stage', choices=['suite', 'infer', 'score', 'statistics'], default='suite')
    p.add_argument('--adapter', type=Path)
    p.add_argument('--split', type=Path)
    p.add_argument('--destination', type=Path)
    p.add_argument('--frames', type=int, default=16)
    p.add_argument('--side', type=int, default=336)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.stage == 'infer':
        infer(args.adapter, args.split, args.destination, args.frames, args.side)
        return
    if args.stage in ('score', 'statistics'):
        from crashx.run_experiments import evaluate_condition
        outputs = json.loads(args.destination.read_text(encoding="utf-8"))
        if args.stage == 'score':
            atomic_json(args.destination.with_name(args.destination.name.replace('_predictions', '_metrics')),
                        evaluate_condition(outputs))
        from dataclasses import asdict
        from crashx.eval.bootstrap_stats import bootstrap_ci, paired_wilcoxon
        from crashx.eval.per_sample_scores import extract_per_sample_vectors, align_paired_vectors
        vectors = extract_per_sample_vectors(outputs)
        intervals = {k: asdict(bootstrap_ci(v, n_bootstrap=2000))
                     for k, v in vectors.items() if k != 'video_id'}
        report = {'intervals': intervals, 'paired_comparisons': {},
                  'note': 'BLEU/ROUGE here are means of per-video scores, not corpus metrics; '
                          'p-values are unadjusted exploratory tests.'}
        if args.destination.parent.name == 'test':
            for name in ('CrashLogic-7B-Greedy', 'CrashLogic-7B-SEASON',
                         'ZeroShot-Qwen2.5-VL-7B'):
                baseline = ROOT / 'results' / f'{name}_predictions.json'
                if not baseline.exists():
                    continue
                a, b = align_paired_vectors(vectors, extract_per_sample_vectors(json.loads(baseline.read_text(encoding="utf-8"))))
                if len(a['video_id']) != len(outputs):
                    raise ValueError('Baseline is missing test predictions')
                report['paired_comparisons'][name] = {
                    k: {'delta_ci': asdict(bootstrap_ci(a[k] - b[k], n_bootstrap=2000)),
                        'wilcoxon': asdict(paired_wilcoxon(a[k], b[k]))}
                    for k in a if k != 'video_id'}
        atomic_json(args.destination.with_name(args.destination.name.replace('_predictions', '_statistics')),
                    report)
        return
    args.root.mkdir(parents=True, exist_ok=True)
    configs = [dict(name=f'visual16-s{seed}', frames=16, side=336, visual=True, seed=seed)
               for seed in (42, 123, 2026)] + [
        dict(name='language16-s42', frames=16, side=336, visual=False, seed=42),
        dict(name='language8-s42', frames=8, side=224, visual=False, seed=42)]
    manifest = dict(configurations=configs, epochs=3, lr=5e-5, rank=32,
                    selection='minimum validation loss; fixed configurations; no test tuning',
                    gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    splits={s: hashlib.sha256((ROOT / f'crashx/data/splits/{s}.jsonl').read_bytes()).hexdigest()
                            for s in ('train', 'val', 'test')})
    manifest_path = args.root / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError('Existing experiment manifest differs; use a new result directory')
    atomic_json(manifest_path, manifest)
    status = {}
    def run(command):
        LOG.info('Running %s', ' '.join(map(str, command)))
        subprocess.run([str(v) for v in command], cwd=ROOT, check=True)
    # First exercise actual processor, quantized forward/backward and visual adapters.
    smoke = args.root / 'smoke_explicit_targets'
    if not (smoke / 'adapter_model.safetensors').exists():
        run([sys.executable, '-m', 'crashx.models.train_qlora', '--train-jsonl',
             ROOT / 'crashx/data/splits/train.jsonl', '--output-dir', smoke,
             '--visual-lora', '--num-frames', '16', '--max-side', '336',
             '--max-seq-len', '8192', '--max-steps', '1', '--grad-accum', '1'])
    for config in configs:
        name = config['name']
        adapter = ROOT / 'outputs/architecture_v4' / name
        status[name] = {'state': 'training', 'started': time.time()}
        atomic_json(args.root / 'status.json', status)
        try:
            if not (adapter / 'train_config.json').exists():
                cmd = [sys.executable, '-m', 'crashx.models.train_qlora', '--train-jsonl',
                       ROOT / 'crashx/data/splits/train.jsonl', '--val-jsonl',
                       ROOT / 'crashx/data/splits/val.jsonl', '--output-dir', adapter,
                       '--epochs', '3', '--lr', '5e-5', '--lora-r', '32', '--lora-alpha', '64',
                       '--seed', config['seed'], '--num-frames', config['frames'],
                       '--max-side', config['side'], '--max-seq-len', '8192']
                if config['visual']:
                    cmd.append('--visual-lora')
                run(cmd)
            for split in ('val', 'test'):
                dest = args.root / split / f'{name}_predictions.json'
                dest.parent.mkdir(parents=True, exist_ok=True)
                status[name]['state'] = f'{split}_inference'
                atomic_json(args.root / 'status.json', status)
                run([sys.executable, '-m', 'crashx.run_architecture_suite', '--stage', 'infer',
                     '--adapter', adapter, '--split', ROOT / f'crashx/data/splits/{split}.jsonl',
                     '--destination', dest, '--frames', config['frames'], '--side', config['side']])
                run([sys.executable, '-m', 'crashx.run_architecture_suite', '--stage', 'score',
                     '--destination', dest])
            status[name]['state'] = 'complete'
        except subprocess.CalledProcessError as exc:
            status[name].update(state='failed', error=str(exc))
            LOG.exception('Condition failed: %s', name)
        atomic_json(args.root / 'status.json', status)
    summaries = {}
    for split in ('val', 'test'):
        summaries[split] = {f.stem.removesuffix('_metrics'): json.loads(f.read_text(encoding="utf-8"))
                            for f in (args.root / split).glob('*_metrics.json')}
    # Across-training-seed uncertainty is separate from sample bootstrap uncertainty.
    import numpy as np
    seed_scores = [summaries['test'][c['name']] for c in configs[:3]
                   if c['name'] in summaries['test']]
    summaries['visual_seed_summary'] = {
        k: {'mean': float(np.mean([s[k] for s in seed_scores])),
            'std': float(np.std([s[k] for s in seed_scores], ddof=1)) if len(seed_scores) > 1 else None,
            'n_seeds': len(seed_scores)}
        for k in (seed_scores[0] if seed_scores else {})
        if isinstance(seed_scores[0][k], (float, int))}
    atomic_json(args.root / 'summary.json', summaries)
    if any(v['state'] != 'complete' for v in status.values()):
        raise SystemExit('Some experiments failed; inspect status.json and the worker log')


if __name__ == '__main__':
    main()
