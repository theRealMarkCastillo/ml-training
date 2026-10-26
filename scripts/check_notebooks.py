"""Validate notebook links and evidence, or execute all lessons in artifact-free copies.

Run with environments/notebooks/.venv/bin/python. --execute saves executed copies
under artifacts/validation/notebooks; --write also updates checked-in outputs.
"""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ['notebooks/lab.py', 'notebooks/data/recorded_results.json',
          'src/ml_training/query_dsl.py', 'src/ml_training/device_data.py',
          'src/ml_training/common.py', 'src/ml_training/__init__.py',
          'environments/notebooks/uv.lock']


def fingerprint(book):
    digest = hashlib.sha256()
    for cell in book.cells:
        digest.update(json.dumps([cell.cell_type, cell.source], ensure_ascii=False).encode())
    for path in INPUTS:
        digest.update(path.encode())
        digest.update((ROOT / path).read_bytes())
    return digest.hexdigest()


def check_structure(path, book):
    nbformat.validate(book)
    for cell in book.cells:
        if cell.cell_type == 'code':
            compile(cell.source, str(path), 'exec')
        elif cell.cell_type == 'markdown':
            for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)', cell.source):
                if '://' in link or link.startswith('#'):
                    continue
                target = (path.parent / link.split('#')[0]).resolve()
                if not target.exists():
                    raise ValueError(f'{path.name}: broken link {link}')


def check_outputs(path, book):
    plots = 0
    counts = []
    for cell in book.cells:
        if cell.cell_type != 'code':
            continue
        if not isinstance(cell.execution_count, int):
            raise ValueError(f'{path.name}: unexecuted cell')
        counts.append(cell.execution_count)
        for output in cell.outputs:
            if output.output_type == 'error':
                raise ValueError(f'{path.name}: error output')
            if 'image/png' in output.get('data', {}):
                plots += 1
    if counts != list(range(1, len(counts) + 1)) or plots < 2:
        raise ValueError(f'{path.name}: expected ordered execution and at least two figures')
    if book.metadata.get('ml_training', {}).get('execution_fingerprint') != fingerprint(book):
        raise ValueError(f'{path.name}: stale output fingerprint; execute again')
    return plots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--stale-only', action='store_true',
                        help='With --execute, reuse outputs whose input fingerprint still matches')
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--verify-sources', action='store_true')
    args = parser.parse_args()
    if args.write and not args.execute:
        parser.error('--write requires --execute')
    if args.stale_only and not args.execute:
        parser.error('--stale-only requires --execute')
    paths = sorted((ROOT / 'notebooks').glob('D*.ipynb'))
    if {p.name[:3] for p in paths} != {f'D{i:02}' for i in range(1, 10)} or len(paths) != 9:
        raise ValueError('Expected exactly one notebook for every demo D01–D09')
    snapshot = json.loads((ROOT / INPUTS[1]).read_text())
    if args.verify_sources:
        for path, expected in snapshot['sources'].items():
            if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Changed source: {path}')
    results = []
    output_dir = ROOT / 'artifacts/validation/notebooks'
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        book = nbformat.read(path, as_version=4)
        check_structure(path, book)
        execute = args.execute
        if args.stale_only:
            try:
                check_outputs(path, book)
                execute = False
            except ValueError:
                pass
        if execute:
            # No artifacts/, model checkpoints, data/generated or training environments.
            with tempfile.TemporaryDirectory(prefix='ml-lesson-') as directory:
                clean = Path(directory)
                shutil.copytree(ROOT / 'notebooks', clean / 'notebooks', ignore=shutil.ignore_patterns('__pycache__', '.ipynb_checkpoints'))
                shutil.copytree(ROOT / 'src', clean / 'src', ignore=shutil.ignore_patterns('__pycache__'))
                manager = KernelManager(kernel_name='python3')
                manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
                client = NotebookClient(book, km=manager, timeout=90,
                                        resources={'metadata': {'path': str(clean / 'notebooks')}},
                                        record_timing=False, allow_errors=False)
                try:
                    client.execute(env={**os.environ, 'MPLCONFIGDIR': str(output_dir / '.mpl-cache'),
                                        'IPYTHONDIR': str(clean / '.ipython')})
                finally:
                    # Supplying our own manager makes us responsible for shutdown.
                    # Release it on both successful execution and a failed cell.
                    if manager.has_kernel:
                        manager.shutdown_kernel(now=True)
                    if client.kc is not None:
                        client.kc.stop_channels()
            # Stored streams should be portable and must not expose temporary host paths.
            for cell in book.cells:
                for output in cell.get('outputs', []):
                    if output.output_type == 'stream':
                        output.text = output.text.replace(str(clean), '<temporary checkout>')
            book.metadata.ml_training['execution_fingerprint'] = fingerprint(book)
            nbformat.write(book, output_dir / path.name)
            if args.write:
                nbformat.write(book, path)
        plots = check_outputs(path, book)
        # Standalone images support visual QA without model reruns.
        number = 0
        for cell in book.cells:
            for output in cell.get('outputs', []):
                if 'image/png' in output.get('data', {}):
                    number += 1
                    (output_dir / f'{path.stem}-{number}.png').write_bytes(base64.b64decode(output.data['image/png']))
        results.append({'notebook': path.name, 'figures': plots, 'executed_this_run': execute,
                        'fingerprint': fingerprint(book)})
        print(f'PASS {path.name}: {plots} figures', flush=True)
    report = {'status': 'passed', 'executed_this_run': sum(r['executed_this_run'] for r in results),
              'execution_mode': 'fresh kernels for executed lessons; fingerprint verification for reused outputs' if args.execute else 'saved output and fingerprint checks',
              'sources_verified': args.verify_sources, 'notebooks': results,
              'scope': 'Plots and small deterministic examples. No model retraining or model inference replay.'}
    name = 'execution.json' if args.execute else 'validation.json'
    (output_dir / name).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
