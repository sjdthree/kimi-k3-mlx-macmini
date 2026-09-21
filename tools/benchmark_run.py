"""Run a local benchmark with owned processes and a fresh output directory.

No process-name kills, output deletion, or implicit checkpoint paths. This launches
real inference and can take hours. Inspect docs/evidence.md before comparing results.
"""
from __future__ import annotations
import argparse
import json
import os
import platform
import secrets
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--cache', type=Path)
    ap.add_argument('--out', type=Path, required=True, help='Fresh directory; never overwritten')
    ap.add_argument('--prompt-file', type=Path, default=Path('prompts/04_polymath_multi_expert.md'))
    ap.add_argument('--resident-gb', type=float, nargs='+', default=[8])
    ap.add_argument('--tokens', type=int, default=40)
    ap.add_argument('--max-memory-gb', type=float, default=48)
    ap.add_argument('--port', type=int, default=8080)
    a = ap.parse_args()
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', a.port))  # fail before starting if occupied
    a.out.mkdir(parents=True, exist_ok=False)
    key = secrets.token_urlsafe(32)
    env = dict(os.environ, AIR_LLM_API_KEYS=key, OPENAI_API_KEY=key)
    root = Path(__file__).resolve().parent.parent
    for index, gb in enumerate(a.resident_gb):
        dest = a.out / f'{index}-r{gb:g}'
        dest.mkdir()
        command = [sys.executable, '-m', 'air_llm.serve', '--model', str(a.model.resolve()),
                   '--manifest', str(a.manifest.resolve()), '--resident-gb', str(gb),
                   '--max-memory-gb', str(a.max_memory_gb), '--max-output-tokens', str(a.tokens),
                   '--port', str(a.port), '--token-log', str((dest / 'tokens.jsonl').resolve())]
        if a.cache:
            command += ['--cache', str(a.cache.resolve())]
        meta = {'command': command, 'platform': platform.platform(),
                'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                'dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root)),
                'checkpoint_revision': os.environ.get('K3_REVISION'),
                'prompt_sha256': __import__('hashlib').sha256(a.prompt_file.read_bytes()).hexdigest()}
        (dest / 'benchmark-meta.json').write_text(json.dumps(meta, indent=2) + '\n')
        with (dest / 'server.log').open('w') as log:
            process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 600
                while True:
                    if process.poll() is not None:
                        raise RuntimeError('Server exited; inspect server.log')
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{a.port}/readyz', timeout=2) as r:
                            if r.status == 200:
                                break
                    except OSError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError('Server readiness deadline exceeded')
                    time.sleep(1)
                subprocess.run([sys.executable, str(root / 'tools/run_long_job.py'),
                                '--prompt-file', str(a.prompt_file.resolve()), '--out', str(dest.resolve()),
                                '--max-tokens', str(a.tokens), '--base-url', f'http://127.0.0.1:{a.port}/v1'],
                               env=env, check=True, cwd=root)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=30)
    print(f'Results in {a.out}; record other machine load and swap separately.')


if __name__ == '__main__':
    main()
