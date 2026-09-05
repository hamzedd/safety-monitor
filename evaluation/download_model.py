"""Download a pinned model; never load weights that fail integrity verification."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPO = 'ayushgupta7777/safetyvision-yolov8'
REVISION = '56a71758b55f0e9f2b4b2d6b51a779a1f882da10'
SOURCE_REVISION = '3fba9d358321b14725ee6d329dd2bc99c89c8320'
EXPECTED = 'ea18ae903a566e8fa76f3ee1c503075522dca269269315e9c862efa170430b35'
URL = f'https://huggingface.co/{REPO}/resolve/{REVISION}/v2/best_640.onnx'
MODEL = ROOT / 'models' / 'safetyvision-v2-640.onnx'


def main():
    MODEL.parent.mkdir(exist_ok=True)
    partial = MODEL.with_suffix('.partial')
    try:
        digest = hashlib.sha256()
        with urllib.request.urlopen(URL, timeout=60) as response, partial.open('wb') as target:
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != EXPECTED:
            raise SystemExit(f'STOP: SHA-256 mismatch: expected {EXPECTED}, got {actual}')
        partial.replace(MODEL)
    finally:
        partial.unlink(missing_ok=True)
    reference = ROOT / 'evaluation' / 'reference'
    reference.mkdir(exist_ok=True)
    for remote, local in [('core/detector.py', 'detector.py.txt'), ('LICENSE', 'LICENSE.txt')]:
        url = f'https://raw.githubusercontent.com/ayushgupta07xx/SafetyVision/{SOURCE_REVISION}/{remote}'
        with urllib.request.urlopen(url, timeout=60) as response:
            (reference / local).write_bytes(response.read())
    manifest = dict(repository=REPO, revision=REVISION, url=URL, sha256=actual,
                    bytes=MODEL.stat().st_size, license='AGPL-3.0',
                    source_revision=SOURCE_REVISION,
                    source_url=f'https://github.com/ayushgupta07xx/SafetyVision/blob/{SOURCE_REVISION}/core/detector.py',
                    preserved_files={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                     for name in ['app.py', 'video_processing.py', '.streamlit/config.toml']})
    (ROOT / 'models' / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
