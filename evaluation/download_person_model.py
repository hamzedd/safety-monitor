"""Download and export the pinned COCO person detector (paired with SafetyVision v2).

Unlike download_model.py, this needs `pip install ultralytics` first (pulls in torch;
deliberately NOT added to requirements.txt, since the app itself only needs onnxruntime
for inference — this script is a one-time, laptop-side export step). Run it locally:

    .\\.venv\\Scripts\\python.exe -m pip install ultralytics
    .\\.venv\\Scripts\\python.exe -m evaluation.download_person_model

The .pt checkpoint download is integrity-checked against a fixed SHA-256 verified from
the official ultralytics/assets GitHub release (never load weights that fail it). The
ONNX export step that follows is NOT bit-reproducible across torch/ultralytics versions,
so its manifest carries the source checkpoint hash and exact export arguments rather than
an expected output hash; validate the actual exported contract at runtime instead, via
evaluation.detector.validate_person_session, before trusting it.
"""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RELEASE_TAG = 'v8.3.0'
URL = f'https://github.com/ultralytics/assets/releases/download/{RELEASE_TAG}/yolov8n.pt'
EXPECTED_SHA256 = 'f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36'
EXPECTED_BYTES = 6549796
CHECKPOINT = ROOT / 'models' / 'yolov8n.pt'
ONNX = ROOT / 'models' / 'yolov8n-person-640.onnx'
EXPORT_ARGS = dict(imgsz=640, dynamic=False, simplify=True, half=False, nms=False, opset=20)


def download_checkpoint():
    CHECKPOINT.parent.mkdir(exist_ok=True)
    partial = CHECKPOINT.with_suffix('.partial')
    try:
        digest = hashlib.sha256()
        size = 0
        with urllib.request.urlopen(URL, timeout=60) as response, partial.open('wb') as target:
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        actual = digest.hexdigest()
        if actual != EXPECTED_SHA256 or size != EXPECTED_BYTES:
            raise SystemExit(f'STOP: yolov8n.pt mismatch: expected {EXPECTED_BYTES}B/{EXPECTED_SHA256}, '
                             f'got {size}B/{actual}')
        partial.replace(CHECKPOINT)
    finally:
        partial.unlink(missing_ok=True)
    return actual


def export_onnx():
    from ultralytics import YOLO  # Imported lazily: only this script needs torch/ultralytics.
    model = YOLO(str(CHECKPOINT))
    exported = Path(model.export(format='onnx', **EXPORT_ARGS))
    exported.replace(ONNX)


def main():
    checkpoint_sha256 = download_checkpoint()
    export_onnx()
    manifest = dict(
        checkpoint_repository='ultralytics/assets', checkpoint_release_tag=RELEASE_TAG,
        checkpoint_url=URL, checkpoint_sha256=checkpoint_sha256, checkpoint_bytes=EXPECTED_BYTES,
        license='AGPL-3.0', role='person-only detector (COCO class 0 of 80); paired with '
        'the SafetyVision v2 model in models/manifest.json for Hardhat/Safety Vest',
        export_args=EXPORT_ARGS,
        note='ONNX bytes are not bit-reproducible across ultralytics/torch versions; the '
             'pinned checkpoint hash above is the verifiable source artifact. Verify the '
             'actual exported contract at runtime with evaluation.detector.'
             'validate_person_session before trusting this file.')
    (ROOT / 'models' / 'manifest_person.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
