"""Render every video frame with actual CPU detections to a silent MP4.

No tracking or inferred safety violations. Processing is slower than playback.
Output uses nominal source FPS; variable-frame-rate timing is approximate.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

import cv2
import onnxruntime as ort

from evaluation.detector import (preprocess, decode, validate_session,
    validate_person_session, verify_model, COCO_NAMES, PERSON_SELECTED)
from evaluation.download_model import MODEL, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--person-model', required=True)
    args = parser.parse_args()
    source = Path(args.video)
    if not source.is_file():
        raise ValueError(f'Video not found: {source}')
    verify_model(MODEL)
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    ppe = ort.InferenceSession(str(MODEL), sess_options=options, providers=['CPUExecutionProvider'])
    person = ort.InferenceSession(args.person_model, sess_options=options, providers=['CPUExecutionProvider'])
    validate_session(ppe)
    validate_person_session(person)
    out_dir = ROOT / 'evaluation' / 'videos'
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = out_dir / f'{source.stem}_detected_{stamp}.mp4'
    capture = cv2.VideoCapture(str(source))
    writer = None
    complete = False
    count = 0
    start = time.perf_counter()
    try:
        if not capture.isOpened():
            raise ValueError('Cannot open input video')
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not 0 < fps < 1000:
            raise ValueError(f'Invalid video FPS: {fps}')
        expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = height = None
        while True:
            ok, original = capture.read()
            if not ok:
                break
            if writer is None:
                h, w = original.shape[:2]
                width = max(2, min(w, 640) // 2 * 2)
                height = max(2, round(h * width / w) // 2 * 2)
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height + 44))
                if not writer.isOpened():
                    raise RuntimeError('MP4 encoder unavailable; output could not be opened')
            frame = cv2.resize(original, (width, height), interpolation=cv2.INTER_AREA)
            tensor, scale, pad = preprocess(frame)
            groups = []
            for session, label in [(person, 'person'), (ppe, 'ppe')]:
                raw = session.run(None, {session.get_inputs()[0].name: tensor})[0]
                kwargs = dict(names=COCO_NAMES, selected=PERSON_SELECTED) if label == 'person' else dict(selected={3, 12})
                groups.extend(decode(raw, scale, pad, frame.shape[:2], confidence=.25, **kwargs))
            # Class colors identify detections, not safe/unsafe status.
            for det in groups:
                x1, y1, x2, y2 = map(round, det['box'])
                color = {0: (255, 200, 50), 3: (0, 220, 255), 12: (220, 100, 220)}[det['class_id']]
                name = {0: 'Person', 3: 'Helmet', 12: 'Vest'}[det['class_id']]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                text = f"{name} {det['confidence']:.2f}"
                # Put person labels below boxes to separate them from helmet labels.
                ty = min(height - 4, y2 + 13) if det['class_id'] == 0 else max(12, y1 - 4)
                tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .35, 1)[0][0]
                tx = max(0, min(x1, width - tw - 2))
                cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1, cv2.LINE_AA)
            annotated = cv2.copyMakeBorder(frame, 0, 44, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
            for line, text in enumerate([f'{count/fps:.2f}s | Detection preview | threshold 0.25',
                                         'Missed equipment is uncertain. No worker tracking.']):
                cv2.putText(annotated, text, (6, height+16+line*19), cv2.FONT_HERSHEY_SIMPLEX, .36, (240,240,240), 1, cv2.LINE_AA)
            writer.write(annotated)
            count += 1
            if count == 1 or count % 25 == 0:
                print(f'Processed {count}/{expected or "?"} frames | {time.perf_counter()-start:.0f}s elapsed', flush=True)
        if not count:
            raise ValueError('No decodable frames')
        if expected > 0 and count < expected:
            raise ValueError(f'Decode stopped early: {count}/{expected}; incomplete output removed')
        complete = True
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not complete:
            output.unlink(missing_ok=True)
    # Verify the encoded file can be reopened and its first frame decoded.
    check = cv2.VideoCapture(str(output))
    try:
        ok, _ = check.read()
        if not ok:
            raise RuntimeError(f'Output verification failed: {output}')
    finally:
        check.release()
    print(f'Saved {count} frames: {output}')
    print('Open in VLC. Output has no audio; nominal source playback speed is preserved.')


if __name__ == '__main__':
    main()
