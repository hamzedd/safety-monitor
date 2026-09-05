"""Compare baseline, lower thresholds and native-detail tiles; no app changes.

Run: python -m evaluation.compare_detection --video PATH --person-model PATH
Outputs are candidates for human review, never accuracy or compliance claims.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort

from evaluation.detector import (preprocess, decode, validate_session,
    validate_person_session, verify_model, COCO_NAMES, PERSON_SELECTED)
from evaluation.download_model import MODEL, ROOT
from evaluation.matching import match_people_to_equipment
from video_processing import inspect_video
from evaluation.geometry import tile_boxes, merge_detections


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def annotate(frame, detections):
    # Numbered boxes and a separate legend prevent overlapping text labels.
    canvas = frame.copy()
    colors = {'person': (50, 200, 50), 'ppe': (0, 200, 255)}
    for i, det in enumerate(detections, 1):
        x1, y1, x2, y2 = map(round, det['box'])
        color = colors[det['source']]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        # Stagger number placement; full names/scores are in the legend below.
        y = min(canvas.shape[0] - 5, max(15, y1 + 16 + (i % 3) * 16))
        cv2.putText(canvas, str(i), (max(0, x1), y), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, str(i), (max(0, x1), y), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, color, 1, cv2.LINE_AA)
    legend = np.full((max(1, len(detections)) * 23 + 12, frame.shape[1], 3), 245, np.uint8)
    for i, det in enumerate(detections, 1):
        text = f"{i}: {det['name']}  score={det['confidence']:.2f}"
        cv2.putText(legend, text, (8, 23 * i), cv2.FONT_HERSHEY_SIMPLEX, .5, (20, 20, 20), 1)
    return np.vstack([canvas, legend])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--person-model', required=True)
    args = parser.parse_args()
    verify_model(MODEL)
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    ppe = ort.InferenceSession(str(MODEL), sess_options=options, providers=['CPUExecutionProvider'])
    person = ort.InferenceSession(args.person_model, sess_options=options, providers=['CPUExecutionProvider'])
    validate_session(ppe)
    validate_person_session(person)
    info = inspect_video(Path(args.video))
    targets = sorted(set(round(x * (info.frame_count - 1) / 5) for x in range(6)))
    out = ROOT / 'evaluation/results' / ('comparison_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(parents=True)
    report = dict(status='requires_manual_review', video_sha256=file_hash(args.video),
                  ppe_sha256=file_hash(MODEL), person_sha256=file_hash(args.person_model),
                  note='Counts are detections, not accuracy. Timings include preprocessing and decoding, exclude image saving. Crops use native pixels; boundary boxes can remain imperfect after NMS.', frames=[])
    capture = cv2.VideoCapture(args.video)
    try:
        for index in range(info.frame_count):
            if not capture.grab():
                raise ValueError(f'Video ended unexpectedly at frame {index}')
            if index not in targets:
                continue
            ok, frame = capture.retrieve()
            if not ok:
                raise ValueError(f'Cannot decode frame {index}')
            height, width = frame.shape[:2]
            sampled = cv2.resize(frame, (min(width, 640), max(1, round(height * min(width, 640) / width))))
            variants = []
            for name, threshold, tiled in [('baseline_040', .40, False), ('full_025', .25, False), ('tiles_025', .25, True)]:
                start = time.perf_counter()
                # Full-frame pass is retained with tiles to preserve large people.
                views = [(sampled, 0, 0, width / sampled.shape[1], height / sampled.shape[0])]
                if tiled:
                    views += [(frame[y1:y2, x1:x2], x1, y1, 1, 1)
                              for x1, y1, x2, y2 in tile_boxes(width, height)]
                detections = []
                for view, ox, oy, sx, sy in views:
                    tensor, scale, pad = preprocess(view)
                    for session, source in [(ppe, 'ppe'), (person, 'person')]:
                        raw = session.run(None, {session.get_inputs()[0].name: tensor})[0]
                        kwargs = dict(names=COCO_NAMES, selected=PERSON_SELECTED) if source == 'person' else dict(selected={3, 12})
                        for det in decode(raw, scale, pad, view.shape[:2], confidence=threshold, **kwargs):
                            x1, y1, x2, y2 = det['box']
                            det.update(source=source, box=[x1*sx+ox, y1*sy+oy, x2*sx+ox, y2*sy+oy])
                            detections.append(det)
                detections = merge_detections(detections)
                # Matching thresholds are calibrated in 640-wide coordinates.
                mapped = [dict(d, box=[d['box'][0]*sampled.shape[1]/width,
                                      d['box'][1]*sampled.shape[0]/height,
                                      d['box'][2]*sampled.shape[1]/width,
                                      d['box'][3]*sampled.shape[0]/height]) for d in detections]
                matches = match_people_to_equipment(mapped)
                elapsed = (time.perf_counter()-start)*1000
                filename = f'{index:06d}_{name}.jpg'
                if not cv2.imwrite(str(out / filename), annotate(sampled, mapped)):
                    raise OSError('Cannot save image')
                variants.append(dict(name=name, confidence=threshold, tile_size=640 if tiled else None,
                    image=filename, counts=dict(Counter(d['name'] for d in mapped)),
                    elapsed_ms=elapsed, detections=mapped, person_matches=matches))
            report['frames'].append(dict(frame_index=index, timestamp_seconds=index/info.fps,
                                         coordinate_shape=list(sampled.shape), variants=variants))
            print(index, [(v['name'], v['counts']) for v in variants], flush=True)
    finally:
        capture.release()
    (out/'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Comparison saved to:', out)


if __name__ == '__main__':
    main()
