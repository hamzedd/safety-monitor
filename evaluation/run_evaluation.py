"""Run with python -m evaluation.run_evaluation --video PATH. No app changes."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import threading
import time

import cv2
import numpy as np
import onnxruntime as ort
import psutil

from evaluation.detector import (COCO_NAMES, PERSON_SELECTED, decode, preprocess,
                                 validate_person_session, validate_session, verify_model)
from evaluation.download_model import MODEL, ROOT
from evaluation.matching import match_people_to_equipment
from video_processing import inspect_video


def memory(process):
    info = process.memory_info()
    return {key: getattr(info, key) for key in ['rss', 'private', 'peak_wset'] if hasattr(info, key)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--person-model', help='Path to the exported COCO person-only ONNX '
                        '(see evaluation.download_person_model). Omit to evaluate Hardhat/'
                        'Safety Vest alone, as before.')
    args = parser.parse_args()
    video = Path(args.video)
    output_dir = ROOT / 'evaluation' / 'results'
    output_dir.mkdir(exist_ok=True)
    verify_model(MODEL)  # Must happen before creating any model session.
    process = psutil.Process()
    baseline = memory(process)
    peaks = baseline.copy()
    stop = threading.Event()

    def monitor():
        while not stop.wait(0.02):
            for key, value in memory(process).items():
                peaks[key] = max(peaks.get(key, 0), value)

    worker = threading.Thread(target=monitor, daemon=True)
    worker.start()
    try:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        start = time.perf_counter()
        session = ort.InferenceSession(str(MODEL), sess_options=options, providers=['CPUExecutionProvider'])
        load_ms = (time.perf_counter() - start) * 1000
        metadata = validate_session(session)
        person_session = None
        person_load_ms = None
        if args.person_model:
            start = time.perf_counter()
            person_session = ort.InferenceSession(str(args.person_model), sess_options=options,
                                                  providers=['CPUExecutionProvider'])
            person_load_ms = (time.perf_counter() - start) * 1000
            validate_person_session(person_session)
        after_load = memory(process)
        info = inspect_video(video)
        targets = sorted(set(round(x * (info.frame_count - 1) / 5) for x in range(6)))
        capture = cv2.VideoCapture(str(video))
        records, warm_ms, person_warm_ms = [], [], []
        cold_ms = person_cold_ms = None
        try:
            for index in range(info.frame_count):
                if not capture.grab():
                    raise ValueError(f'Video decode ended at frame {index}')
                if index not in targets:
                    continue
                ok, frame = capture.retrieve()
                if not ok:
                    raise ValueError(f'Cannot retrieve frame {index}')
                height, width = frame.shape[:2]
                if width > 640:
                    frame = cv2.resize(frame, (640, max(1, round(height * 640 / width))), interpolation=cv2.INTER_AREA)
                start = time.perf_counter()
                tensor, scale, pad = preprocess(frame)
                preprocess_ms = (time.perf_counter() - start) * 1000
                runs = []
                for repeat in range(4):
                    start = time.perf_counter()
                    raw = session.run(None, {'images': tensor})[0]
                    elapsed = (time.perf_counter() - start) * 1000
                    runs.append(elapsed)
                    if cold_ms is None:
                        cold_ms = elapsed
                    else:
                        warm_ms.append(elapsed)
                start = time.perf_counter()
                ppe_detections = decode(raw, scale, pad, frame.shape[:2])
                decode_ms = (time.perf_counter() - start) * 1000
                original_nms = decode(raw, scale, pad, frame.shape[:2], class_aware=False)
                diagnostic = decode(raw, scale, pad, frame.shape[:2], confidence=.25)
                for det in ppe_detections:
                    det['source'] = 'ppe'

                person_detections, person_runs = [], []
                if person_session is not None:
                    for repeat in range(4):
                        start = time.perf_counter()
                        person_raw = person_session.run(None, {'images': tensor})[0]
                        elapsed = (time.perf_counter() - start) * 1000
                        person_runs.append(elapsed)
                        if person_cold_ms is None:
                            person_cold_ms = elapsed
                        else:
                            person_warm_ms.append(elapsed)
                    person_detections = decode(person_raw, scale, pad, frame.shape[:2],
                                               names=COCO_NAMES, selected=PERSON_SELECTED)
                    for det in person_detections:
                        det['source'] = 'person'

                detections = ppe_detections + person_detections
                filename = f'frame_{index:06d}_{index / info.fps:.3f}s.jpg'
                annotated = frame.copy()
                colors = {3: (0, 210, 255), 11: (255, 200, 0), 12: (180, 80, 255), 0: (60, 220, 60)}
                for det in detections:
                    x1, y1, x2, y2 = map(round, det['box'])
                    color = colors[det['class_id']]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = f"{det['name']} {det['confidence']:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .45, 1)
                    tx = max(0, min(x1, annotated.shape[1] - tw - 4))
                    ty = max(th + 4, y1)
                    cv2.rectangle(annotated, (tx, ty - th - 4), (tx + tw + 4, ty + 3), color, -1)
                    cv2.putText(annotated, label, (tx + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1, cv2.LINE_AA)
                if not cv2.imwrite(str(output_dir / filename), annotated):
                    raise OSError('Cannot save annotated frame')
                max_raw_target_scores = {name: float(raw[0, 4 + cls].max())
                                         for cls, name in [(3, 'Hardhat'), (11, 'Person'), (12, 'Safety Vest')]}
                if person_session is not None:
                    max_raw_target_scores['person'] = float(person_raw[0, 4, :].max())
                person_matches = match_people_to_equipment(detections) if person_session is not None else None
                records.append(dict(frame_index=index, timestamp_seconds=index / info.fps,
                                    image=filename, shape=list(frame.shape), detections=detections,
                                    counts=dict(Counter(d['name'] for d in detections)),
                                    diagnostic_counts_at_025=dict(Counter(d['name'] for d in diagnostic)),
                                    max_raw_target_scores=max_raw_target_scores,
                                    class_agnostic_selected_counts=dict(Counter(d['name'] for d in original_nms)),
                                    inference_ms=runs, preprocess_ms=preprocess_ms, decode_ms=decode_ms,
                                    person_inference_ms=person_runs or None,
                                    person_matches=person_matches))
        finally:
            capture.release()
        person_report = None
        if person_session is not None:
            person_manifest_path = ROOT / 'models/manifest_person.json'
            person_report = dict(
                model=str(args.person_model),
                manifest=json.loads(person_manifest_path.read_text()) if person_manifest_path.exists() else None,
                input_shape=person_session.get_inputs()[0].shape,
                output_shape=person_session.get_outputs()[0].shape,
                providers=person_session.get_providers(),
                load_ms=person_load_ms, cold_inference_ms=person_cold_ms,
                warm_inference=dict(count=len(person_warm_ms),
                                    mean_ms=statistics.mean(person_warm_ms) if person_warm_ms else None,
                                    median_ms=statistics.median(person_warm_ms) if person_warm_ms else None,
                                    min_ms=min(person_warm_ms) if person_warm_ms else None,
                                    max_ms=max(person_warm_ms) if person_warm_ms else None))
        report = dict(video=str(video), video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                      video_info=vars(info), duration_seconds=info.duration,
                      model_manifest=json.loads((ROOT / 'models/manifest.json').read_text()),
                      person_model=person_report,
                      metadata=metadata, input_shape=session.get_inputs()[0].shape,
                      output_shape=session.get_outputs()[0].shape, providers=session.get_providers(),
                      versions={p: importlib.metadata.version(p) for p in ['onnxruntime','psutil','numpy','opencv-python-headless','streamlit']},
                      python=platform.python_version(), platform=platform.platform(),
                      threads=dict(intra_op=2, inter_op=1), confidence_threshold=.4, nms_iou=.45,
                      load_ms=load_ms, cold_inference_ms=cold_ms,
                      warm_inference=dict(count=len(warm_ms), mean_ms=statistics.mean(warm_ms), median_ms=statistics.median(warm_ms),
                                          min_ms=min(warm_ms), max_ms=max(warm_ms)),
                      memory_bytes=dict(baseline=baseline, after_load=after_load, sampled_peak=peaks.copy(), final=memory(process)),
                      frames=records)
        (output_dir / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({key: value for key, value in report.items() if key not in ['metadata','model_manifest','frames']}, indent=2))
        print('Frame counts:', [(r['timestamp_seconds'], r['counts']) for r in records])
        if person_session is not None:
            print('Per-person matches:', [(r['timestamp_seconds'],
                 [{'hardhat': m['hardhat'], 'vest': m['vest']} for m in r['person_matches']['people']])
                 for r in records])
            print('Unmatched equipment (detected but not confidently linked to any person):',
                 [(r['timestamp_seconds'],
                   dict(hardhats=len(r['person_matches']['unmatched_hardhats']),
                        vests=len(r['person_matches']['unmatched_vests'])))
                  for r in records])
    finally:
        stop.set()
        worker.join()


if __name__ == '__main__':
    main()
