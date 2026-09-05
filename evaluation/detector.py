# SPDX-License-Identifier: AGPL-3.0-only
# Pre/postprocessing adapted from Ayush Gupta's SafetyVision core/detector.py.
# Pinned source and original license are recorded in models/manifest.json.
import ast
import hashlib
from pathlib import Path

import cv2
import numpy as np

from evaluation.download_model import EXPECTED

NAMES = dict(enumerate(['Fall-Detected', 'Gloves', 'Goggles', 'Hardhat', 'Mask',
                      'NO-Gloves', 'NO-Goggles', 'NO-Hardhat', 'NO-Mask',
                      'NO-Safety Vest', 'No_Harness', 'Person', 'Safety Vest']))
SELECTED = {3, 11, 12}


def verify_model(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != EXPECTED:
        raise ValueError(f'STOP: model SHA-256 mismatch: {digest}')
    return digest


def validate_session(session):
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or (inputs[0].name, inputs[0].shape, inputs[0].type) != (
        'images', [1, 3, 640, 640], 'tensor(float)'
    ):
        raise ValueError('Unexpected model input contract')
    if len(outputs) != 1 or (outputs[0].shape, outputs[0].type) != ([1, 17, 8400], 'tensor(float)'):
        raise ValueError('Unexpected model output contract')
    metadata = session.get_modelmeta().custom_metadata_map
    if ast.literal_eval(metadata.get('names', '{}')) != NAMES:
        raise ValueError('Unexpected class mapping')
    return metadata


def preprocess(image):
    height, width = image.shape[:2]
    scale = min(640 / height, 640 / width)
    resized_height, resized_width = round(height * scale), round(width * scale)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_height, pad_width = 640 - resized_height, 640 - resized_width
    top, left = pad_height // 2, pad_width // 2
    padded = cv2.copyMakeBorder(resized, top, pad_height - top, left, pad_width - left,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = (padded.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(tensor), scale, (left, top)


def decode(output, scale, pad, shape, confidence=0.40, iou=0.45, class_aware=True):
    if output.ndim != 3 or output.shape[:2] != (1, 17):
        raise ValueError('Expected output shape (1, 17, N)')
    if not np.isfinite(output).all():
        raise ValueError('Non-finite model output')
    predictions = output[0].T
    # Argmax over ALL 13 classes first, exactly as in the original decoder.
    # Excluded-class predictions must not be relabeled as a selected class.
    classes = predictions[:, 4:].argmax(axis=1)
    scores = predictions[:, 4:].max(axis=1)
    keep = (scores > confidence) & np.isin(classes, list(SELECTED))
    boxes = predictions[keep, :4].copy()
    scores, classes = scores[keep], classes[keep]
    if not len(boxes):
        return []
    boxes[:, :2] -= boxes[:, 2:] / 2  # center xywh -> top-left xywh
    indices = []
    groups = [np.flatnonzero(classes == c) for c in sorted(SELECTED)] if class_aware else [np.arange(len(boxes))]
    for group in groups:
        if len(group):
            local = cv2.dnn.NMSBoxes(boxes[group].tolist(), scores[group].tolist(), confidence, iou)
            indices.extend(group[np.asarray(local, dtype=int).reshape(-1)].tolist())
    height, width = shape
    detections = []
    for index in sorted(indices, key=lambda i: float(scores[i]), reverse=True):
        x, y, w, h = boxes[index]
        mapped = [(x - pad[0]) / scale, (y - pad[1]) / scale,
                  (x + w - pad[0]) / scale, (y + h - pad[1]) / scale]
        mapped = np.clip(mapped, [0, 0, 0, 0], [width, height, width, height]).tolist()
        if mapped[2] <= mapped[0] or mapped[3] <= mapped[1]:
            continue
        detections.append(dict(class_id=int(classes[index]), name=NAMES[int(classes[index])],
                               confidence=float(scores[index]), box=mapped))
    return detections
