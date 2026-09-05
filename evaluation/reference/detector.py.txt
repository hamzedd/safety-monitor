"""YOLOv8s (v2) ONNX inference + PPE violation detection.

Loads v2 ONNX weights (best_640.onnx, 13-class) from HuggingFace Hub on first
init (cached under ~/.cache/huggingface/hub/), or from an explicit local path
(used by the Lambda container, which bakes the weights into the image). Runs
CPU inference via onnxruntime, applies NMS, and surfaces every PPE-absence
("NO-X") detection as a violation, pairing each with the best-overlapping
Person bbox when one exists.

Brief reference:
    Layer 1 — Computer Vision (core)
    Violation rule: ADR-010 — surface every NO-X detection; attach the
    nearest Person bbox (IoU >= PERSON_IOU_MIN) when present, else None.
"""

from __future__ import annotations

import ast
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

# ─── Constants ──────────────────────────────────────────────────────────────
HF_REPO = "ayushgupta7777/safetyvision-yolov8"
# Default = 640 (Lambda/local). HF Spaces sets SV_ONNX_FILENAME=v2/best_896.onnx
# to pull the higher-accuracy export. img_size auto-derives from the loaded ONNX
# input shape, so the filename is the ONLY thing that must change.
ONNX_FILENAME = os.getenv("SV_ONNX_FILENAME", "v2/best_640.onnx")

IMG_SIZE = 640  # fallback only; real value derived per-session from the ONNX input shape
DEFAULT_CONF_THRESHOLD = 0.40
DEFAULT_IOU_THRESHOLD = 0.45
PERSON_IOU_MIN = 0.05  # min IoU between NO-X and Person to attach person_bbox

# Risk tiers per violation class. Keys MUST byte-match the ONNX metadata names
# (case, hyphen vs underscore, spacing). Brief Layer 1 designates NO-Hardhat /
# NO-Safety Vest = HIGH, NO-Mask = MEDIUM, NO-Gloves = context-dependent,
# No_Harness = CRITICAL. NO-Goggles and Fall-Detected are v2-only (absent from
# the original brief) — tiers below are a sensible default; override if needed.
RISK_LEVELS: dict[str, str] = {
    "Fall-Detected": "CRITICAL",   # v2-only: detected fall event (not a NO-X)
    "No_Harness": "CRITICAL",      # fall protection at height
    "NO-Hardhat": "HIGH",
    "NO-Safety Vest": "HIGH",
    "NO-Goggles": "MEDIUM",        # v2-only: eye protection
    "NO-Mask": "MEDIUM",
    "NO-Gloves": "LOW",            # context-dependent (brief)
}
VIOLATION_CLASSES = set(RISK_LEVELS.keys())
PERSON_CLASS = "Person"


# ─── Result types ───────────────────────────────────────────────────────────
@dataclass
class Detection:
    cls: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in original image coords


@dataclass
class Violation:
    type: str
    risk_level: str
    confidence: float
    bbox: tuple[float, float, float, float]
    person_bbox: tuple[float, float, float, float] | None


@dataclass
class DetectionResult:
    detections: list[Detection]
    violations: list[Violation]
    image_shape: tuple[int, int]  # (h, w)
    inference_ms: float


# ─── Detector ───────────────────────────────────────────────────────────────
class PPEDetector:
    """YOLOv8 ONNX detector. First instance downloads weights from HF Hub
    (unless onnx_path is supplied, e.g. by the Lambda container)."""

    _instance: PPEDetector | None = None
    img_size: int = IMG_SIZE  # class default; __init__ sets the real value per ONNX

    def __init__(
        self,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        hf_repo: str = HF_REPO,
        onnx_path: str | Path | None = None,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Lambda bakes the .onnx into the container image and passes onnx_path
        # to skip the HF download. Local dev / HF Spaces leave it None → pull
        # from the Hub (cached after first run). v2 has no external-data sidecar.
        if onnx_path is not None:
            onnx_path = str(onnx_path)
        else:
            onnx_path = hf_hub_download(repo_id=hf_repo, filename=ONNX_FILENAME)

        logger.info("Loading ONNX model from %s", onnx_path)
        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        # ONNX static shape is [1, 3, H, W]; derive H. Fall back if dynamic.
        try:
            self.img_size = int(inp.shape[2])
        except (TypeError, ValueError):
            self.img_size = IMG_SIZE
        logger.info("ONNX input size: %d", self.img_size)

        # ultralytics embeds class names in ONNX metadata as a Python-dict string
        meta = self.session.get_modelmeta().custom_metadata_map
        names_raw = meta.get("names", "{}")
        self.class_names: dict[int, str] = ast.literal_eval(names_raw)
        logger.info(
            "Loaded %d classes: %s",
            len(self.class_names),
            list(self.class_names.values()),
        )

    @classmethod
    def get(cls) -> PPEDetector:
        """Singleton accessor for Lambda warm-container reuse."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── preprocessing ───────────────────────────────────────────────────────
    def _letterbox(
        self, img: np.ndarray
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        """Resize preserving aspect ratio, pad to img_size x img_size with gray."""
        size = self.img_size
        h, w = img.shape[:2]
        scale = min(size / h, size / w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_h, pad_w = size - new_h, size - new_w
        top, left = pad_h // 2, pad_w // 2
        padded = cv2.copyMakeBorder(
            resized, top, pad_h - top, left, pad_w - left,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )
        return padded, scale, (left, top)

    def _preprocess(
        self, img_bgr: np.ndarray
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        padded, scale, pad = self._letterbox(img_rgb)
        tensor = padded.astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)[None, :]  # HWC → NCHW
        return np.ascontiguousarray(tensor), scale, pad

    # ── postprocessing ──────────────────────────────────────────────────────
    def _postprocess(
        self,
        output: np.ndarray,
        scale: float,
        pad: tuple[int, int],
        orig_shape: tuple[int, int],
    ) -> list[Detection]:
        # YOLOv8 ONNX output: (1, 4 + num_classes, N)
        preds = output[0].T  # → (N, 4 + num_classes)
        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        confidences = class_scores.max(axis=1)

        # Confidence filter
        keep = confidences > self.conf_threshold
        if not keep.any():
            return []
        boxes_xywh = boxes_xywh[keep]
        class_ids = class_ids[keep]
        confidences = confidences[keep]

        # xywh (center) → xyxy
        cx, cy, w, h = boxes_xywh.T
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # NMS via OpenCV (expects [x, y, w, h])
        boxes_for_nms = np.stack([x1, y1, w, h], axis=1).tolist()
        idxs = cv2.dnn.NMSBoxes(
            boxes_for_nms, confidences.tolist(),
            self.conf_threshold, self.iou_threshold,
        )
        if len(idxs) == 0:
            return []
        idxs_arr = np.array(idxs).flatten()

        # Undo letterbox: subtract pad, divide by scale, clamp to image bounds
        pad_x, pad_y = pad
        orig_h, orig_w = orig_shape
        results: list[Detection] = []
        for i in idxs_arr:
            bx1 = max(0.0, (boxes_xyxy[i, 0] - pad_x) / scale)
            by1 = max(0.0, (boxes_xyxy[i, 1] - pad_y) / scale)
            bx2 = min(float(orig_w), (boxes_xyxy[i, 2] - pad_x) / scale)
            by2 = min(float(orig_h), (boxes_xyxy[i, 3] - pad_y) / scale)
            results.append(Detection(
                cls=self.class_names[int(class_ids[i])],
                confidence=float(confidences[i]),
                bbox=(bx1, by1, bx2, by2),
            ))
        return results

    # ── violation logic ─────────────────────────────────────────────────────
    @staticmethod
    def _iou(
        a: tuple[float, float, float, float], b: tuple[float, float, float, float]
    ) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / union if union > 0 else 0.0

    def _detect_violations(self, dets: list[Detection]) -> list[Violation]:
        """Surface every NO-X detection as a violation. Pair with the
        best-overlapping Person bbox when one exists (IoU >= PERSON_IOU_MIN);
        otherwise leave person_bbox=None.

        ADR-010: the NO-X training classes are themselves annotated on people
        without PPE, so the class label is sufficient violation evidence.
        Strict Person + NO-X pairing was defensive redundancy that introduced
        false negatives on occluded / partial-pose workers — the worse
        failure mode for a safety screening tool.
        """
        persons = [d for d in dets if d.cls == PERSON_CLASS]
        violations: list[Violation] = []
        for d in dets:
            if d.cls not in VIOLATION_CLASSES:
                continue
            best_iou, best_person = 0.0, None
            for p in persons:
                iou = self._iou(d.bbox, p.bbox)
                if iou > best_iou:
                    best_iou, best_person = iou, p
            paired = best_iou >= PERSON_IOU_MIN and best_person is not None
            violations.append(Violation(
                type=d.cls,
                risk_level=RISK_LEVELS[d.cls],
                confidence=d.confidence,
                bbox=d.bbox,
                person_bbox=best_person.bbox if (paired and best_person is not None) else None,
            ))
        return violations

    # ── public API ──────────────────────────────────────────────────────────
    def predict(self, image: np.ndarray | str | Path) -> DetectionResult:
        """Run detection. Accepts BGR ndarray or image path."""
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Could not read image: {image}")
        else:
            img = image
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected BGR image (H, W, 3), got shape {img.shape}")

        h, w = img.shape[:2]
        tensor, scale, pad = self._preprocess(img)

        t0 = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: tensor})
        inference_ms = (time.perf_counter() - t0) * 1000

        detections = self._postprocess(outputs[0], scale, pad, (h, w))
        violations = self._detect_violations(detections)

        return DetectionResult(
            detections=detections,
            violations=violations,
            image_shape=(h, w),
            inference_ms=inference_ms,
        )


# ─── Annotation helper ──────────────────────────────────────────────────────
def draw_annotations(image: np.ndarray, result: DetectionResult) -> np.ndarray:
    """Draw bboxes on a BGR image copy.

    Red = violation (NO-X paired with a person), Yellow = person,
    Green = positive PPE / other.
    """
    out = image.copy()
    violation_bbox_set = {tuple(v.bbox) for v in result.violations}
    for det in result.detections:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        if tuple(det.bbox) in violation_bbox_set:
            color = (0, 0, 255)       # red — BGR
        elif det.cls == PERSON_CLASS:
            color = (0, 255, 255)     # yellow — BGR
        else:
            color = (0, 255, 0)       # green — BGR

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{det.cls} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            out, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )
    return out
