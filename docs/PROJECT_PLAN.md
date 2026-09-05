# Construction-site safety review plan

Pipeline: upload video → check format/size → resize and sample frames → detect people,
helmets and vests → match equipment to each person → for each equipment type, decide
equipment detected / possible missing / uncertain → show annotated frames and timestamps
for human review.

**The central rule: a missing detection is never treated as proof equipment is absent.**
Three outcomes per equipment type, not two:

| Signal | Meaning | Shown as |
| --- | --- | --- |
| Equipment detected on the person | Equipment detected | Green |
| Relevant body area visible, no equipment detected | Possible missing equipment | Red |
| Relevant body area not visible/assessable (occluded, too far, blurry) | Unable to assess / uncertain | Grey |

Green never means the worker or site is fully safe; red never means a confirmed
violation — both require human review. Live cameras, gloves, automatic alerts, and
identifying individual workers are explicitly out of scope for this first version.

## Milestone 1: local upload and sampling

Streamlit accepts one MP4 up to 100 MiB, previews it, and reports duration, resolution and nominal frame rate. OpenCV visits frames sequentially and emits approximately one frame per second with an estimated timestamp. Each sample is resized to at most 640 pixels wide without upscaling. Processing is explicit, with progress and at most 12 retained preview JPEGs. Session state prevents repeated processing on UI reruns. Temporary input copies are removed after inspection and processing, including failures.

No detection models, simulated AI results, or safety classifications are included.

## Build order and status

1. **Video upload and frame extraction** — done (Milestone 1 above).
2. **Helmet-and-vest detection model** — evaluated; see [MODEL_EVALUATION.md](MODEL_EVALUATION.md).
   SafetyVision v2 (YOLOv8s ONNX) detects Hardhat/Safety Vest plausibly but never detects
   Person on real footage (raw confidence ~0.0002), so it cannot anchor matching alone.
   Decision: pair it with a separately verified COCO person detector
   (`evaluation/download_person_model.py`, Ultralytics YOLOv8n, checkpoint hash-pinned and
   verified) for the "person" side only; Hardhat/Vest stay on the original model.
   `evaluation/run_evaluation.py --person-model PATH` runs and merges both. **Run against
   real footage on 2026-09-06**: the paired detector found 3-5 people per frame where
   SafetyVision v2 alone found zero (raw score ~0.0002); combined latency ~451ms/frame warm,
   peak memory well under the 8GB budget. See MODEL_EVALUATION.md's real-footage addendum.
   Still pending: manual visual confirmation that the person boxes land on the actual
   workers, not background.
3. **Person-to-equipment matching and uncertainty rules** — implemented in
   `evaluation/matching.py` (`match_people_to_equipment`), independent of any upstream
   logic; 7 unit tests. Approximates head/torso zones as fixed fractions (top 25% / 20-65%)
   of each person box, since detections give no pose/keypoints. Equipment matches the
   person whose zone it overlaps most (≥50% of the equipment box inside the zone),
   one-to-one. Classifies each equipment type as detected / possible_missing / uncertain
   per the table above — never collapses "not detected" into "missing". **Known
   limitation**: only two visibility signals downgrade a call to uncertain — a very small
   person box, or a box whose top touches the frame edge (head likely cropped). Genuine
   occlusion (a hand over the head) or unusual poses (crouching, bending) aren't detected
   by box geometry alone and will read as possible_missing rather than uncertain; this
   needs either pose estimation or human review to catch. Wired into
   `evaluation/run_evaluation.py` as `person_matches` per frame — **not yet reviewed
   against real footage** (needs the same manual visual check as the person-detector boxes
   above, comparing each match/uncertain/missing call against what's actually in frame).
4. **Annotated frames and timestamped findings list in the app** — not started. Depends on
   steps 2-3 producing real per-frame results to display; reuses Milestone 1's incremental
   per-sample callback and 12-preview retention limit.
5. **Compare against manually checked footage** — not started. Calibrate thresholds on
   reviewed examples; record false positives/negatives before treating results as anything
   more than a review aid.

A human review queue with editable decisions and reviewer-correction tracking is a later
milestone beyond this build order, once steps 1-5 produce a working, testable demo.

## Constraints and decisions

- Windows, Python 3.12, 8 GB RAM, CPU only; bounded image retention and sequential processing.
- Current timestamp estimates assume constant nominal FPS. Variable-frame-rate presentation timestamps require a later decoding improvement.
- Future persistent frame/evidence storage needs explicit retention, access, deletion and disk-budget policies.
- Human review remains the final decision stage. Sampling may miss short events and cannot establish site-wide compliance.
