# Construction-site safety review plan

Pipeline: video → sampled frames → people/helmet/vest detection → equipment-to-person matching → equipment detected / possible missing / uncertain → human review.

## Milestone 1: local upload and sampling

Streamlit accepts one MP4 up to 100 MiB, previews it, and reports duration, resolution and nominal frame rate. OpenCV visits frames sequentially and emits approximately one frame per second with an estimated timestamp. Each sample is resized to at most 640 pixels wide without upscaling. Processing is explicit, with progress and at most 12 retained preview JPEGs. Session state prevents repeated processing on UI reruns. Temporary input copies are removed after inspection and processing, including failures.

No detection models, simulated AI results, or safety classifications are included.

## Future milestones

1. Validate CPU-compatible people, helmet and vest detectors on representative, consented site footage. Measure memory, latency and detection quality before choosing models. Preserve source timestamps and pass samples incrementally to inference.
2. Match equipment to individual people using spatial relationships and confidence. Handle multiple nearby people, partial bodies, occlusion and small subjects explicitly.
3. Report evidence as equipment detected, possible missing, or uncertain. A missing detection alone must not be treated as proof that equipment is absent. Calibrate thresholds on reviewed examples and retain relevant image evidence.
4. Provide a human review queue with timestamps, evidence and editable decisions. Record reviewer corrections and evaluate false positives and false negatives before operational use.

## Constraints and decisions

- Windows, Python 3.12, 8 GB RAM, CPU only; bounded image retention and sequential processing.
- Current timestamp estimates assume constant nominal FPS. Variable-frame-rate presentation timestamps require a later decoding improvement.
- Future persistent frame/evidence storage needs explicit retention, access, deletion and disk-budget policies.
- Human review remains the final decision stage. Sampling may miss short events and cannot establish site-wide compliance.
