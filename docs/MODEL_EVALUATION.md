# SafetyVision v2 CPU evaluation

> CURRENT REVIEW STATUS: the tables below are historical reports. The committed
> `evaluation/results/metrics.json` and JPEGs contain the original single-model run,
> not paired-model or current matching evidence. The paired-run numbers below have
> not been independently reproduced in this checkout. Visual verification remains
> pending. Do not treat this report as approval to integrate safety assessments.
> Matching now emits only detected spatial associations or uncertain; it never
> infers missing equipment from unmatched boxes.
> New evaluator runs save to separate timestamped subdirectories of
> `evaluation/results/` and include a run ID, matching policy and person model hash.
> A fresh paired run on the source video and manual image review are still required.


Date: 2026-09-05. Status: evaluation complete; **not recommended for integration as the combined people/PPE detector on this footage**. Runtime and memory are practical, but no people were detected in six frames where people are visibly present. Helmet and vest detections are incomplete. No missing-equipment, compliance, or violation conclusions were produced.

## Working app preserved

`app.py`, `video_processing.py`, and `.streamlit/config.toml` remain byte-for-byte unchanged. Before/after SHA-256 values:

| File | SHA-256 |
| --- | --- |
| app.py | `656ea98b3af8e461c6eddac9a4961620789f6d2b01700ac99fed5b9d901c7c5c` |
| video_processing.py | `aa106da07a8187f55a3291a376e9aae35367b31c38de269872586818223b7b24` |
| .streamlit/config.toml | `bc33d6efb8d6b2ac795048ba9239e9b9b95687a49e2043e0a4fa0d5f00c2b77d` |

Evaluation code lives in `evaluation/` and is not imported by Streamlit. The existing virtual environment gained CPU evaluation dependencies; existing app package versions were retained. `.gitignore` excludes model binaries and incomplete downloads.

## Source, revision, integrity and license

- Publisher repository: [ayushgupta7777/safetyvision-yolov8](https://huggingface.co/ayushgupta7777/safetyvision-yolov8).
- Pinned model repository revision: `56a71758b55f0e9f2b4b2d6b51a779a1f882da10`.
- Repository file: `v2/best_640.onnx`.
- [Pinned download](https://huggingface.co/ayushgupta7777/safetyvision-yolov8/resolve/56a71758b55f0e9f2b4b2d6b51a779a1f882da10/v2/best_640.onnx).
- [Publisher file/hash page](https://huggingface.co/ayushgupta7777/safetyvision-yolov8/blob/56a71758b55f0e9f2b4b2d6b51a779a1f882da10/v2/best_640.onnx).
- Downloaded size: **44,764,727 bytes** (42.69 MiB).
- Publisher SHA-256: `ea18ae903a566e8fa76f3ee1c503075522dca269269315e9c862efa170430b35`.
- Independently calculated complete-file SHA-256: `ea18ae903a566e8fa76f3ee1c503075522dca269269315e9c862efa170430b35` — **MATCH**.
- Local weights: `models/safetyvision-v2-640.onnx`; machine-readable provenance: [manifest](../models/manifest.json).
- Model card and actual embedded metadata both declare **AGPL-3.0**. ONNX Runtime is [MIT licensed](https://github.com/microsoft/onnxruntime/blob/main/LICENSE). ONNX conversion does not change the weight license. Dataset rights have not been independently audited.
- Original implementation pinned at `3fba9d358321b14725ee6d329dd2bc99c89c8320`: [core/detector.py](https://github.com/ayushgupta07xx/SafetyVision/blob/3fba9d358321b14725ee6d329dd2bc99c89c8320/core/detector.py).
- A reference copy is retained as `evaluation/reference/detector.py.txt`, with its original AGPL license in `evaluation/reference/LICENSE.txt`. Adapted evaluation preprocessing/decoding carries AGPL attribution. Upstream violation logic is not used.

The downloader streams to a partial file, compares the entire SHA-256 against the fixed publisher value, and only promotes matching weights. The evaluator hashes the model again before constructing an inference session. A mismatch stops evaluation.

## Actual model contract

Inspected by loading the verified file through ONNX Runtime, rather than inferred from its filename:

| Property | Actual value |
| --- | --- |
| Input name | `images` |
| Input shape/type | `[1, 3, 640, 640]`, `tensor(float)` / FP32 |
| Output name | `output0` |
| Output shape/type | `[1, 17, 8400]`, `tensor(float)` / FP32 |
| Active provider | `CPUExecutionProvider` only |
| Embedded task/version | `detect`, Ultralytics `8.4.51` |
| Embedded export flags | batch 1, half false, dynamic false, nms false, end2end false |

Complete actual class metadata:

```text
0 Fall-Detected
1 Gloves
2 Goggles
3 Hardhat
4 Mask
5 NO-Gloves
6 NO-Goggles
7 NO-Hardhat
8 NO-Mask
9 NO-Safety Vest
10 No_Harness
11 Person
12 Safety Vest
```

Only IDs **3, 11, 12** are emitted or annotated. The network inherently produces 13 class scores; the other categories are excluded from evaluation results. No missing-equipment logic or equipment-to-person matching is performed.

## Preprocessing and decoding verification

The pinned source's `_letterbox`, `_preprocess`, and `_postprocess` methods were inspected directly. Focused tests execute only those three reviewed methods from the saved reference to compare numerical results, without importing the original application or its violation logic.

1. Frames are resized to width 640, preserving aspect ratio, with OpenCV area interpolation, matching milestone 1's working resolution. No JPEG encode/decode occurs before inference.
2. BGR is converted to RGB. A 640-square letterbox uses linear interpolation and gray padding `(114,114,114)`, with rounded resized dimensions and floor-half top/left padding, exactly as in the source.
3. Pixels become contiguous FP32 NCHW divided by 255. For this video, 640x360 becomes 640x640 with 140 pixels of padding above and below.
4. The raw output is transposed from channel-first predictions. Its first four channels are center-x, center-y, width, height; the remaining 13 are class scores. There is **no separate objectness multiplication, sigmoid, or embedded NMS** in our decoder.
5. As in the source, the winning class is selected across all 13 scores before filtering to the three requested classes. This prevents an excluded-class prediction being reassigned to a target category.
6. Confidence is strictly greater than **0.40** and NMS IoU is **0.45**, matching source defaults. Intentional difference: primary evaluation uses **per-class NMS**, so overlapping people and PPE do not suppress one another. A class-agnostic NMS comparison over the same selected predictions produced identical counts on all six frames. This comparison is not a full reproduction of upstream all-class NMS.
7. Coordinates subtract letterbox padding, divide by scale, and clip to sampled-image boundaries; empty boxes are discarded. Coordinates in JSON refer to the 640x360 sampled image.

Tests cover pixel-exact reference preprocessing for landscape, portrait and odd dimensions; known-coordinate mapping and reference score parity; clipping; same-class duplicate suppression; preservation of different overlapping classes; excluded-class filtering; empty outputs and invalid shapes/non-finite values.

## Video and sampling

- User-specified source: `C:\Users\ASUS\Downloads\12098511-hd_1920_1080_50fps.mp4`.
- SHA-256: `bed96770c9ae7b482106bc6efd91998c709a19e1c068b80ec37165e2388a36d1`.
- Actual decoded metadata: **1280x720, 50 FPS, 736 frames, 14.72 seconds**. The filename's 1920x1080 text does not match actual metadata.
- Six evenly spaced frame indices: 0, 147, 294, 441, 588, 735. Decoded sequentially; only one image is inferred at a time.
- Estimated timestamps use frame index / nominal FPS, consistent with milestone 1.

## Detections and visual review

| Timestamp | Person | Hardhat | Safety Vest | Annotated image |
| --- | ---: | ---: | ---: | --- |
| 0.00 s | 0 | 2 | 1 | [Frame 0](../evaluation/results/frame_000000_0.000s.jpg) |
| 2.94 s | 0 | 2 | 0 | [Frame 147](../evaluation/results/frame_000147_2.940s.jpg) |
| 5.88 s | 0 | 2 | 1 | [Frame 294](../evaluation/results/frame_000294_5.880s.jpg) |
| 8.82 s | 0 | 2 | 0 | [Frame 441](../evaluation/results/frame_000441_8.820s.jpg) |
| 11.76 s | 0 | 1 | 0 | [Frame 588](../evaluation/results/frame_000588_11.760s.jpg) |
| 14.70 s | 0 | 3 | 0 | [Frame 735](../evaluation/results/frame_000735_14.700s.jpg) |

All six annotated images were visually inspected. Multiple workers, helmets and high-visibility garments are visible. Some helmet and vest boxes align plausibly, but many visible objects are missed. These are repeated frame detections, not counts of unique people or unique equipment.

Diagnostic only: lowering confidence to 0.25 still produces **zero Person detections**. The maximum raw Person score across anchors is only about **0.00020–0.00025** per frame, so the failure occurs before NMS and is not explained by the 0.40 threshold. At 0.25, helmet counts are 5/4/4/3/4/5 and vest counts 2/1/2/2/1/0. These lower-threshold counts are recorded in JSON but are not the annotated baseline and are not validated accuracy claims.

**Decision:** do not integrate this checkpoint as the sole three-class detector. The cause of the extremely low person scores is unresolved; domain mismatch or an issue with the published checkpoint/export needs further investigation. A future comparison against its PyTorch checkpoint or a separately verified person detector would be a separate evaluation. No inference of absent equipment is justified.

Six correlated frames from one clip are not a labeled validation set. No precision, recall, mAP, false-positive rate, or site-wide performance is claimed.

## Measured timing and memory

Final reproducible run: Windows 10 build 19045, Python 3.12.4 x64, ONNX Runtime CPU, batch 1, sequential execution, intra-op threads 2, inter-op threads 1.

| Measurement | Value |
| --- | ---: |
| Model session construction | 248.68 ms |
| Cold inference (first `session.run` in fresh evaluation process) | 318.34 ms |
| Warm inference mean / median | 302.88 / 300.46 ms |
| Warm inference min / max | 286.01 / 333.13 ms |
| Warm calls | 23 |
| Baseline process RSS | 56.79 MiB |
| Process RSS after session load | 108.55 MiB |
| Sampled peak RSS | 286.00 MiB |
| Windows peak working set | 291.30 MiB |
| Baseline private bytes | 263.54 MiB |
| Sampled peak private bytes | 543.50 MiB |

Timings use `time.perf_counter()`. Four inference calls were made per frame: the first call on frame 0 is cold; the other 23 are warm. Session construction excludes Python imports and SHA verification. Inference timings measure `session.run` only, excluding decoding, resize, preprocessing, postprocessing and JPEG output; per-frame preprocessing/postprocessing times are retained in JSON. Cold means a fresh process/session, **not a cold Windows filesystem cache**: the file was downloaded/inspected earlier and hashed before loading.

Memory uses **psutil 7.2.2 `Process().memory_info()`**, polled every 20 ms in a background thread, plus phase snapshots. On Windows, `rss` is working-set resident memory, `private` is private committed memory (not necessarily resident), and `peak_wset` is the OS-maintained peak working set. MiB means bytes / 1,048,576. Polling can miss brief RSS/private peaks; the OS peak working-set counter supplements it. Measurements cover the evaluation Python process, including runtime, decoder and arrays; they exclude the separate Streamlit server, browser, OS, and other programs. Baseline is after imports/hash verification and before session creation. This is not a guaranteed memory ceiling for other videos.

An earlier run without the diagnostic-score fields had 208.01 ms load, 317.18 ms cold, and 302.34 ms warm median. The final run above supersedes it; both produced the same baseline detection counts.

## Environment and verification

Tested versions: `onnxruntime==1.29.0`, `psutil==7.2.2`, `flatbuffers==25.12.19`, `numpy==2.5.2`, `opencv-python-headless==5.0.0.93`, `streamlit==1.63.0`, Python 3.12.4. Full environment is pinned in [requirements.txt](../requirements.txt).

- **13 unittest checks passed**: 6 detection checks and 7 original milestone 1 checks.
- `pip check`: **No broken requirements found**.
- Actual CPU inference completed and six labeled JPEGs were saved.
- [Raw measurements, exact confidence scores, boxes and metadata](../evaluation/results/metrics.json).

## Reproduce in PowerShell

```powershell
Set-Location D:\Projects\safety-monitor
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe evaluation\download_model.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m evaluation.run_evaluation --video 'C:\Users\ASUS\Downloads\12098511-hd_1920_1080_50fps.mp4'
```

The evaluation command now writes JPEGs and metrics to a fresh timestamped directory under `evaluation/results/`; it does not update this narrative report automatically. It never changes the source video or Streamlit UI. Keep a copy of measurements if comparing subsequent runs.

## Addendum: pairing a separate person detector

Given this model's Person-detection failure above, the decision was made to keep this
model for Hardhat/Safety Vest only and pair it with a separately verified COCO person
detector (Ultralytics YOLOv8n) for locating people — see
[PROJECT_PLAN.md](PROJECT_PLAN.md) build order step 2.

- Source checkpoint: `yolov8n.pt`, official [`ultralytics/assets` release `v8.3.0`](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt).
- Downloaded and independently hashed: **6,549,796 bytes**, SHA-256
  `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`. Pinned in
  `evaluation/download_person_model.py`.
- License: AGPL-3.0, same family already accepted for the SafetyVision v2 weights above —
  no new license category introduced.
- The `.pt` checkpoint must be exported to ONNX locally (`pip install ultralytics`, then
  `python -m evaluation.download_person_model`); this needs torch and is **not** part of
  the app's runtime dependencies (`requirements.txt` is unchanged). The exported ONNX
  bytes are not bit-reproducible across torch/ultralytics versions, so unlike the model
  above, there is no fixed expected ONNX hash — `evaluation.detector.validate_person_session`
  checks the actual input/output shapes and, if present, embedded class names at runtime
  instead.
- `evaluation/run_evaluation.py` accepts `--person-model PATH` to run both models per frame
  and merge Person boxes (tagged `source: "person"`) with Hardhat/Safety Vest boxes (tagged
  `source: "ppe"`) into one annotated frame and one detections list.

### Real-footage run: person detector actually detects people

Date: 2026-09-06. Same source video as the table above
(`C:\Users\ASUS\Downloads\12098511-hd_1920_1080_50fps.mp4`, SHA-256
`bed96770c9ae7b482106bc6efd91998c709a19e1c068b80ec37165e2388a36d1`, 1280x720, 50 FPS, 736
frames, 14.72s), same six evenly spaced frame indices, both models run per frame via
`python -m evaluation.run_evaluation --video <path> --person-model models\yolov8n-person-640.onnx`.

| Timestamp | Hardhat | Safety Vest | Person (new) |
| --- | ---: | ---: | ---: |
| 0.00 s | 2 | 1 | 5 |
| 2.94 s | 2 | 0 | 5 |
| 5.88 s | 2 | 1 | 3 |
| 8.82 s | 2 | 0 | 4 |
| 11.76 s | 1 | 0 | 5 |
| 14.70 s | 3 | 0 | 4 |

Unlike SafetyVision v2 alone (zero Person detections, raw score ~0.0002 on every frame),
the paired YOLOv8n COCO detector produces 3-5 person detections per frame, consistent with
the multiple workers already visually confirmed present in these frames. **This resolves
the original blocker**: there is now a person box to anchor equipment-to-person matching
against. Annotated frames with the new person boxes are saved locally to
`evaluation/results/` on the machine that ran this command; visual confirmation that the
boxes land on the actual visible workers (not background) is a manual step still to be
recorded here once reviewed.

Actual model contract (verified via `validate_person_session` before trusting the file):
input `[1, 3, 640, 640]` FP32, output `[1, 84, 8400]` FP32 (4 box + 80 COCO classes),
`CPUExecutionProvider` only.

Measured timing and memory (Windows, same machine/config as the table above, Intel Core
i7-8565U, intra-op threads 2, inter-op threads 1):

| Measurement | SafetyVision v2 (Hardhat/Vest) | YOLOv8n (Person) |
| --- | ---: | ---: |
| Session load | 272.86 ms | 109.85 ms |
| Cold inference | 372.83 ms | 110.60 ms |
| Warm inference mean / median | 338.75 / 320.75 ms | 112.37 / 112.08 ms |
| Warm inference min / max | 285.41 / 498.53 ms | 102.45 / 149.33 ms |

Combined warm per-frame inference is roughly 451 ms (339 + 112) — less than double the
original single-model ~303 ms, since the person model is substantially lighter. Memory:
baseline RSS 59.24 MiB, after loading both models 130.17 MiB, sampled peak RSS 365.66 MiB
/ peak private 651.47 MiB (up from 286.00 MiB / 543.50 MiB with one model), final RSS
311.19 MiB. These reported process measurements exclude other applications and are not an 8 GB system memory guarantee.

**Decision: the pairing is viable for build-order step 3** (equipment-to-person matching),
pending the visual box-placement check above. Latency and memory both stayed well within
practical bounds for sequential, single-video CPU processing.

## Re-evaluate the corrected matching logic on Windows

Use the existing locally exported person model and original video:

```powershell
Set-Location D:\Projects\safety-monitor
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m evaluation.run_evaluation --video "C:\Users\ASUS\Downloads\12098511-hd_1920_1080_50fps.mp4" --person-model models\yolov8n-person-640.onnx
```

Review the printed results directory's images and `metrics.json` together. Confirm
box placement for each worker, missed detections, and ambiguous associations.
Do not compare an older JPEG with a newer metrics file. The original MP4 and ONNX
weights were not available in the review environment, so no new inference accuracy
or performance claim accompanies the matching correction.
