# Construction Site Safety Review — Milestone 1

Local Streamlit/OpenCV application for one MP4 upload, metadata, video playback and approximately one sampled frame per second. No detection models or AI results are included.

## Windows setup (PowerShell)

Run from the project directory. Activation is unnecessary, avoiding PowerShell execution-policy changes.

```powershell
Set-Location D:\Projects\safety-monitor
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The existing `.venv` can be reused; skip the creation command if it already exists. Requirements pin the tested environment, including transitive dependencies.

## Launch

```powershell
Set-Location D:\Projects\safety-monitor
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://127.0.0.1:8501. Stop the foreground server with Ctrl+C. The project configuration binds the server to localhost and caps uploads at 100 MiB (104,857,600 bytes), labelled 100 MB in the UI.

1. Upload one MP4; review duration, resolution, nominal FPS and playback.
2. Select **Process Video** and wait for progress to finish.
3. Review up to the first 12 sampled images and download all timestamps as JSON.

Changing the upload resets results. Ordinary UI reruns reuse session results and do not decode the video again. A browser refresh/new session may require uploading and processing again.

## Verification

```powershell
Set-Location D:\Projects\safety-monitor
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

Tests generate small MP4 fixtures locally and remove them afterward. They cover metadata, one-second sampling, fractional FPS, aspect-ratio resizing, no upscaling, progress, invalid uploads, size limits, temporary-file cleanup (including interrupted processing), and initial Streamlit rendering.

Verified on Windows with Python 3.12.4, Streamlit 1.63.0, OpenCV headless 5.0.0.93 and NumPy 2.5.2: all 7 unittest tests passed, and `pip check` reported no broken requirements. The Streamlit interaction test also verifies 14 samples produce only 12 retained previews, reruns do not repeat sampling, and replacing the upload with invalid data clears results. Upload widgets are mocked in that test; browser playback was not manually verified. Expected OpenCV corruption diagnostics may appear while invalid-file tests pass.

## Resource use and limitations

- CPU only. Frames are visited sequentially; source frames are never loaded into a full-video array. Each sample is resized to at most 640 pixels wide and JPEG encoded. Only 12 preview JPEGs and the timestamp list are retained; later images are discarded after the callback. Future detection can consume that callback incrementally.
- Streamlit holds the upload and playback data in memory. OpenCV still decodes at source resolution before resizing, so unusually large dimensions and very long videos can consume significant time/resources even below the upload limit. 8 GB is suitable for ordinary footage, not a guaranteed bound for arbitrary inputs.
- Timestamps and duration use nominal FPS/frame-count metadata; variable-frame-rate footage and inaccurate metadata can reduce accuracy. Sampling may miss events shorter than one second. Portrait videos retain their aspect ratio, so height can exceed 640.
- MP4 is a container: browser codec support differs from OpenCV. H.264 MP4 is generally suitable for browser preview; other codecs may sample successfully without browser playback. Unsupported, empty or corrupt videos show errors; some partial corruption cannot be identified reliably by OpenCV.
- Temporary input copies are removed immediately after inspection/processing, including normal exceptions. A forced process termination or power loss can leave a `safety-monitor-*` directory in the Windows temporary folder. No uploaded videos are written to the repository. Session previews disappear with the session; there is no persistent evidence storage.
- Processing is synchronous; wait for completion before replacing an upload. There is no cancellation UI in this milestone.

See [the project plan](docs/PROJECT_PLAN.md) for the future detection, matching and human review pipeline.

API references: [Streamlit upload limits](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader) and [session state](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state).
