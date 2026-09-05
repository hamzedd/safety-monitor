import hashlib
import json

import cv2
import streamlit as st

from video_processing import VideoError, inspect_video, sample_video, temporary_video

st.set_page_config(page_title='Construction Site Safety Review', layout='wide')
st.title('Construction Site Safety Review')
st.caption('Milestone 1 · Video upload and frame sampling · CPU only')
st.write('Upload one MP4 up to 100 MB. Sample approximately one frame per second for later human review.')
upload = st.file_uploader('Choose an MP4 video', type=['mp4'], accept_multiple_files=False)

if upload is None:
    st.session_state.pop('video', None)
    st.info('Upload a video to begin.')
    st.stop()

data = upload.getbuffer()
identity = hashlib.sha256(data).hexdigest()
if st.session_state.get('video', {}).get('identity') != identity:
    state = {'identity': identity, 'info': None, 'error': None, 'result': None}
    st.session_state.video = state
    try:
        with temporary_video(data, upload.name) as path:
            state['info'] = inspect_video(path)
    except (VideoError, OSError, cv2.error) as exc:
        state['error'] = f'Unable to read video: {exc}'

state = st.session_state.video
if state['error']:
    st.error(state['error'])
    st.stop()

info = state['info']
columns = st.columns(3)
columns[0].metric('Duration', f'{info.duration:.2f} seconds')
columns[1].metric('Resolution', f'{info.width} × {info.height}')
columns[2].metric('Frame rate', f'{info.fps:.3f} FPS')
st.video(upload.getvalue(), format='video/mp4')
st.caption('Browser playback depends on the MP4 codec. Sampling uses OpenCV. Timestamps are estimated from the nominal frame rate.')

if st.button('Process Video', type='primary', disabled=state['result'] is not None):
    progress = st.progress(0.0, text='Sampling video…')
    previews = []

    def retain_preview(timestamp, jpeg):
        if len(previews) < 12:
            previews.append((timestamp, jpeg))

    try:
        with temporary_video(data, upload.name) as path:
            timestamps = sample_video(path, info, retain_preview,
                                      lambda value: progress.progress(value, text='Sampling video…'))
        state['result'] = {'timestamps': timestamps, 'previews': previews}
        st.rerun()
    except (VideoError, OSError, cv2.error) as exc:
        progress.empty()
        st.error(f'Processing failed: {exc}')

if state['result'] is not None:
    result = state['result']
    st.success(f"Processing complete: {len(result['timestamps'])} frames sampled.")
    st.caption('Showing the first 12 sampled frames at most. Only these preview images are retained; every sample timestamp is retained.')
    columns = st.columns(3)
    for index, (timestamp, jpeg) in enumerate(result['previews']):
        columns[index % 3].image(jpeg, caption=f'{timestamp:.3f} s', width='stretch')
    st.download_button('Download sample timestamps (JSON)',
                       json.dumps({'timestamp_basis': 'frame index / nominal FPS',
                                   'timestamps_seconds': result['timestamps']}, indent=2),
                       file_name='sample_timestamps.json', mime='application/json')
