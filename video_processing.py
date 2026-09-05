"""Incremental CPU video sampling; no detection or persistent video storage."""
from contextlib import contextmanager
from dataclasses import dataclass
import math
from pathlib import Path
import tempfile

import cv2

MAX_UPLOAD_BYTES = 100 * 1024 * 1024


class VideoError(ValueError):
    pass


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration(self):
        return self.frame_count / self.fps


@contextmanager
def temporary_video(data, filename):
    if Path(filename).suffix.lower() != '.mp4':
        raise VideoError('Please upload an MP4 file.')
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise VideoError('Video must be nonempty and no larger than 100 MB (100 MiB).')
    # A closed file is required before OpenCV can open it on Windows.
    with tempfile.TemporaryDirectory(prefix='safety-monitor-') as directory:
        path = Path(directory) / 'upload.mp4'
        path.write_bytes(data)
        yield path


def inspect_video(path):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise VideoError('Cannot open video. It may be corrupt or use an unsupported codec.')
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or fps <= 0 or not math.isfinite(count) or count < 1:
            raise VideoError('Video has invalid frame rate or duration metadata.')
        ok, frame = capture.read()
        if not ok or frame is None:
            raise VideoError('Video contains no decodable frames.')
        height, width = frame.shape[:2]
        return VideoInfo(width, height, fps, int(count))
    finally:
        capture.release()


def sample_video(path, info, on_sample=None, on_progress=None):
    """Visit frames sequentially, emitting the first frame at/after each second.

    Only the caller's callback decides whether to retain an emitted JPEG.
    Timestamps use frame index / nominal FPS (approximate for variable FPS).
    """
    capture = cv2.VideoCapture(str(path))
    timestamps = []
    next_second = 0
    index = 0
    try:
        if not capture.isOpened():
            raise VideoError('Cannot open video for processing.')
        while capture.grab():
            timestamp = index / info.fps
            if timestamp + 1e-9 >= next_second:
                ok, frame = capture.retrieve()
                if not ok or frame is None:
                    raise VideoError(f'Cannot decode sampled frame at {timestamp:.2f} seconds.')
                height, width = frame.shape[:2]
                if width > 640:
                    frame = cv2.resize(frame, (640, max(1, round(height * 640 / width))),
                                       interpolation=cv2.INTER_AREA)
                ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    raise VideoError('Could not encode sampled frame.')
                timestamps.append(timestamp)
                if on_sample:
                    on_sample(timestamp, jpeg.tobytes())
                next_second = math.floor(timestamp) + 1
            index += 1
            if on_progress and (index % max(1, round(info.fps)) == 0):
                on_progress(min(index / info.frame_count, 0.99))
        if index == 0:
            raise VideoError('Video contains no decodable frames.')
        if index < info.frame_count - max(2, math.ceil(info.fps * 0.1)):
            raise VideoError('Video ended before its declared length; it may be truncated.')
        if on_progress:
            on_progress(1.0)
        return timestamps
    except cv2.error as exc:
        raise VideoError('OpenCV could not decode this video.') from exc
    finally:
        capture.release()
