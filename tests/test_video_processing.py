import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest

from video_processing import MAX_UPLOAD_BYTES, VideoError, inspect_video, sample_video, temporary_video


class VideoTests(unittest.TestCase):
    def make_video(self, path, fps=10, count=35, size=(800, 400)):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        self.assertTrue(writer.isOpened())
        try:
            for index in range(count):
                frame = np.full((size[1], size[0], 3), index % 255, dtype=np.uint8)
                writer.write(frame)
        finally:
            writer.release()

    def test_sampling_metadata_resize_timestamps_progress_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.mp4'
            self.make_video(source)
            with temporary_video(source.read_bytes(), 'source.mp4') as path:
                info = inspect_video(path)
                self.assertEqual((info.width, info.height), (800, 400))
                self.assertAlmostEqual(info.duration, 3.5)
                samples, progress = [], []
                timestamps = sample_video(path, info, lambda t, b: samples.append((t, b)), progress.append)
                self.assertEqual(timestamps, [0.0, 1.0, 2.0, 3.0])
                for _, jpeg in samples:
                    decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                    self.assertEqual(decoded.shape[:2], (320, 640))
                self.assertEqual(progress[-1], 1.0)
                self.assertEqual(progress, sorted(progress))
            self.assertFalse(path.exists())

    def test_fractional_fps_and_no_upscale(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fractional.mp4'
            self.make_video(path, fps=29.97, count=65, size=(160, 80))
            info = inspect_video(path)
            samples = []
            timestamps = sample_video(path, info, lambda t, b: samples.append(b))
            self.assertEqual(len(timestamps), 3)
            for target, timestamp in enumerate(timestamps):
                self.assertLessEqual(abs(timestamp - target), 1 / info.fps)
            frame = cv2.imdecode(np.frombuffer(samples[0], np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(frame.shape[:2], (80, 160))

    def test_invalid_video_and_cleanup_on_failure(self):
        path = None
        with self.assertRaises(VideoError):
            with temporary_video(b'This is not an MP4', 'invalid.mp4') as path:
                inspect_video(path)
        self.assertIsNotNone(path)
        self.assertFalse(path.exists())

    def test_empty_wrong_extension_and_size_limit(self):
        class Oversized:
            def __len__(self):
                return MAX_UPLOAD_BYTES + 1
        for data, name in [(b'', 'empty.mp4'), (b'x', 'wrong.avi'), (Oversized(), 'large.mp4')]:
            with self.subTest(name=name), self.assertRaises(VideoError):
                with temporary_video(data, name):
                    self.fail('Invalid upload was accepted')

    def test_callback_failure_cleans_up(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.mp4'
            self.make_video(source, count=1)
            def fail(*args):
                raise RuntimeError('Interrupted processing')
            with self.assertRaises(RuntimeError):
                with temporary_video(source.read_bytes(), 'source.mp4') as path:
                    sample_video(path, inspect_video(path), fail)
            self.assertFalse(path.exists())

    def test_app_initial_render(self):
        app = AppTest.from_file(str(Path(__file__).parents[1] / 'app.py')).run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn('Upload a video', app.info[0].value)

    def test_app_process_preview_limit_and_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'long.mp4'
            self.make_video(path, fps=2, count=28, size=(160, 80))
            upload = BytesIO(path.read_bytes())
            upload.name = 'long.mp4'
            with patch('streamlit.file_uploader', return_value=upload), patch(
                'video_processing.sample_video', wraps=sample_video
            ) as sampler:
                app = AppTest.from_file(str(Path(__file__).parents[1] / 'app.py')).run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(sampler.call_count, 0)
                app.button[0].click().run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(sampler.call_count, 1)
                result = app.session_state['video']['result']
                self.assertEqual(len(result['timestamps']), 14)
                self.assertEqual(len(result['previews']), 12)
                app.run()
                self.assertEqual(sampler.call_count, 1)
                self.assertTrue(app.button[0].disabled)
                invalid = BytesIO(b'not video')
                invalid.name = 'invalid.mp4'
                with patch('streamlit.file_uploader', return_value=invalid):
                    app.run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(len(app.error), 1)
                self.assertIsNone(app.session_state['video']['result'])


if __name__ == '__main__':
    unittest.main()
