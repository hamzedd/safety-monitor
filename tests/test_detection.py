import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from evaluation.detector import COCO_NAMES, NAMES, PERSON_SELECTED, preprocess, decode


def reference_decoder():
    """Execute only the three reviewed math methods, not upstream app/download code."""
    path = Path(__file__).parents[1] / 'evaluation/reference/detector.py.txt'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PPEDetector')
    methods = [n for n in original.body if isinstance(n, ast.FunctionDef)
               and n.name in {'_letterbox', '_preprocess', '_postprocess'}]
    cls = ast.ClassDef(name='Reference', bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    namespace = {'np': np, 'cv2': cv2, 'Detection': SimpleNamespace}
    exec(compile(module, str(path), 'exec'), namespace)
    instance = namespace['Reference']()
    instance.img_size, instance.conf_threshold, instance.iou_threshold = 640, .4, .45
    instance.class_names = NAMES
    return instance


def output(*entries):
    result = np.zeros((1, 17, len(entries)), dtype=np.float32)
    for index, (box, cls, score) in enumerate(entries):
        result[0, :4, index] = box
        result[0, 4 + cls, index] = score
    return result


class DetectionTests(unittest.TestCase):
    def test_preprocessing_matches_pinned_source_landscape_portrait_odd(self):
        ref = reference_decoder()
        for shape in [(360, 640), (641, 301), (51, 97)]:
            image = np.random.default_rng(42).integers(0, 256, (*shape, 3), dtype=np.uint8)
            actual = preprocess(image)
            expected = ref._preprocess(image)
            np.testing.assert_array_equal(actual[0], expected[0])
            self.assertEqual(actual[1:], expected[1:])
            self.assertTrue(actual[0].flags.c_contiguous)

    def test_landscape_mapping_and_score_matches_original(self):
        # 640x360 -> 640 square: top pad 140. Known original box (100,50,300,250).
        raw = output(([200, 290, 200, 200], 11, .8))
        actual = decode(raw, 1., (0, 140), (360, 640))
        expected = reference_decoder()._postprocess(raw, 1., (0, 140), (360, 640))
        np.testing.assert_allclose(actual[0]['box'], [100, 50, 300, 250])
        np.testing.assert_allclose(actual[0]['box'], expected[0].bbox)
        self.assertAlmostEqual(actual[0]['confidence'], .8)
        self.assertEqual(actual[0]['name'], expected[0].cls)

    def test_portrait_mapping_and_clipping(self):
        raw = output(([320, 320, 640, 640], 3, .9))
        result = decode(raw, .5, (160, 0), (1280, 640))
        np.testing.assert_allclose(result[0]['box'], [0, 0, 640, 1280])

    def test_nms_same_class_suppressed_different_class_preserved(self):
        raw = output(([200, 200, 100, 100], 11, .9),
                     ([201, 200, 100, 100], 11, .8),
                     ([200, 200, 100, 100], 12, .85))
        result = decode(raw, 1, (0, 0), (640, 640))
        self.assertEqual([d['class_id'] for d in result], [11, 12])
        self.assertEqual(len(decode(raw, 1, (0, 0), (640, 640), class_aware=False)), 1)

    def test_excluded_class_winner_not_reassigned(self):
        raw = output(([100, 100, 50, 50], 7, .95))
        raw[0, 4 + 3, 0] = .8
        self.assertEqual(decode(raw, 1, (0, 0), (640, 640)), [])

    def test_empty_invalid_shape_and_nonfinite(self):
        self.assertEqual(decode(output(), 1, (0, 0), (640, 640)), [])
        for raw in [np.zeros((1, 8400, 17)), np.full((1, 17, 1), np.nan)]:
            with self.assertRaises(ValueError):
                decode(raw, 1, (0, 0), (640, 640))

    def test_decode_generalizes_to_a_different_class_count_and_selection(self):
        # 84-channel COCO-shaped output (4 box + 80 classes); only class 0 ("person") selected.
        result = np.zeros((1, 84, 2), dtype=np.float32)
        result[0, :4, 0] = [200, 200, 100, 100]
        result[0, 4, 0] = .9  # person
        result[0, :4, 1] = [400, 400, 50, 50]
        result[0, 4 + 2, 1] = .95  # car: not in PERSON_SELECTED, must be dropped
        detections = decode(result, 1, (0, 0), (640, 640), names=COCO_NAMES, selected=PERSON_SELECTED)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]['class_id'], 0)
        self.assertEqual(detections[0]['name'], 'person')
        np.testing.assert_allclose(detections[0]['box'], [150, 150, 250, 250])
        with self.assertRaises(ValueError):
            decode(np.zeros((1, 17, 1), dtype=np.float32), 1, (0, 0), (640, 640),
                   names=COCO_NAMES, selected=PERSON_SELECTED)

    def test_safetyvision_defaults_unchanged_by_generalization(self):
        raw = output(([200, 200, 100, 100], 3, .7))
        self.assertEqual(decode(raw, 1, (0, 0), (640, 640))[0]['name'], NAMES[3])
