import unittest
from evaluation.geometry import tile_boxes, merge_detections

class GeometryTests(unittest.TestCase):
    def test_tiles_cover_edges_and_stay_in_bounds(self):
        for width,height in [(1280,720),(300,200),(720,1280),(641,641)]:
            boxes=tile_boxes(width,height)
            self.assertEqual(min(b[0] for b in boxes),0)
            self.assertEqual(max(b[2] for b in boxes),width)
            self.assertEqual(max(b[3] for b in boxes),height)
            self.assertEqual(len(boxes),len(set(boxes)))
            for x1,y1,x2,y2 in boxes:
                self.assertTrue(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height)
                self.assertLessEqual(x2-x1,640)
                self.assertLessEqual(y2-y1,640)
    def test_merge_preserves_other_classes_and_is_order_independent(self):
        a=dict(source='ppe',class_id=3,confidence=.8,box=[0,0,20,20])
        b=dict(a,confidence=.6,box=[1,1,21,21])
        c=dict(a,source='person',class_id=0)
        self.assertEqual(merge_detections([a,b,c]),merge_detections([c,b,a]))
        self.assertEqual(len(merge_detections([a,b,c])),2)
    def test_invalid_tile_arguments(self):
        with self.assertRaises(ValueError): tile_boxes(0,20)
