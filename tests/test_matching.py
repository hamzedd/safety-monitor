import unittest

from evaluation.matching import match_people_to_equipment


def person(box, confidence=.9):
    return dict(class_id=0, name='person', confidence=confidence, box=box, source='person')


def hardhat(box, confidence=.8):
    return dict(class_id=3, name='Hardhat', confidence=confidence, box=box, source='ppe')


def vest(box, confidence=.8):
    return dict(class_id=12, name='Safety Vest', confidence=confidence, box=box, source='ppe')


class MatchingTests(unittest.TestCase):
    def test_both_detected_when_boxes_land_in_head_and_torso_zones(self):
        # Person 100..300 tall (height 200): head zone y in [100,150], torso y in [140,230].
        people = [person([50, 100, 150, 300])]
        equipment = [hardhat([60, 105, 140, 145]), vest([55, 160, 145, 220])]
        result = match_people_to_equipment(people + equipment)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['hardhat'], 'detected')
        self.assertEqual(result[0]['vest'], 'detected')

    def test_possible_missing_when_no_equipment_at_all(self):
        people = [person([50, 100, 150, 300])]
        result = match_people_to_equipment(people)
        self.assertEqual(result[0]['hardhat'], 'possible_missing')
        self.assertEqual(result[0]['vest'], 'possible_missing')

    def test_too_small_person_is_uncertain_for_both_regardless_of_equipment(self):
        people = [person([50, 100, 70, 130])]  # height 30 < MIN_PERSON_HEIGHT
        equipment = [hardhat([50, 100, 70, 110]), vest([50, 115, 70, 125])]
        result = match_people_to_equipment(people + equipment)
        self.assertEqual(result[0]['hardhat'], 'uncertain')
        self.assertEqual(result[0]['vest'], 'uncertain')

    def test_head_cropped_at_frame_top_is_uncertain_not_missing(self):
        people = [person([50, 0, 150, 300])]  # box top touches y=0
        result = match_people_to_equipment(people)
        self.assertEqual(result[0]['hardhat'], 'uncertain')
        # Torso isn't checked for edge-cropping by this heuristic (documented limitation).
        self.assertEqual(result[0]['vest'], 'possible_missing')

    def test_equipment_outside_target_zone_does_not_count(self):
        # Vest-shaped box placed over the head zone must not satisfy the hardhat match.
        people = [person([50, 100, 150, 300])]
        equipment = [vest([60, 105, 140, 145])]
        result = match_people_to_equipment(people + equipment)
        self.assertEqual(result[0]['hardhat'], 'possible_missing')
        self.assertEqual(result[0]['vest'], 'possible_missing')

    def test_each_equipment_box_assigned_to_at_most_one_person(self):
        # Two people close together; only one real hardhat box, closer to the first person.
        people = [person([0, 100, 100, 300]), person([90, 100, 190, 300])]
        equipment = [hardhat([10, 105, 90, 145])]
        result = match_people_to_equipment(people + equipment)
        detected = [r['hardhat'] for r in result].count('detected')
        missing = [r['hardhat'] for r in result].count('possible_missing')
        self.assertEqual(detected, 1)
        self.assertEqual(missing, 1)

    def test_low_overlap_equipment_not_matched(self):
        # Hardhat box mostly below the head zone: overlap ratio should fall under threshold.
        people = [person([50, 100, 150, 300])]
        equipment = [hardhat([60, 140, 140, 220])]
        result = match_people_to_equipment(people + equipment)
        self.assertEqual(result[0]['hardhat'], 'possible_missing')
