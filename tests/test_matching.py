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
        self.assertEqual(len(result['people']), 1)
        self.assertEqual(result['people'][0]['hardhat'], 'detected')
        self.assertEqual(result['people'][0]['vest'], 'detected')
        self.assertEqual(result['unmatched_hardhats'], [])
        self.assertEqual(result['unmatched_vests'], [])

    def test_uncertain_when_visibility_and_equipment_are_unknown(self):
        people = [person([50, 100, 150, 300])]
        result = match_people_to_equipment(people)
        self.assertEqual(result['people'][0]['hardhat'], 'uncertain')
        self.assertEqual(result['people'][0]['vest'], 'uncertain')

    def test_too_small_person_is_uncertain_for_both_regardless_of_equipment(self):
        people = [person([50, 100, 70, 130])]  # height 30 < MIN_PERSON_HEIGHT
        equipment = [hardhat([50, 100, 70, 110]), vest([50, 115, 70, 125])]
        result = match_people_to_equipment(people + equipment)
        self.assertEqual(result['people'][0]['hardhat'], 'uncertain')
        self.assertEqual(result['people'][0]['vest'], 'uncertain')
        # A too-small person still leaves the equipment itself unmatched/surfaced.
        self.assertEqual(len(result['unmatched_hardhats']), 1)
        self.assertEqual(len(result['unmatched_vests']), 1)

    def test_head_cropped_at_frame_top_is_uncertain_not_missing(self):
        people = [person([50, 0, 150, 300])]  # box top touches y=0
        result = match_people_to_equipment(people)
        self.assertEqual(result['people'][0]['hardhat'], 'uncertain')
        # Torso visibility is also unverified.
        self.assertEqual(result['people'][0]['vest'], 'uncertain')

    def test_equipment_outside_target_zone_does_not_count(self):
        # Vest-shaped box placed over the head zone must not satisfy the hardhat match.
        people = [person([50, 100, 150, 300])]
        equipment = [vest([60, 105, 140, 145])]
        result = match_people_to_equipment(people + equipment)
        self.assertEqual(result['people'][0]['hardhat'], 'uncertain')
        self.assertEqual(result['people'][0]['vest'], 'uncertain')
        self.assertEqual(len(result['unmatched_vests']), 1)

    def test_each_equipment_box_assigned_to_at_most_one_person(self):
        # Two people close together; only one real hardhat box, closer to the first person.
        people = [person([0, 100, 100, 300]), person([90, 100, 190, 300])]
        equipment = [hardhat([10, 105, 90, 145])]
        result = match_people_to_equipment(people + equipment)
        detected = [r['hardhat'] for r in result['people']].count('detected')
        missing = [r['hardhat'] for r in result['people']].count('uncertain')
        self.assertEqual(detected, 1)
        self.assertEqual(missing, 1)
        self.assertEqual(result['unmatched_hardhats'], [])

    def test_low_overlap_equipment_not_matched_and_surfaced_as_unmatched(self):
        # Hardhat box mostly below the head zone: overlap ratio should fall under threshold.
        people = [person([50, 100, 150, 300])]
        equipment = [hardhat([60, 140, 140, 220])]
        result = match_people_to_equipment(people + equipment)
        self.assertEqual(result['people'][0]['hardhat'], 'uncertain')
        self.assertEqual(result['unmatched_hardhats'], equipment)

    def test_unmatched_equipment_from_an_undetected_person_is_surfaced(self):
        # No person boxes at all: a real hardhat detection must not vanish silently.
        equipment = [hardhat([60, 105, 140, 145])]
        result = match_people_to_equipment(equipment)
        self.assertEqual(result['people'], [])
        self.assertEqual(result['unmatched_hardhats'], equipment)

    def test_overlapping_people_are_ambiguous_in_either_order(self):
        a, b = person([0, 100, 100, 300]), person([50, 100, 150, 300])
        hat = hardhat([60, 105, 100, 145])
        for people in ([a, b], [b, a]):
            result = match_people_to_equipment(people + [hat])
            self.assertTrue(all(p['hardhat'] == 'uncertain' for p in result['people']))
            self.assertEqual(result['unmatched_hardhats'], [hat])

    def test_two_candidate_helmets_remain_unresolved(self):
        hats = [hardhat([60, 105, 100, 140]), hardhat([100, 105, 140, 140])]
        for equipment in (hats, hats[::-1]):
            result = match_people_to_equipment([person([50, 100, 150, 300])] + equipment)
            self.assertEqual(result['people'][0]['hardhat'], 'uncertain')
            self.assertEqual(len(result['unmatched_hardhats']), 2)

    def test_bent_or_cropped_worker_without_match_never_implies_absence(self):
        for box in ([10, 100, 200, 180], [10, 200, 100, 360], [10, 0, 100, 150]):
            result = match_people_to_equipment([person(box)])['people'][0]
            self.assertEqual((result['hardhat'], result['vest']), ('uncertain', 'uncertain'))

    def test_distinct_matches_are_invariant_to_person_order(self):
        a, b = person([0, 100, 100, 300]), person([200, 100, 300, 300])
        hats = [hardhat([10, 105, 90, 145]), hardhat([210, 105, 290, 145])]
        def normalized(people):
            return {tuple(p['person_box']): p['hardhat_detection']['box']
                    for p in match_people_to_equipment(people + hats)['people']}
        self.assertEqual(normalized([a,b]), normalized([b,a]))
