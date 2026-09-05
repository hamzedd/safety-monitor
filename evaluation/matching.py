"""Conservative spatial equipment association for human review.

Fixed head/torso zones are only approximations for upright people. They do not
establish visibility or pose; absence of a match always remains uncertain.
"""

MIN_PERSON_HEIGHT = 40  # pixels in the ~640-wide sampled image; below this, too small/far to assess
HEAD_FRACTION = (0.0, 0.25)  # top 25% of the person box height
TORSO_FRACTION = (0.20, 0.65)  # 20%-65% down the person box height
OVERLAP_THRESHOLD = 0.5  # fraction of the equipment box that must fall inside the target zone

HARDHAT_CLASS_ID = 3
VEST_CLASS_ID = 12


def _zone(person_box, fraction):
    x1, y1, x2, y2 = person_box
    height = y2 - y1
    top, bottom = fraction
    return [x1, y1 + top * height, x2, y1 + bottom * height]


def _overlap_ratio(box, zone):
    ix1, iy1 = max(box[0], zone[0]), max(box[1], zone[1])
    ix2, iy2 = min(box[2], zone[2]), min(box[3], zone[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    area = (box[2] - box[0]) * (box[3] - box[1])
    return ((ix2 - ix1) * (iy2 - iy1)) / area if area > 0 else 0.0


def _assign(equipment, zones):
    """Accept only unambiguous one-to-one candidate pairs, independent of list order.

    Multiple candidate people OR multiple candidate equipment boxes are unresolved.
    No greedy tie-break can turn that ambiguity into a confident association.
    """
    candidates = [[j for j, zone in enumerate(zones)
                   if zone is not None and _overlap_ratio(det['box'], zone) >= OVERLAP_THRESHOLD]
                  for det in equipment]
    reverse = [[i for i, people in enumerate(candidates) if j in people]
               for j in range(len(zones))]
    return {j: items[0] for j, items in enumerate(reverse)
            if len(items) == 1 and len(candidates[items[0]]) == 1}


def match_people_to_equipment(detections):
    """Return positive spatial associations and uncertainty, never inferred absence.

    Bounding boxes cannot establish head/torso visibility. Therefore all unmatched
    people remain uncertain, including cropped, bent, occluded and small workers.
    A detected association is a spatial estimate, not proof of wearing equipment.
    Unmatched equipment remains available for review.
    """
    people = [d for d in detections if d.get('source') == 'person']
    hardhats = [d for d in detections if d.get('source') == 'ppe' and d['class_id'] == HARDHAT_CLASS_ID]
    vests = [d for d in detections if d.get('source') == 'ppe' and d['class_id'] == VEST_CLASS_ID]
    # Include small people in candidate competition so their equipment cannot be
    # confidently assigned to a larger overlapping worker instead.
    head_matches = _assign(hardhats, [_zone(p['box'], HEAD_FRACTION) for p in people])
    vest_matches = _assign(vests, [_zone(p['box'], TORSO_FRACTION) for p in people])
    results, used_hardhats, used_vests = [], set(), set()
    for j, person in enumerate(people):
        box = person['box']
        result = dict(person_box=box, person_confidence=person['confidence'])
        small = box[3] - box[1] < MIN_PERSON_HEIGHT
        for name, equipment, matches, used in (
                ('hardhat', hardhats, head_matches, used_hardhats),
                ('vest', vests, vest_matches, used_vests)):
            index = matches.get(j)
            if not small and index is not None:
                used.add(index)
                result[name] = 'detected'
                result[name + '_detection'] = equipment[index]
                result[name + '_reason'] = 'unambiguous_spatial_association'
            else:
                result[name] = 'uncertain'
                result[name + '_reason'] = ('person_too_small' if small else
                    'no_unambiguous_match_visibility_unverified')
        results.append(result)
    return dict(people=results,
               unmatched_hardhats=[d for i, d in enumerate(hardhats) if i not in used_hardhats],
               unmatched_vests=[d for i, d in enumerate(vests) if i not in used_vests])
