"""Person-to-equipment spatial matching. Independent implementation, not derived from the
pinned upstream reference (evaluation/reference/detector.py.txt) or any violation logic —
see evaluation/reference/LICENSE.txt for that reference's own terms.

Boxes only give a person's overall extent, not pose/keypoints, so "head" and "torso" are
approximated as fixed fractions of each person box's height. This assumes a roughly
upright, fully visible figure: it does not model crouching, bending, or genuine occlusion
(e.g. a hand over the head) — those cases fall back to "possible_missing" rather than
"uncertain" because there is no signal here to distinguish them. Only two visibility
signals are used to downgrade to uncertain: a very small person box (too far/blurry to
trust either way) and a box whose top touches the frame edge (head likely cropped out of
frame). A missing detection is never itself treated as proof of absence.
"""

MIN_PERSON_HEIGHT = 40  # pixels in the ~640-wide sampled image; below this, too small/far to assess
EDGE_MARGIN = 2  # pixels; a person box top within this of the frame edge may have a cropped head
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


def _best_match(equipment, zone, used):
    best_index, best_ratio = None, 0.0
    for index, det in enumerate(equipment):
        if index in used:
            continue
        ratio = _overlap_ratio(det['box'], zone)
        if ratio > best_ratio:
            best_index, best_ratio = index, ratio
    return (best_index, best_ratio) if best_index is not None and best_ratio >= OVERLAP_THRESHOLD else (None, 0.0)


def match_people_to_equipment(detections):
    """Classify Hardhat/Safety Vest per person as 'detected', 'possible_missing', or
    'uncertain'. `detections` is the merged ppe+person list run_evaluation.py already
    produces (each tagged with 'source' and, for ppe, 'class_id').

    Returns a dict with 'people' (one entry per detected person; each equipment box is
    matched to at most one person) and 'unmatched_hardhats'/'unmatched_vests' — real
    detections that overlapped no person's zone by >=OVERLAP_THRESHOLD. Unmatched
    equipment is surfaced rather than silently dropped: it means a hardhat/vest WAS
    detected somewhere in the frame, but couldn't be confidently linked to a specific
    person (e.g. the person detector missed that worker, or the box geometry didn't land
    inside the estimated head/torso zone) — a different failure mode than "nothing was
    detected at all", and one that should not be read as "possible_missing" for anyone."""
    people = [d for d in detections if d.get('source') == 'person']
    hardhats = [d for d in detections if d.get('source') == 'ppe' and d['class_id'] == HARDHAT_CLASS_ID]
    vests = [d for d in detections if d.get('source') == 'ppe' and d['class_id'] == VEST_CLASS_ID]

    results = []
    used_hardhats, used_vests = set(), set()
    for person in people:
        box = person['box']
        result = dict(person_box=box, person_confidence=person['confidence'])
        if (box[3] - box[1]) < MIN_PERSON_HEIGHT:
            result['hardhat'] = 'uncertain'
            result['vest'] = 'uncertain'
            results.append(result)
            continue

        hardhat_index, _ = _best_match(hardhats, _zone(box, HEAD_FRACTION), used_hardhats)
        if hardhat_index is not None:
            used_hardhats.add(hardhat_index)
            result['hardhat'] = 'detected'
            result['hardhat_detection'] = hardhats[hardhat_index]
        elif box[1] <= EDGE_MARGIN:
            result['hardhat'] = 'uncertain'
        else:
            result['hardhat'] = 'possible_missing'

        vest_index, _ = _best_match(vests, _zone(box, TORSO_FRACTION), used_vests)
        if vest_index is not None:
            used_vests.add(vest_index)
            result['vest'] = 'detected'
            result['vest_detection'] = vests[vest_index]
        else:
            result['vest'] = 'possible_missing'

        results.append(result)
    return dict(people=results,
               unmatched_hardhats=[det for index, det in enumerate(hardhats) if index not in used_hardhats],
               unmatched_vests=[det for index, det in enumerate(vests) if index not in used_vests])
