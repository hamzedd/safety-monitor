"""Dependency-free tiling and class-aware duplicate suppression."""

def tile_boxes(width, height, size=640, overlap=0.25):
    if width <= 0 or height <= 0 or size <= 0 or not 0 <= overlap < 1:
        raise ValueError('Invalid tiling dimensions or overlap')
    def starts(length):
        end = max(0, length-size)
        return sorted(set(list(range(0, end+1, max(1, int(size*(1-overlap))))) + [end]))
    return [(x, y, min(x+size,width), min(y+size,height))
            for y in starts(height) for x in starts(width)]


def merge_detections(detections, threshold=.45):
    def iou(a,b):
        intersection = max(0,min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))
        union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
        return intersection/union if union > 0 else 0
    kept=[]
    for d in sorted(detections, key=lambda d: (-d['confidence'],d['source'],d['class_id'],tuple(d['box']))):
        if not any((d['source'],d['class_id']) == (k['source'],k['class_id']) and
                   iou(d['box'],k['box']) > threshold for k in kept):
            kept.append(d)
    return kept
