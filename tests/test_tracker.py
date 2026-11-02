"""Greedy IoU tracker on synthetic detections."""
from __future__ import annotations

from typing import Callable

import numpy as np

from lipread.tracker import track_faces
from lipread.types import Detection


def _det(frame: int, x: float, y: float = 40.0, size: float = 60.0) -> Detection:
    pts = np.zeros((5, 2))
    return Detection(frame=frame, box=(x, y, size, size), score=0.9, points5=pts)


def _frames(n: int, faces: dict[str, Callable[[int], float]], missing: dict[str, set[int]] | None = None) -> list[list[Detection]]:
    """Build dets_per_frame: each named face moves via its x(frame) function, minus missed frames."""
    missing = missing or {}
    out: list[list[Detection]] = []
    for f in range(n):
        dets = []
        for name, xfun in faces.items():
            if f in missing.get(name, set()):
                continue
            dets.append(_det(f, xfun(f)))
        out.append(dets)
    return out


def test_two_faces_give_two_tracks_that_keep_their_identity():
    n = 30
    dets = _frames(n, {"A": lambda f: 10 + 0.5 * f, "B": lambda f: 200 - 0.5 * f})
    tracks = track_faces(dets, [(0, n - 1)])
    assert len(tracks) == 2
    assert [t.track_id for t in tracks] == [1, 2]
    for t in tracks:
        assert t.shot_id == 0
        assert len(t.detections) == n
        assert t.start_frame == 0 and t.end_frame == n - 1
        assert t.num_frames == n
    # each track follows one face: x grows for the first, shrinks for the second
    xs = [[d.box[0] for d in t.detections] for t in tracks]
    assert xs[0][-1] > xs[0][0]
    assert xs[1][-1] < xs[1][0]


def test_gap_within_max_gap_keeps_one_track():
    n = 30
    dets = _frames(n, {"A": lambda f: 50.0}, missing={"A": set(range(10, 15))})   # 5 missed frames
    tracks = track_faces(dets, [(0, n - 1)], max_gap=8)
    assert len(tracks) == 1
    t = tracks[0]
    assert len(t.detections) == n - 5
    assert t.start_frame == 0 and t.end_frame == n - 1
    assert t.num_frames == n                                 # the gap is inside the track
    frames = [d.frame for d in t.detections]
    assert 12 not in frames and 9 in frames and 15 in frames


def test_gap_longer_than_max_gap_splits_the_track():
    n = 40
    dets = _frames(n, {"A": lambda f: 50.0}, missing={"A": set(range(10, 22))})   # 12 missed frames
    tracks = track_faces(dets, [(0, n - 1)], max_gap=8)
    assert len(tracks) == 2
    assert tracks[0].end_frame == 9
    assert tracks[1].start_frame == 22


def test_shot_cut_resets_tracks_even_when_the_face_is_in_place():
    shots = [(0, 9), (10, 19)]
    dets = _frames(20, {"A": lambda f: 50.0})        # same face, same place, across the cut
    tracks = track_faces(dets, shots)
    assert len(tracks) == 2
    assert [t.shot_id for t in tracks] == [0, 1]
    assert tracks[0].end_frame == 9 and tracks[1].start_frame == 10
    assert [t.track_id for t in tracks] == [1, 2]


def test_no_detections_gives_no_tracks():
    assert track_faces([[] for _ in range(20)], [(0, 19)]) == []


def test_frames_beyond_detection_list_count_as_empty():
    dets = _frames(10, {"A": lambda f: 50.0})
    tracks = track_faces(dets, [(0, 19)], max_gap=2)     # shot runs past the list: frames 10..19 empty
    assert len(tracks) == 1
    assert tracks[0].end_frame == 9


def _box_det(frame: int, x: float, y: float, size: float) -> Detection:
    return Detection(frame=frame, box=(x, y, size, size), score=0.9, points5=np.zeros((5, 2)))


def test_centre_distance_fallback_links_a_fast_pan():
    """Low IoU after a jump (the face also shrinks to 40 px), but the centre moved less than half
    the track's height and the size ratio is in range: the fallback keeps one track."""
    # track: 60 px box, centre (130, 70). next: 40 px box, centre (155, 70): IoU ~0.24, distance 25 < 30
    dets = [[_box_det(0, 100.0, 40.0, 60.0)], [_box_det(1, 135.0, 50.0, 40.0)]]
    tracks = track_faces(dets, [(0, 1)], iou_thresh=0.3)
    assert len(tracks) == 1
    assert len(tracks[0].detections) == 2


def test_centre_fallback_rejects_a_size_jump_outside_the_ratio():
    """Same centre distance, but a 20 px box next to a 60 px one is not the same face."""
    dets = [[_box_det(0, 100.0, 40.0, 60.0)], [_box_det(1, 140.0, 60.0, 20.0)]]   # centre 20 px away
    tracks = track_faces(dets, [(0, 1)], iou_thresh=0.3)
    assert len(tracks) == 2


def test_far_jump_is_not_linked():
    dets = [[_det(0, 100.0)], [_det(1, 400.0)]]
    tracks = track_faces(dets, [(0, 1)], iou_thresh=0.3)
    assert len(tracks) == 2


def test_tracks_are_sorted_by_start_then_id():
    dets = [[_det(0, 10.0)], [_det(1, 10.0), _det(1, 300.0)], [_det(2, 300.0)]]
    tracks = track_faces(dets, [(0, 2)], max_gap=0)
    keys = [(t.start_frame, t.track_id) for t in tracks]
    assert keys == sorted(keys)
