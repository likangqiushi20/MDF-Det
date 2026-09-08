import torch

from hmrn.model import HMRN
from hmrn.tracking import Detection, TrackPoint, associate, decode


def test_hmrn_outputs_stride_four_heads():
    model = HMRN().eval()
    with torch.no_grad():
        output = model(torch.zeros(2, 3, 128, 160))
    assert output["center"].shape == (2, 1, 32, 40)
    assert output["motion"].shape == (2, 2, 32, 40)
    assert output["center"].min() >= 0
    assert output["center"].max() <= 1


def test_decode_keeps_local_maximum_and_rescales_motion():
    center = torch.zeros(1, 1, 8, 8)
    motion = torch.zeros(1, 2, 8, 8)
    center[0, 0, 2, 3] = 0.9
    center[0, 0, 2, 4] = 0.8
    motion[0, :, 2, 3] = torch.tensor([-1.0, 2.0])
    detections = decode(center, motion, 0.5)[0]
    assert len(detections) == 1
    assert (detections[0].x, detections[0].y) == (12, 8)
    assert (detections[0].dx, detections[0].dy) == (-4, 8)


def test_greedy_association_reuses_track_and_allocates_new_id():
    previous = [TrackPoint(7, 20, 30, 0.8)]
    detections = [
        Detection(24, 28, 0.9, -4, 2),
        Detection(100, 100, 0.7, 0, 0),
    ]
    tracks, next_id = associate(detections, previous, 10, radius=10)
    assert [track.track_id for track in tracks] == [7, 10]
    assert next_id == 11
