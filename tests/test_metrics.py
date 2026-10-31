import numpy as np
import pytest


def test_iou_ignores_unknown_and_handles_false_positive_classes():
    from reactoccflow.evaluation.occupancy import occupancy_metrics
    target = np.array([0, 0, 1, 17, 255])
    pred = np.array([0, 1, 1, 2, 0])
    report = occupancy_metrics(pred, target)
    # class0: 1/2; class1:1/2; hallucinated class2:0/1; absent classes omitted.
    assert report['mean_iou'] == pytest.approx(1 / 3)
    assert report['valid_voxels'] == 4
    assert report['class_iou'][3] is None
    legacy = occupancy_metrics(pred, target, protocol='legacy')
    assert legacy['class_iou'][2] == 1.0
    assert legacy['mean_iou'] > report['mean_iou']


def test_visibility_masks_exclude_unobserved_labels():
    from reactoccflow.evaluation.occupancy import occupancy_metrics
    report = occupancy_metrics(np.array([0, 1]), np.array([0, 0]), visibility=np.array([True, False]))
    assert report['mean_iou'] == 1.0
    assert report['valid_voxels'] == 1


def test_dataset_iou_aggregates_counts_instead_of_averaging_window_scores():
    from reactoccflow.evaluation.occupancy import OccupancyAccumulator
    acc = OccupancyAccumulator(times=1)
    acc.update(np.array([[0]]), np.array([[0]]))
    acc.update(np.ones((1, 9), dtype=int), np.zeros((1, 9), dtype=int))
    report = acc.compute()
    # class0=1/10, class1=0/9; average .05, not mean(window mIoUs)=.5.
    assert report['standard']['mean_iou'] == pytest.approx(.05)
    assert report['seen'][0][0] == 10
    assert report['windows'] == 2


def test_accumulator_keeps_forecast_horizons_and_roundtrips_state():
    from reactoccflow.evaluation.occupancy import OccupancyAccumulator
    acc = OccupancyAccumulator(times=2)
    acc.update(np.array([[0, 1], [0, 0]]), np.array([[0, 1], [1, 1]]))
    state = acc.state_dict()
    restored = OccupancyAccumulator(times=2)
    restored.load_state_dict(state)
    report = restored.compute()
    assert report['standard']['per_time_mean_iou'] == [1.0, 0.0]
    assert report['valid_voxels'] == [2, 2]
