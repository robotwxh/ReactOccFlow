"""Explicit IoU protocols adapted from OccFM's multi_step_MeanIou.

Source commit/hash: docs/provenance/evaluation_sources.json. Standard protocol
includes hallucinated classes in the denominator; legacy reproduces the source
rule setting IoU=1 whenever a class has no ground-truth voxels. Values are
fractions, not percentages. Dataset-scale evaluation must aggregate counts
before computing IoU rather than average these scene-level smoke reports.
"""
import numpy as np


def occupancy_metrics(prediction, target, *, visibility=None, num_classes=18,
                      empty_class=17, ignore_label=255, protocol='standard'):
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape:
        raise ValueError('prediction and target shapes must match')
    if protocol not in {'standard', 'legacy'}:
        raise ValueError('protocol must be standard or legacy')
    mask = target != ignore_label
    if visibility is not None:
        if np.shape(visibility) != target.shape:
            raise ValueError('visibility shape must match target')
        mask &= np.asarray(visibility, dtype=bool)
    pred, truth = prediction[mask], target[mask]
    if np.any((pred < 0) | (pred >= num_classes)) or np.any((truth < 0) | (truth >= num_classes)):
        raise ValueError('labels outside configured class range')
    seen = np.bincount(truth.astype(np.int64), minlength=num_classes)
    positive = np.bincount(pred.astype(np.int64), minlength=num_classes)
    correct = np.bincount(truth[truth == pred].astype(np.int64), minlength=num_classes)
    union = seen + positive - correct
    ious = []
    for category in range(num_classes):
        if protocol == 'legacy' and seen[category] == 0:
            ious.append(1.0)
        elif union[category] == 0:
            ious.append(None)
        else:
            ious.append(float(correct[category] / union[category]))
    evaluated = [value for i, value in enumerate(ious) if i != empty_class and value is not None]
    return {'protocol': protocol, 'mean_iou': float(np.mean(evaluated)) if evaluated else None,
            'class_iou': ious, 'valid_voxels': int(mask.sum()),
            'counts': {'seen': seen.tolist(), 'positive': positive.tolist(), 'correct': correct.tolist()}}


class OccupancyAccumulator:
    """Dataset-level count aggregation, including per-horizon standard/legacy IoU."""

    def __init__(self, times=6, num_classes=18, empty_class=17):
        if times < 1 or num_classes < 1 or not 0 <= empty_class < num_classes:
            raise ValueError('Invalid metric dimensions')
        self.times, self.num_classes, self.empty_class = times, num_classes, empty_class
        self.seen = np.zeros((times, num_classes), dtype=np.int64)
        self.positive = np.zeros_like(self.seen)
        self.correct = np.zeros_like(self.seen)
        self.valid_voxels = np.zeros(times, dtype=np.int64)
        self.windows = 0

    def update(self, prediction, target, *, visibility=None):
        """Consume one window with leading physical-time axis (no batch axis)."""
        prediction, target = np.asarray(prediction), np.asarray(target)
        if prediction.shape != target.shape or prediction.shape[0] != self.times:
            raise ValueError('Expected matching predictions with leading time axis')
        if visibility is not None and np.shape(visibility) != target.shape:
            raise ValueError('Visibility shape mismatch')
        for time in range(self.times):
            report = occupancy_metrics(prediction[time], target[time],
                visibility=None if visibility is None else visibility[time],
                num_classes=self.num_classes, empty_class=self.empty_class)
            self.seen[time] += report['counts']['seen']
            self.positive[time] += report['counts']['positive']
            self.correct[time] += report['counts']['correct']
            self.valid_voxels[time] += report['valid_voxels']
        self.windows += 1

    def state_dict(self):
        return {'seen': self.seen.copy(), 'positive': self.positive.copy(),
                'correct': self.correct.copy(), 'valid_voxels': self.valid_voxels.copy(),
                'windows': self.windows}

    def load_state_dict(self, state):
        for name in ['seen', 'positive', 'correct', 'valid_voxels']:
            value = np.asarray(state[name])
            if value.shape != getattr(self, name).shape or value.dtype.kind not in 'iu' or np.any(value < 0):
                raise ValueError(f'Invalid accumulator state: {name}')
            setattr(self, name, value.astype(np.int64, copy=True))
        if np.any(self.correct > self.seen) or np.any(self.correct > self.positive):
            raise ValueError('Invalid intersection counts')
        self.windows = int(state['windows'])
        if self.windows < 0:
            raise ValueError('Invalid window count')

    def compute(self):
        def summarize(seen, positive, correct, protocol):
            union = seen + positive - correct
            result = []
            for i in range(self.num_classes):
                if protocol == 'legacy' and seen[i] == 0:
                    result.append(1.0)
                elif union[i] == 0:
                    result.append(None)
                else:
                    result.append(float(correct[i] / union[i]))
            values = [x for i, x in enumerate(result) if i != self.empty_class and x is not None]
            return {'mean_iou': float(np.mean(values)) if values else None, 'class_iou': result}
        report = {'windows': self.windows, 'seen': self.seen.tolist(),
                  'positive': self.positive.tolist(), 'correct': self.correct.tolist(),
                  'valid_voxels': self.valid_voxels.tolist()}
        for protocol in ['standard', 'legacy']:
            summary = summarize(self.seen.sum(0), self.positive.sum(0), self.correct.sum(0), protocol)
            per_time = [summarize(self.seen[t], self.positive[t], self.correct[t], protocol) for t in range(self.times)]
            summary['per_time_mean_iou'] = [row['mean_iou'] for row in per_time]
            summary['per_time_class_iou'] = [row['class_iou'] for row in per_time]
            summary['mean_over_horizons'] = float(np.mean([row['mean_iou'] for row in per_time if row['mean_iou'] is not None])) if any(row['mean_iou'] is not None for row in per_time) else None
            report[protocol] = summary
        return report
