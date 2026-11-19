"""Independent annotation-oracle research labels in fixed current ego E0.

Trajectory and frame traversal adapted after copying IR-WM NuScenesTraj and
NuScenesWorldDatasetV1; occupancy path convention copied from OccFM's dataset.
Source commits, original SHA-256 hashes, and pre-adaptation snapshots are in
``docs/provenance/research_sources.json``. No nuScenes/MMCV runtime is needed.

Ground-truth annotations are an ORACLE representation, not sensor perception.
Future records are traversed only by build_sample, after build_observation has
completed actor selection, history features, current dimensions and classes.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from reactoccflow.geometry import (
    GridSpec, inverse_transform, quaternion_rotation, transform_matrix,
    transform_points, warp_occupancy_nearest,
)

OCC3D_CLASSES = (
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation', 'free',
)
# Explicit official fine-to-coarse Occ3D mapping. Unlisted official categories
# (animal, emergency vehicle, stroller/wheelchair, debris, etc.) map to others.
CATEGORY_TO_OCC3D = {
    'movable_object.barrier': 1,
    'vehicle.bicycle': 2,
    'vehicle.bus.bendy': 3,
    'vehicle.bus.rigid': 3,
    'vehicle.car': 4,
    'vehicle.construction': 5,
    'vehicle.motorcycle': 6,
    'human.pedestrian.adult': 7,
    'human.pedestrian.child': 7,
    'human.pedestrian.construction_worker': 7,
    'human.pedestrian.police_officer': 7,
    'movable_object.trafficcone': 8,
    'vehicle.trailer': 9,
    'vehicle.truck': 10,
    'flat.driveable_surface': 11,
    'flat.other': 12,
    'flat.sidewalk': 13,
    'flat.terrain': 14,
    'static.manmade': 15,
    'static.vegetation': 16,
}
ACTOR_CLASSES = frozenset({2, 3, 4, 5, 6, 7, 9, 10})
LABEL_VERSION = 1


def _json_records(path, chunk_size=1024 * 1024):
    """Stream a JSON array using stdlib; do not materialize sensor tables."""
    decoder = json.JSONDecoder()
    with Path(path).open() as stream:
        buffer, position, eof = '', 0, False
        started = False
        while True:
            if position >= len(buffer) and not eof:
                buffer, position = stream.read(chunk_size), 0
                eof = not buffer
            while position < len(buffer) and (buffer[position].isspace() or (started and buffer[position] == ',')):
                position += 1
            if not started:
                if position == len(buffer) and not eof:
                    continue
                if position == len(buffer) or buffer[position] != '[':
                    raise ValueError(f'Expected JSON array: {path}')
                position += 1
                started = True
                continue
            if position < len(buffer) and buffer[position] == ']':
                return
            if position >= len(buffer):
                if eof:
                    raise ValueError(f'Truncated JSON array: {path}')
                continue
            try:
                record, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError as error:
                if eof:
                    raise ValueError(f'Invalid JSON array: {path}') from error
                remaining = buffer[position:]
                more = stream.read(chunk_size)
                eof = not more
                buffer, position = remaining + more, 0
                continue
            if not isinstance(record, dict):
                raise ValueError(f'Expected JSON records: {path}')
            yield record
            position = end


@dataclass(slots=True)
class _Annotation:
    instance_token: str
    # xyz, scalar-first quaternion, wlh: compact allocation, float64 geometry.
    geometry: np.ndarray
    semantic_class: int
    num_lidar_pts: int
    visibility_token: str


class NuScenesLabelIndex:
    """Minimal official-table index with streamed LiDAR keyframe selection.

    ``scenes`` optionally restricts retained samples/annotations to scene names
    or tokens. All sensor tables are streamed; only LIDAR_TOP keyframe ego poses
    and calibrations are retained. ``sample_period_s`` and ``max_gap_s`` enforce
    adjacent keyframe timing rather than silently bridging missing samples.
    """

    def __init__(self, root, version='v1.0-trainval', *, scenes=None,
                 sample_period_s=.5, max_gap_s=.8):
        self.root = Path(root)
        self.version = version
        self.sample_period_s = float(sample_period_s)
        self.max_gap_s = float(max_gap_s)
        if not 0 < self.sample_period_s <= self.max_gap_s:
            raise ValueError('Expected positive sample period and max gap >= period')
        table_root = self.root / version
        all_scenes = {row['token']: row for row in _json_records(table_root / 'scene.json')}
        by_name = {row['name']: token for token, row in all_scenes.items()}
        selected = set(all_scenes) if scenes is None else {by_name.get(scene, scene) for scene in scenes}
        if not selected <= all_scenes.keys():
            raise ValueError('Unknown requested scene')
        self.scenes = {token: all_scenes[token] for token in selected}
        self.scene_names = {row['name']: token for token, row in self.scenes.items()}
        self.samples = {row['token']: row for row in _json_records(table_root / 'sample.json') if row['scene_token'] in selected}
        sensor_channels = {row['token']: row['channel'] for row in _json_records(table_root / 'sensor.json')}
        self.calibrations = {
            row['token']: row for row in _json_records(table_root / 'calibrated_sensor.json')
            if sensor_channels[row['sensor_token']] == 'LIDAR_TOP'
        }
        self.lidar = {}
        for row in _json_records(table_root / 'sample_data.json'):
            token = row['sample_token']
            if token in self.samples and row.get('is_key_frame') and row['calibrated_sensor_token'] in self.calibrations:
                if token in self.lidar:
                    raise ValueError(f'Multiple LIDAR_TOP keyframes for {token}')
                self.lidar[token] = {key: row[key] for key in ('token', 'ego_pose_token', 'calibrated_sensor_token')}
        needed_poses = {row['ego_pose_token'] for row in self.lidar.values()}
        self.ego_poses = {row['token']: row for row in _json_records(table_root / 'ego_pose.json') if row['token'] in needed_poses}
        categories = {row['token']: row['name'] for row in _json_records(table_root / 'category.json')}
        instance_classes = {
            row['token']: CATEGORY_TO_OCC3D.get(categories[row['category_token']], 0)
            for row in _json_records(table_root / 'instance.json')
        }
        self.annotations = {token: [] for token in self.samples}
        for row in _json_records(table_root / 'sample_annotation.json'):
            token = row['sample_token']
            semantic = instance_classes[row['instance_token']]
            if token not in self.samples or semantic not in ACTOR_CLASSES:
                continue
            geometry = np.asarray(row['translation'] + row['rotation'] + row['size'], dtype=np.float64)
            if geometry.shape != (10,) or not np.isfinite(geometry).all() or (geometry[7:] <= 0).any():
                raise ValueError(f'Invalid annotation geometry: {row["token"]}')
            if np.linalg.norm(geometry[3:7]) == 0:
                raise ValueError(f'Invalid annotation quaternion: {row["token"]}')
            self.annotations[token].append(_Annotation(
                row['instance_token'], geometry, semantic,
                int(row.get('num_lidar_pts', 0)), str(row.get('visibility_token', '0')),
            ))
        self._ego_transforms = {}

    def _scene_token(self, scene):
        token = self.scene_names.get(scene, scene)
        if token not in self.scenes:
            raise ValueError(f'Unknown indexed scene: {scene}')
        return token

    def _sample(self, scene, token):
        scene_token = self._scene_token(scene)
        if token not in self.samples or self.samples[token]['scene_token'] != scene_token:
            raise ValueError(f'Missing sample or cross-scene token: {token}')
        return self.samples[token]

    def ego_to_global(self, sample_token):
        if sample_token not in self._ego_transforms:
            if sample_token not in self.lidar:
                raise ValueError(f'Missing LIDAR_TOP keyframe for {sample_token}')
            pose_token = self.lidar[sample_token]['ego_pose_token']
            if pose_token not in self.ego_poses:
                raise ValueError(f'Missing LiDAR ego pose for {sample_token}')
            pose = self.ego_poses[pose_token]
            self._ego_transforms[sample_token] = transform_matrix(pose['translation'], pose['rotation'])
        return self._ego_transforms[sample_token].copy()

    def _adjacent(self, scene, sample, direction):
        other_token = sample[direction]
        if not other_token:
            return None
        other = self._sample(scene, other_token)
        opposite = 'next' if direction == 'prev' else 'prev'
        if other[opposite] != sample['token']:
            raise ValueError('Nonreciprocal sample chain; missing samples cannot be bridged')
        elapsed = (other['timestamp'] - sample['timestamp']) / 1e6
        delta = -elapsed if direction == 'prev' else elapsed
        if not self.sample_period_s / 2 <= delta <= self.max_gap_s:
            raise ValueError(f'Invalid adjacent sample timestamp gap: {delta}s')
        return other

    def sample_tokens(self, scene):
        """Validated official scene order; no token/date sorting substitutes."""
        scene_token = self._scene_token(scene)
        record = self.scenes[scene_token]
        tokens, seen = [], set()
        sample = self._sample(scene, record['first_sample_token'])
        while sample is not None:
            if sample['token'] in seen:
                raise ValueError('Sample chain contains a cycle')
            seen.add(sample['token'])
            tokens.append(sample['token'])
            sample = self._adjacent(scene, sample, 'next')
        if tokens[-1] != record['last_sample_token'] or len(tokens) != record['nbr_samples']:
            raise ValueError('Incomplete official scene chain')
        return tokens

    def _history(self, scene, current_token, history):
        if history < 1:
            raise ValueError('history must be positive')
        sample = self._sample(scene, current_token)
        chain = [sample]
        for _ in range(history - 1):
            sample = self._adjacent(scene, sample, 'prev')
            if sample is None:
                break
            chain.append(sample)
        return [None] * (history - len(chain)) + list(reversed(chain))

    @staticmethod
    def _actor_pose(annotation, global_to_e0):
        pose = np.zeros(6, dtype=np.float32)
        pose[:2] = transform_points(global_to_e0, annotation.geometry[:3])[:2]
        heading = global_to_e0[:3, :3] @ quaternion_rotation(annotation.geometry[3:7])[:, 0]
        yaw = np.arctan2(heading[1], heading[0])
        pose[2:4] = [np.sin(yaw), np.cos(yaw)]
        return pose

    @staticmethod
    def _ego_pose(ego_transform, global_to_e0):
        relative = global_to_e0 @ ego_transform
        yaw = np.arctan2(relative[1, 0], relative[0, 0])
        return np.array([*relative[:2, 3], np.sin(yaw), np.cos(yaw), 0, 0], dtype=np.float32)

    def build_observation(self, scene, current_token, *, history=4, max_actors=8):
        """Only current/past records influence every returned observation value."""
        if max_actors < 1:
            raise ValueError('max_actors must be positive')
        chain = self._history(scene, current_token, history)
        current = chain[-1]
        global_to_e0 = inverse_transform(self.ego_to_global(current_token))
        observed = [ann for ann in self.annotations[current_token] if ann.num_lidar_pts > 0]
        observed.sort(key=lambda ann: (float(np.linalg.norm(transform_points(global_to_e0, ann.geometry[:3])[:2])), ann.instance_token))
        selected = observed[:max_actors]
        selected_tokens = [ann.instance_token for ann in selected]
        if len(set(selected_tokens)) != len(selected_tokens):
            raise ValueError('Duplicate current instance annotations')
        actor_history = np.zeros((max_actors, history, 6), np.float32)
        actor_mask = np.zeros((max_actors, history), bool)
        velocity_mask = np.zeros_like(actor_mask)
        actor_size = np.zeros((max_actors, 3), np.float32)
        actor_class = np.full(max_actors, -1, np.int16)
        actor_z = np.zeros(max_actors, np.float32)
        actor_visibility = np.zeros(max_actors, np.int8)
        ego_history = np.zeros((history, 6), np.float32)
        ego_mask = np.array([sample is not None for sample in chain], dtype=bool)
        ego_velocity_mask = np.zeros(history, bool)
        history_times = np.zeros(history, np.float64)
        for slot, ann in enumerate(selected):
            # Current wlh -> E0 body length,width,height. Dimensions stay intrinsic.
            actor_size[slot] = ann.geometry[[8, 7, 9]]
            actor_class[slot] = ann.semantic_class
            actor_z[slot] = transform_points(global_to_e0, ann.geometry[:3])[2]
            actor_visibility[slot] = int(ann.visibility_token)
        for time_index, sample in enumerate(chain):
            if sample is None:
                continue
            history_times[time_index] = (sample['timestamp'] - current['timestamp']) / 1e6
            ego_history[time_index] = self._ego_pose(self.ego_to_global(sample['token']), global_to_e0)
            by_instance = {ann.instance_token: ann for ann in self.annotations[sample['token']] if ann.num_lidar_pts > 0}
            for slot, instance_token in enumerate(selected_tokens):
                if instance_token in by_instance:
                    actor_history[slot, time_index] = self._actor_pose(by_instance[instance_token], global_to_e0)
                    actor_mask[slot, time_index] = True
            if time_index and chain[time_index - 1] is not None:
                dt = history_times[time_index] - history_times[time_index - 1]
                ego_history[time_index, 4:] = (ego_history[time_index, :2] - ego_history[time_index - 1, :2]) / dt
                ego_velocity_mask[time_index] = True
                common = actor_mask[:, time_index] & actor_mask[:, time_index - 1]
                actor_history[common, time_index, 4:] = (actor_history[common, time_index, :2] - actor_history[common, time_index - 1, :2]) / dt
                velocity_mask[common, time_index] = True
        # speed, longitudinal acceleration, yaw rate, curvature; all backward-only.
        ego_state, ego_state_mask = np.zeros(4, np.float32), np.zeros(4, bool)
        if history >= 2 and ego_velocity_mask[-1]:
            dt = history_times[-1] - history_times[-2]
            speed = np.linalg.norm(ego_history[-1, 4:])
            yaw_current = np.arctan2(ego_history[-1, 2], ego_history[-1, 3])
            yaw_previous = np.arctan2(ego_history[-2, 2], ego_history[-2, 3])
            yaw_delta = np.arctan2(np.sin(yaw_current - yaw_previous), np.cos(yaw_current - yaw_previous))
            yaw_rate = yaw_delta / dt
            ego_state[[0, 2, 3]] = [speed, yaw_rate, yaw_rate / speed if speed > 1e-3 else 0]
            ego_state_mask[[0, 2, 3]] = [True, True, speed > 1e-3]
            if history >= 3 and ego_velocity_mask[-2]:
                previous_dt = history_times[-2] - history_times[-3]
                ego_state[1] = (speed - np.linalg.norm(ego_history[-2, 4:])) / ((dt + previous_dt) / 2)
                ego_state_mask[1] = True
        observation = {
            'actor_history': actor_history, 'actor_observed_mask': actor_mask,
            'actor_velocity_mask': velocity_mask, 'actor_size': actor_size,
            'actor_class': actor_class, 'actor_z': actor_z,
            'actor_visibility': actor_visibility,
            'ego_history': ego_history, 'ego_observed_mask': ego_mask,
            'ego_velocity_mask': ego_velocity_mask, 'ego_current': ego_history[-1].copy(),
            'ego_state': ego_state, 'ego_state_mask': ego_state_mask,
            'history_mask': ego_mask.copy(), 'history_time_s': history_times,
        }
        metadata = {
            'version': LABEL_VERSION, 'oracle': True, 'coordinate_frame': 'E0',
            'axes': 'x-forward,y-left,z-up', 'units': 'metres,seconds',
            'scene': self.scenes[current['scene_token']]['name'], 'token': current_token,
            'timestamp_us': int(current['timestamp']),
            'actor_instance_tokens': selected_tokens + [''] * (max_actors - len(selected_tokens)),
            'history_sample_tokens': [sample['token'] if sample else '' for sample in chain],
            'actor_selection': 'current num_lidar_pts>0, movable Occ3D classes, distance then instance_token',
            'actor_size_order': 'length,width,height', 'actor_state_order': 'x,y,sin_yaw,cos_yaw,vx,vy',
            'ego_state_order': 'speed,longitudinal_acceleration,yaw_rate,curvature',
            'e0_to_global': self.ego_to_global(current_token).tolist(),
        }
        return observation, metadata

    def build_sample(self, scene, current_token, *, history=4, future=6, max_actors=8):
        if future < 0:
            raise ValueError('future must be nonnegative')
        observation, metadata = self.build_observation(scene, current_token, history=history, max_actors=max_actors)
        selected = metadata['actor_instance_tokens']
        actor_future = np.zeros((max_actors, future, 6), np.float32)
        actor_exists = np.zeros((max_actors, future), bool)
        actor_velocity_mask = np.zeros_like(actor_exists)
        actor_future_z = np.zeros((max_actors, future), np.float32)
        ego_future = np.zeros((future, 6), np.float32)
        ego_mask = np.zeros(future, bool)
        future_times = np.zeros(future, np.float64)
        global_to_e0 = inverse_transform(self.ego_to_global(current_token))
        sample = self._sample(scene, current_token)
        initial_timestamp = sample['timestamp']
        previous_actor = observation['actor_history'][:, -1].copy()
        previous_exists = observation['actor_observed_mask'][:, -1].copy()
        previous_ego = observation['ego_current'].copy()
        previous_time = 0.
        future_tokens = []
        for step in range(future):
            sample = self._adjacent(scene, sample, 'next')
            if sample is None:
                break
            token = sample['token']
            future_tokens.append(token)
            future_times[step] = (sample['timestamp'] - initial_timestamp) / 1e6
            dt = future_times[step] - previous_time
            ego_future[step] = self._ego_pose(self.ego_to_global(token), global_to_e0)
            ego_future[step, 4:] = (ego_future[step, :2] - previous_ego[:2]) / dt
            ego_mask[step] = True
            by_instance = {ann.instance_token: ann for ann in self.annotations[token]}
            for slot, instance_token in enumerate(selected):
                if instance_token and instance_token in by_instance:
                    actor_future[slot, step] = self._actor_pose(by_instance[instance_token], global_to_e0)
                    actor_exists[slot, step] = True
                    actor_future_z[slot, step] = transform_points(global_to_e0, by_instance[instance_token].geometry[:3])[2]
                    if previous_exists[slot]:
                        actor_future[slot, step, 4:] = (actor_future[slot, step, :2] - previous_actor[slot, :2]) / dt
                        actor_velocity_mask[slot, step] = True
            previous_actor = actor_future[:, step].copy()
            previous_exists = actor_exists[:, step].copy()
            previous_ego = ego_future[step].copy()
            previous_time = future_times[step]
        target = {
            'actor_future': actor_future, 'actor_target_mask': actor_exists.copy(),
            'actor_future_exists': actor_exists, 'actor_velocity_mask': actor_velocity_mask,
            'actor_existence_mask': np.asarray([bool(token) for token in selected])[:, None] & ego_mask[None, :],
            'actor_future_z': actor_future_z,
            'ego_future': ego_future, 'ego_target_mask': ego_mask,
            'future_time_s': future_times,
        }
        metadata['future_sample_tokens'] = future_tokens + [''] * (future - len(future_tokens))
        return {'observation': observation, 'target': target, 'metadata': metadata}

    def occupancy_to_e0(self, source_token, current_token, *, source_frame):
        """Explicit ego-vs-LiDAR source convention; never conflate the frames."""
        if source_frame not in {'ego', 'lidar'}:
            raise ValueError('source_frame must be explicitly ego or lidar')
        source = self.samples[source_token]
        current = self.samples[current_token]
        if source['scene_token'] != current['scene_token']:
            raise ValueError('Cross-scene occupancy transforms are forbidden')
        transform = inverse_transform(self.ego_to_global(current_token)) @ self.ego_to_global(source_token)
        if source_frame == 'lidar':
            calibration = self.calibrations[self.lidar[source_token]['calibrated_sensor_token']]
            transform = transform @ transform_matrix(calibration['translation'], calibration['rotation'])
        return transform

    def load_occupancy_e0(self, source_token, current_token, *, source_frame,
                          source_grid=GridSpec(), target_grid=GridSpec(),
                          visibility_keys=('mask_lidar', 'mask_camera')):
        """Warp local semantics and requested masks; () reads no sensor masks."""
        source = self.samples[source_token]
        scene_name = self.scenes[source['scene_token']]['name']
        path = self.root / 'gts' / scene_name / source_token / 'labels.npz'
        with np.load(path, allow_pickle=False) as label:
            semantics = label['semantics']
            if not np.isin(semantics, np.array([*range(18), 255])).all():
                raise ValueError('Occupancy contains values outside Occ3D18 and unknown255')
            masks = {key: label[key].astype(bool) for key in visibility_keys if key in label}
        result = warp_occupancy_nearest(
            semantics, self.occupancy_to_e0(source_token, current_token, source_frame=source_frame),
            visibility=masks, source_grid=source_grid, target_grid=target_grid,
        )
        result.update({'source_frame': source_frame, 'coordinate_frame': 'E0', 'oracle': True})
        return result


def build_research_labels(index, output, scenes, *, history=4, future=6, max_actors=8):
    """Write numeric per-scene actor/ego shards and explicit JSON identities.

    Occupancy is warped on demand with load_occupancy_e0 rather than duplicating
    3-D grids for every history/target combination. Scenes are supplied by the
    caller so calibration/evaluation membership is never baked into fitting.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    selected = [index._scene_token(scene) for scene in scenes]
    if not selected or len(set(selected)) != len(selected):
        raise ValueError('Provide nonempty unique scenes')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.research-', dir=output.parent))
    manifest = {
        'version': LABEL_VERSION, 'format': 'reactoccflow_research_labels',
        'oracle': True, 'oracle_source': 'official_annotations_and_ego_poses',
        'coordinate_frame': 'E0', 'axes': 'x-forward,y-left,z-up', 'units': 'metres,seconds',
        'history': history, 'future': future, 'max_actors': max_actors,
        'grid': GridSpec().as_dict(), 'semantic_classes': list(OCC3D_CLASSES),
        'category_to_occ3d': CATEGORY_TO_OCC3D, 'unknown_label': 255, 'free_label': 17,
        'occupancy_storage': 'on_demand_nearest_3d_before_encoding',
        'occupancy_source_frame': 'requires_explicit_ego_or_lidar',
        'source_version': index.version,
        'sample_period_s': index.sample_period_s, 'max_gap_s': index.max_gap_s,
        'provenance': 'docs/provenance/research_sources.json',
        'num_frames': 0, 'shards': [], 'frames': [],
    }
    try:
        for scene_token in selected:
            scene_name = index.scenes[scene_token]['name']
            samples = [index.build_sample(scene_token, token, history=history, future=future, max_actors=max_actors)
                       for token in index.sample_tokens(scene_token)]
            arrays = {f'{group}.{key}': np.stack([sample[group][key] for sample in samples])
                      for group in ('observation', 'target') for key in samples[0][group]}
            name = f'{scene_name}.npz'
            np.savez_compressed(staging / name, **arrays)
            shard_id = len(manifest['shards'])
            manifest['shards'].append({'file': name, 'scene': scene_name, 'count': len(samples)})
            for row, sample in enumerate(samples):
                manifest['frames'].append({**sample['metadata'], 'shard': shard_id, 'row': row})
            manifest['num_frames'] += len(samples)
        (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest
