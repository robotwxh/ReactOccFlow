# T4 annotation-oracle E0 research labels

This is an **oracle representation experiment** based on official nuScenes
annotations and ego poses. It is not a raw-LiDAR detector, tracker, or perception
front end. Future ground truth only appears in the `target` group and label-frame
construction; it is not an inference input or an actor-selection signal.

## Public API

```python
from reactoccflow.data.research import NuScenesLabelIndex, build_research_labels

index = NuScenesLabelIndex('/home/lthpc/data/nuscenes', scenes=['scene-0001'])
observation, metadata = index.build_observation('scene-0001', current_token)
sample = index.build_sample('scene-0001', current_token, history=4, future=6,
                            max_actors=8)
manifest = build_research_labels(index, output_dir, scenes=['scene-0001'])
label = index.load_occupancy_e0(source_token, current_token, source_frame='ego')
```

`scenes` accepts indexed scene names or stable tokens. The constructor streams
`sample_data.json`, `ego_pose.json`, and `sample_annotation.json` rather than
loading the full multi-GB sensor tables. It retains only selected samples,
LIDAR_TOP keyframe sample-data records/poses, LiDAR calibration, and movable
annotation records. Raw official `sample.json` has neither the SDK's `data`
lookup nor its `anns` lookup: these reverse indexes are built independently.
No nuScenes, MMCV, mmdet, reference-project import, CUDA JIT, or modified source
environment is required.

`sample_tokens(scene)` validates the official sample chain, reciprocal links,
scene boundaries, first/last identities, count, and time gaps. Adjacent timing
must be between 0.25 and 0.8 seconds by default (nominal 0.5 seconds). Missing
records raise an error; missing samples are never silently bridged. Real scene
starts/tails produce padded masks. History slots include current and are ordered
oldest-to-current; future slots correspond to the next six sample-chain entries.
Actual relative timestamps, including normal jitter, are retained in seconds.

## Tensor and metadata contract

All arrays are numeric or Boolean; shards load with `allow_pickle=False`.
Default shapes exclude a batch dimension.

| Group/key | Shape | Meaning |
| --- | --- | --- |
| observation.actor_history | 8,4,6 | x,y,sin(yaw),cos(yaw),vx,vy in E0 |
| observation.actor_observed_mask | 8,4 | Selected actor annotated with lidar points in observed slot |
| observation.actor_velocity_mask | 8,4 | Current and immediately preceding observed slot available |
| observation.actor_size | 8,3 | Current intrinsic length,width,height, in metres |
| observation.actor_class | 8 | Current Occ3D semantic ID; -1 for padded actors |
| observation.actor_z | 8 | Current E0 box centre z |
| observation.actor_visibility | 8 | Current official visibility token; padded 0 |
| observation.ego_history | 4,6 | E0 pose/heading and backward-only planar velocity |
| observation.ego_current | 6 | Current ego-history state |
| observation.ego_observed_mask/history_mask | 4 | Existing current/past slots |
| observation.ego_velocity_mask | 4 | Valid backward velocity slots |
| observation.history_time_s | 4 | Relative seconds, padded 0 with false mask |
| observation.ego_state | 4 | Speed, longitudinal acceleration, yaw rate, curvature |
| observation.ego_state_mask | 4 | Availability of each backward-derived ego feature |
| target.actor_future | 8,6,6 | Future selected-actor state in fixed E0 |
| target.actor_target_mask | 8,6 | Annotated future pose validity |
| target.actor_future_exists | 8,6 | Selected instance annotated at that exact future sample |
| target.actor_existence_mask | 8,6 | Selected current actor and valid future frame; includes disappearance |
| target.actor_velocity_mask | 8,6 | Immediately preceding actor pose available; no missing-step bridging |
| target.actor_future_z | 8,6 | Future actor centre z for box overlap |
| target.ego_future | 6,6 | Future ego E0 state; target only |
| target.ego_target_mask | 6 | Future sample exists |
| target.future_time_s | 6 | Actual relative target times, padded 0 |

Dimensions and class come from the **current** annotation. Official box size
`(width,length,height)` becomes `(length,width,height)`. E0 yaw is the projected
heading of the box's intrinsic local x-axis after full 3-D rotation; no upstream
LiDAR-specific yaw convention is reused. Annotation position is the box centre.
Z is not part of the six dynamic actor-state channels; current/future centre z
is carried separately for geometry. Padded dynamic states and unavailable
velocities are 0 and must be consumed with the corresponding masks.

Metadata contains stable actor instance tokens, current identity/timestamp,
history/future sample identities, units/axis labels, and the **current** E0-to-
global transform. `build_observation` never reads future samples or poses and
returns no future metadata. `build_sample` calls it first, then separately
traverses future records and adds future identities to metadata. Pass only the
observation dictionary as inference inputs; the full label sample and training
manifest contain target identities and must not be passed wholesale to a model.

## Actor observation, association, and target policy

A participating actor must have a current annotation with `num_lidar_pts >0`
and a movable class from `{bicycle,bus,car,construction_vehicle,motorcycle,
pedestrian,trailer,truck}`. Zero-LiDAR-point current annotations are excluded even
if marked visible by cameras. Selection sorts current **E0 planar distance**
then stable `instance_token`, taking the nearest 8. The historical selected
identity is associated only by that token, never by future boxes or nearest
future trajectories. Past slots with no LiDAR points are unobserved and masked.
Historical and current velocities use backward differences between consecutive
observed slots only; no future or centered-difference velocities are imported.
The first available history slot has zero velocity and false velocity mask.

Future pose/existence comes from whether the same instance has an annotation
at the exact next sample. A future annotation with zero LiDAR points can still
be a valid oracle target. No annotation means false existence/pose mask; the
separate existence-validity mask still allows disappearance supervision when a
future frame exists. These are **annotation-existence** labels, not proof that
an object physically ceases to exist or is observable. Reappearing targets do
not bridge a missing immediately preceding target to create velocity labels.
New instances absent now, zero-point actors, static objects, and actors beyond
slot 8 remain represented by occupancy rather than being fabricated as selected
actor tracks.

## Occ3D18 semantic map

| ID | Class | Official category mapping |
| --- | --- | --- |
|0|others|All unlisted official categories/noise |
|1|barrier|movable_object.barrier |
|2|bicycle|vehicle.bicycle |
|3|bus|vehicle.bus.bendy, vehicle.bus.rigid |
|4|car|vehicle.car |
|5|construction_vehicle|vehicle.construction |
|6|motorcycle|vehicle.motorcycle |
|7|pedestrian|human.pedestrian.adult/child/construction_worker/police_officer |
|8|traffic_cone|movable_object.trafficcone |
|9|trailer|vehicle.trailer |
|10|truck|vehicle.truck |
|11|driveable_surface|flat.driveable_surface |
|12|other_flat|flat.other |
|13|sidewalk|flat.sidewalk |
|14|terrain|flat.terrain |
|15|manmade|static.manmade |
|16|vegetation|static.vegetation |
|17|free|Data-defined empty voxel |

Unknown is255 and is not free 17, others 0, or a new semantic class. Emergency
vehicles, animals, stroller/wheelchair/personal-mobility annotations and other
unlisted categories map to others and are not selected as the movable 8-class
actor set. No assignment is inferred from future annotation properties.

## Column-vector frame and grid geometry

The independent geometry API uses column vectors: `p_destination = R @ p_source
+ t`, represented by a 4x4 homogeneous transform. nuScenes quaternions are
scalar-first `(w,x,y,z)`; SciPy receives explicit scalar-last reorderings.

IR-WM's row-vector chain in `union2one` has factor order
`L_t_to_E_t.T @ E_t_to_G.T @ G_to_E_0.T @ E_0_to_L_0.T`.
Transposing that whole chain reverses factor order and gives the conventional
column-vector chain. The research destination is **E0**, not L0:

- Ego source: `T_E0_from_Et = inv(T_G_from_E0) @ T_G_from_Et`.
- LiDAR source: append `T_Et_from_Lt` on the right of that chain.

The `source_frame` keyword is mandatory in `occupancy_to_e0` and
`load_occupancy_e0`. Unsupported or cross-scene conventions raise errors.

Default grid storage axes are XYZ, shape `(200,200,16)`, lower corner
`(-40,-40,-1)`m, voxel dimensions `(0.4,0.4,0.4)`m and half-open extent
`[-40,40) x [-40,40) x [-1,5.4)`. Voxel centres are lower corner plus
`(index+0.5)*voxel_size`. The destination centres are mapped backwards through
the inverse SE3 and sampled at the nearest source centre. Semantic IDs are
never interpolated. Out-of-grid sources become255 with false visibility;
there is no free-space fill, border clamp, or 2-D latent warp. Both source
`mask_lidar` and `mask_camera` are independently preserved. Occupancy is warped
**in 3-D before encoding**; chunks of at most 65536 centres avoid repeatedly
allocating a 640000x3 coordinate array.

## Verified local gts coordinate convention

On 2026-10-28, six actual files were checked: first/middle/last keyframes in
`scene-0001` and `scene-0541`. All have `(200,200,16)` semantics and independent
`mask_lidar`/`mask_camera`. Moving official annotation boxes were transformed
from global to each proposed source frame. Grid voxel centres inside each full
oriented 3-D box were counted against the box's Occ3D class:

| Proposed source frame | Matching/inside voxels | Class match fraction | Boxes |
| --- | --- | --- | --- |
| Ego |8136 /27854 |0.29209449 |50 |
| LiDAR |0 /23046 |0.0 |47 |

This is empirical support for **ego-frame** local Occ3D labels, not LiDAR-frame
labels. Partial box fill explains why the ego fraction is below 1; it is not a
voxel-IoU result. LiDAR extrinsics are materially different: scene-0001 has
translation `(0.943713,0,1.84023)`m and quaternion
`(0.70779551,-0.00649224,0.01064621,-0.70630731)` (about -90 deg yaw), while
scene-0541 has `(0.985793,0,1.84019)` and a similar rotation. Applying these
extrinsics to already ego-frame grids would rotate/translate labels incorrectly.
Use `source_frame='ego'` for the verified local `gts` files; a different dataset
must supply independently verified source conventions.

The checked sample identities were:

- scene-0001: `e93e98b63d3b40209056d129dc53ceee`,
  `5c95b81d9d404c219b4393da7e020280`, `40e413c922184255a94f08d3c10037e0`.
- scene-0541: `21eb494e2f9f45f2a673f03d0d43ba56`,
  `1e1b109f0ed54310a7b369f8ff5f14af`, `fa52be6e34984c24b1e2c62b60621adf`.

## Storage, provenance, and boundaries

`build_research_labels` writes one compressed numeric NPZ per supplied scene and
an atomic JSON manifest with frame identities/metadata. It refuses an existing
output directory, never rewrites inputs, and removes only its own temporary
staging directory on failure. The manifest records grid, classes, masks, timing,
version, oracle source, timing policy (`sample_period_s`, `max_gap_s`), and provenance. Scene/split membership is caller-owned;
no official validation scenes are hardcoded into a fitting set. The builder
stores actors/ego only and loads occupancy on demand to avoid repeated 3-D grids.
Creating encoded E0 caches, integration datasets, scene splits, evaluation,
visual validation and full 850-scene preprocessing are coordinator tasks.

Both reference repositories were searched before implementing. Five ordinary,
independent snapshots were copied before adaptation under
`reactoccflow/geometry/source_snapshots/`; they are audit text and never imported.
Exact original commits/hashes and scope are recorded in
`docs/provenance/research_sources.json`. The runtime dependency closure is NumPy,
SciPy Rotation and the Python standard library. Root license handling is tracked
in `THIRD_PARTY_NOTICES.md`; OccFM licensing remains unresolved for redistribution.

## Timing-policy ruling from the complete official census

The original 0.75-second maximum rejected one valid official reciprocal edge:
scene-0994, `6b05de2a752e4abcb3c2487c08b53bbb` ->
`d1ea6f7d23c14c2789b7357683dcf54a`, measured at **0.750945 seconds**.
`docs/implementation/keyframe_timing_audit.json` records all 33,299 official
same-scene edges: minimum 0.394377 s, median 0.499883 s, maximum 0.750945 s;
only this edge exceeded 0.75 s. Both scene membership and reciprocal sample
links are valid, so this was an overly strict guard rather than missing data.

Default `max_gap_s` is therefore explicitly **0.8 seconds**. Nominal
`sample_period_s` stays 0.5 seconds and both values are persisted in the label
manifest. Every velocity and relative timestamp continues to use measured
timestamp differences. No frame is dropped, resampled, interpolated, or bridged;
reciprocity, scene identity, scene sample counts and the 0.25-second minimum
remain checked, and differences over 0.8 seconds still raise errors. A caller
can explicitly request a stricter maximum. Regression tests reproduce
`[0,0.5,1.250945,1.750945]` s and independently expect velocity
`-1/0.750945` m/s for a one-metre negative-y displacement.
