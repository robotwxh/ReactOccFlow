"""Project-owned column-vector SE(3) and 3-D occupancy geometry.

Adapted after copying IR-WM transform/grid functions. Source snapshots, source
commits and hashes are recorded in docs/provenance/research_sources.json.
IR-WM's row-vector transform chains are transposed/reversed here deliberately.
No upstream runtime, JIT extension, or reference directory is imported.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation


def quaternion_rotation(quaternion_wxyz) -> np.ndarray:
    """Return a rotation matrix for nuScenes scalar-first (w,x,y,z)."""
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) == 0:
        raise ValueError('Quaternion must contain four finite values and have nonzero norm')
    return Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()


def transform_matrix(translation=(0, 0, 0), quaternion_wxyz=(1, 0, 0, 0)) -> np.ndarray:
    """Column-vector source-to-destination transform: p_dest = R @ p + t."""
    translation = np.asarray(translation, dtype=np.float64)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError('Translation must have three finite components')
    result = np.eye(4)
    result[:3, :3] = quaternion_rotation(quaternion_wxyz)
    result[:3, 3] = translation
    return result


def _checked_transform(transform):
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError('Transform must be a finite 4x4 SE(3) matrix')
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError('Invalid SE(3) homogeneous row')
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6):
        raise ValueError('Transform rotation must be orthonormal with determinant +1')
    return transform


def inverse_transform(transform) -> np.ndarray:
    transform = _checked_transform(transform)
    result = np.eye(4)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def transform_vectors(transform, vectors) -> np.ndarray:
    transform = _checked_transform(transform)
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim < 1 or vectors.shape[-1] != 3 or not np.isfinite(vectors).all():
        raise ValueError('Vectors must have finite xyz coordinates in their final dimension')
    return vectors @ transform[:3, :3].T


def transform_points(transform, points) -> np.ndarray:
    transform = _checked_transform(transform)
    return transform_vectors(transform, points) + transform[:3, 3]


@dataclass(frozen=True)
class GridSpec:
    """XYZ storage, metre lower corner, and voxel dimensions (not latent cells)."""
    shape: tuple[int, int, int] = (200, 200, 16)
    origin: tuple[float, float, float] = (-40., -40., -1.)
    voxel_size: tuple[float, float, float] = (.4, .4, .4)

    def __post_init__(self):
        shape = tuple(self.shape)
        origin = tuple(float(x) for x in self.origin)
        voxel = tuple(float(x) for x in self.voxel_size)
        if len(shape) != 3 or any(not isinstance(x, (int, np.integer)) or x < 1 for x in shape):
            raise ValueError('Grid shape must contain three positive integers')
        if len(origin) != 3 or len(voxel) != 3 or not np.isfinite(origin + voxel).all() or any(x <= 0 for x in voxel):
            raise ValueError('Grid origin/voxel size must be finite and voxel sizes positive')
        object.__setattr__(self, 'shape', shape)
        object.__setattr__(self, 'origin', origin)
        object.__setattr__(self, 'voxel_size', voxel)

    def coordinates_to_indices(self, coordinates):
        """Containing-cell/nearest-centre index and validity; never clamp bounds."""
        coordinates = np.asarray(coordinates, dtype=np.float64)
        if coordinates.ndim < 1 or coordinates.shape[-1] != 3:
            raise ValueError('Coordinates must end in xyz')
        finite = np.isfinite(coordinates).all(axis=-1)
        safe = np.where(np.isfinite(coordinates), coordinates, 0)
        continuous = (safe - self.origin) / self.voxel_size
        # Avoid overflowing integer casts for finite but wildly invalid positions.
        representable = (np.abs(continuous) < np.iinfo(np.int64).max / 2).all(axis=-1)
        indices = np.floor(np.where(representable[..., None], continuous, -1)).astype(np.int64)
        valid = finite & representable & (indices >= 0).all(axis=-1) & (indices < self.shape).all(axis=-1)
        return indices, valid

    def centres(self, start=0, stop=None):
        """Generate only the requested flat XYZ-order chunk, avoiding full grids."""
        count = int(np.prod(self.shape))
        stop = count if stop is None else stop
        flat = np.arange(start, stop, dtype=np.int64)
        x, remainder = np.divmod(flat, self.shape[1] * self.shape[2])
        y, z = np.divmod(remainder, self.shape[2])
        return (np.stack((x, y, z), axis=-1) + .5) * self.voxel_size + self.origin

    def as_dict(self):
        return {'shape': list(self.shape), 'origin': list(self.origin), 'voxel_size': list(self.voxel_size), 'axis_order': 'xyz'}


def warp_occupancy_nearest(source, source_to_e0, *, visibility=None,
                           source_grid=GridSpec(), target_grid=GridSpec(),
                           unknown=255, chunk_size=65536) -> dict:
    """Reverse-map destination voxel centres in 3-D, before occupancy encoding.

    Semantic IDs are never interpolated. Out-of-source cells are unknown (255),
    with false visibility, and source unknowns remain unknown. Visibility can be
    one mask or a dictionary of independent masks (e.g. lidar and camera).
    """
    source = np.asarray(source)
    if source.shape != source_grid.shape or source.dtype.kind not in 'ui':
        raise ValueError('Occupancy must be an integer array matching source grid XYZ shape')
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    if unknown < 0 or unknown > np.iinfo(source.dtype).max:
        raise ValueError('Unknown label is not representable by source dtype')
    e0_to_source = inverse_transform(source_to_e0)
    masks = {} if visibility is None else (visibility if isinstance(visibility, dict) else {'visibility': visibility})
    masks = {key: np.asarray(value, dtype=bool) for key, value in masks.items()}
    if any(mask.shape != source.shape for mask in masks.values()):
        raise ValueError('Visibility masks must match occupancy shape')
    output = np.full(target_grid.shape, unknown, dtype=source.dtype)
    in_bounds = np.zeros(target_grid.shape, dtype=bool)
    output_masks = {key: np.zeros(target_grid.shape, dtype=bool) for key in masks}
    flat_output, flat_bounds = output.reshape(-1), in_bounds.reshape(-1)
    for start in range(0, output.size, chunk_size):
        stop = min(start + chunk_size, output.size)
        target_points = target_grid.centres(start, stop)
        source_points = target_points @ e0_to_source[:3, :3].T + e0_to_source[:3, 3]
        indices, valid = source_grid.coordinates_to_indices(source_points)
        valid_indices = tuple(indices[valid].T)
        flat_output[start:stop][valid] = source[valid_indices]
        flat_bounds[start:stop] = valid
        for key, mask in masks.items():
            output_masks[key].reshape(-1)[start:stop][valid] = mask[valid_indices]
    result = {'semantics': output, 'in_bounds': in_bounds}
    if visibility is not None:
        result['visibility'] = output_masks if isinstance(visibility, dict) else output_masks['visibility']
    return result
