"""Analytic checks independent of the adapted reference implementations."""
import numpy as np
import pytest


def test_column_se3_translation_rotation_vectors_and_inverse():
    from reactoccflow.geometry import transform_matrix, inverse_transform, transform_points, transform_vectors
    root = np.sqrt(.5)
    transform = transform_matrix([2, 3, 4], [root, 0, 0, root])
    np.testing.assert_allclose(transform_points(transform, [[1, 0, 0]]), [[2, 4, 4]], atol=1e-12)
    np.testing.assert_allclose(transform_vectors(transform, [[1, 0, 0]]), [[0, 1, 0]], atol=1e-12)
    np.testing.assert_allclose(transform_points(inverse_transform(transform), [[2, 4, 4]]), [[1, 0, 0]], atol=1e-12)


def test_invalid_geometry_rejected_and_grid_bounds_not_clamped():
    from reactoccflow.geometry import GridSpec, transform_matrix, transform_points
    grid = GridSpec(shape=(2, 2, 2), origin=(0, 0, 0), voxel_size=(1, 1, 1))
    indices, valid = grid.coordinates_to_indices([[.2, .2, .2], [-.1, 0, 0], [2, 1, 1], [np.nan, 0, 0]])
    assert valid.tolist() == [True, False, False, False]
    assert indices[1, 0] == -1
    with pytest.raises(ValueError):
        transform_matrix([0, 0, 0], [0, 0, 0, 0])
    with pytest.raises(ValueError):
        transform_points(np.eye(4), [[np.nan, 0, 0]])


def test_nearest_warp_preserves_unknown_visibility_and_out_of_range():
    from reactoccflow.geometry import GridSpec, transform_matrix, warp_occupancy_nearest
    grid = GridSpec(shape=(3, 2, 1), origin=(0, 0, 0), voxel_size=(1, 1, 1))
    source = np.array([[[4], [255]], [[7], [17]], [[10], [1]]], dtype=np.uint8)
    visibility = source != 255
    same = warp_occupancy_nearest(source, np.eye(4), visibility=visibility, source_grid=grid, target_grid=grid)
    np.testing.assert_array_equal(same['semantics'], source)
    np.testing.assert_array_equal(same['visibility'], visibility)
    shifted = warp_occupancy_nearest(source, transform_matrix([1, 0, 0]), visibility=visibility, source_grid=grid, target_grid=grid, chunk_size=2)
    np.testing.assert_array_equal(shifted['semantics'][:, :, 0], [[255, 255], [4, 255], [7, 17]])
    assert not shifted['in_bounds'][0].any()
    assert not shifted['visibility'][1, 1, 0]


def test_warp_rotates_in_3d_before_encoding():
    from reactoccflow.geometry import GridSpec, transform_matrix, warp_occupancy_nearest
    grid = GridSpec(shape=(2, 2, 2), origin=(-1, -1, -1), voxel_size=(1, 1, 1))
    source = np.full(grid.shape, 17, dtype=np.uint8)
    source[1, 1, 0] = 4
    root = np.sqrt(.5)
    rotated = warp_occupancy_nearest(source, transform_matrix([0, 0, 0], [root, root, 0, 0]), source_grid=grid, target_grid=grid)
    assert rotated['semantics'][1, 1, 1] == 4
    assert np.count_nonzero(rotated['semantics'] == 4) == 1


def test_yaw90_nearest_warp_and_independent_visibility_masks():
    from reactoccflow.geometry import GridSpec, transform_matrix, warp_occupancy_nearest
    grid = GridSpec(shape=(2, 2, 1), origin=(-1, -1, 0), voxel_size=(1, 1, 1))
    source = np.array([[[1], [2]], [[3], [4]]], dtype=np.uint8)
    root = np.sqrt(.5)
    warped = warp_occupancy_nearest(source, transform_matrix(quaternion_wxyz=[root, 0, 0, root]),
                                  source_grid=grid, target_grid=grid,
                                  visibility={'mask_lidar': source == 4, 'mask_camera': source == 2})
    np.testing.assert_array_equal(warped['semantics'][:, :, 0], [[2, 4], [1, 3]])
    assert warped['visibility']['mask_lidar'][0, 1, 0]
    assert warped['visibility']['mask_camera'][0, 0, 0]
    assert warped['in_bounds'].all()
