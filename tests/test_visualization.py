import numpy as np


def test_semantic_projection_distinguishes_unknown_empty_and_top_voxel():
    from reactoccflow.visualization import semantic_bev
    semantic=np.array([[[255,255,255],[17,17,255]],[[4,17,17],[4,7,17]]],dtype=np.uint8)
    np.testing.assert_array_equal(semantic_bev(semantic),[[255,17],[4,7]])


def test_footprint_local_length_and_yaw_rotation():
    from reactoccflow.visualization import actor_footprint
    state=np.array([3,4,1,0,0,0]) # yaw +90 degrees
    polygon=actor_footprint(state,[4,2,1])
    np.testing.assert_allclose(polygon.mean(axis=0),[3,4])
    np.testing.assert_allclose(polygon.max(axis=0)-polygon.min(axis=0),[2,4],atol=1e-6)
