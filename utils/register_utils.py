import numpy as np
import open3d as o3d
import cv2 as cv

from . import image_utils
from . import pose_utils


def sample_grid(img_height, img_width, grid_size=10, mask=None):
    pix_grid = []
    for i in range(img_height):
        for j in range(img_width):
            if (mask[i, j] != 0) and (i % grid_size == 0) and (j % grid_size == 0):
                pix_grid.append((j, i))
    return pix_grid


def extract_3d_correspondences(
    depth_map_list_1, depth_map_list_2, intrinsics, poses, mask, grid_size=10
):
    assert depth_map_list_1[0].shape == depth_map_list_2[0].shape
    height, width = depth_map_list_1[0].shape
    ## sample grid for pts per frame
    # mask = cv.erode(
    #     mask,
    #     cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20)),
    #     iterations=8,
    # )

    pts_2d = sample_grid(height, width, grid_size=grid_size, mask=mask)

    assert len(depth_map_list_1) == len(depth_map_list_2)

    pts_1 = []
    pts_2 = []
    for depth_1, depth_2, pose in zip(depth_map_list_1, depth_map_list_2, poses):
        for v, u in pts_2d:
            if depth_1[u, v] <= 0 or depth_2[u, v] <= 0:
                continue

            pt1 = image_utils.project_2d_to_3d(u, v, depth_1, pose, intrinsics)
            pt2 = image_utils.project_2d_to_3d(u, v, depth_2, pose, intrinsics)

            pts_1.append(pt1)
            pts_2.append(pt2)

    return np.asarray(pts_1), np.asarray(pts_2)


def register_3d(
    source_pts,
    target_pts,
    dist_thresh=3.0,
    n_samples=6,
    iterations=500,
    confidence=0.8,
):
    assert len(source_pts) == len(target_pts)
    n_pts = len(source_pts)

    source_pc = o3d.geometry.PointCloud()
    source_pc.points = o3d.utility.Vector3dVector(source_pts)

    target_pc = o3d.geometry.PointCloud()
    target_pc.points = o3d.utility.Vector3dVector(target_pts)

    ## assuming correspondences are in order
    idxs = list(range(0, n_pts))
    corres = np.stack((idxs, idxs), axis=1)
    corres = o3d.utility.Vector2iVector(corres)

    reg_res = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source=source_pc,
        target=target_pc,
        corres=corres,
        max_correspondence_distance=dist_thresh,
        ransac_n=n_samples,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iteration=iterations, confidence=confidence
        ),
    )

    # greedy_its = 500
    # best_ransac = None
    # best_rot_err = 360
    # for i in range(greedy_its):
    #     ransac_result = (
    #         o3d.pipelines.registration.registration_ransac_based_on_correspondence(
    #             source=source_pc,
    #             target=target_pc,
    #             corres=corres,
    #             max_correspondence_distance=dist_thresh,
    #             ransac_n=n_samples,
    #             criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
    #                 max_iteration=iterations, confidence=confidence
    #             ),
    #         )
    #     )
    #     tmp_transform = ransac_result.transformation
    #     transformed_colmap_poses_tmp = pose_utils.transform_poses(
    #         scaled_colmap, tmp_transform
    #     )
    #     rot_errs, _ = err.calculate_pose_error(
    #         transformed_colmap_poses_tmp, self.polaris_poses
    #     )
    #     mean_rot_error = np.rad2deg(np.mean(rot_errs))
    #     if best_ransac is None:
    #         best_ransac = ransac_result
    #         best_rot_err = mean_rot_error
    #         continue
    #     if best_rot_err > mean_rot_error:
    #         best_ransac = ransac_result
    #         best_rot_err = mean_rot_error
    # new_pts = np.array(
    #     [reg_res.transformation @ np.hstack([pt, 1]) for pt in source_pts]
    # )
    # new_pts = new_pts[:, :3]
    # breakpoint()

    return reg_res.transformation


def find_scale_to_unit(points):
    for i, p in enumerate(points):
        if i == 0:
            max_bound = np.asarray(p, dtype=np.float32)
            min_bound = np.asarray(p, dtype=np.float32)
        else:
            temp = np.asarray(p, dtype=np.float32)
            if np.any(np.isnan(temp)):
                continue
            max_bound = np.maximum(max_bound, temp)
            min_bound = np.minimum(min_bound, temp)

    scale_to_unit = np.linalg.norm(max_bound - min_bound, ord=2)

    return scale_to_unit


def find_scale(source_pts, target_pts):
    # returns source to unit, target to unit, source to target scale
    scale_s = find_scale_to_unit(source_pts)
    scale_t = find_scale_to_unit(target_pts)
    return 1.0 / scale_s, 1.0 / scale_t, scale_t / scale_s


def register_rigid(source, target):
    """
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
    A: Nxm numpy array of corresponding points
    B: Nxm numpy array of corresponding points
    Returns:
    T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
    R: mxm rotation matrix
    t: mx1 translation vector
    """

    assert source.shape == target.shape

    # get number of dimensions
    m = source.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(source, axis=0)
    centroid_B = np.mean(target, axis=0)
    AA = source - centroid_A
    BB = target - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # breakpoint()
    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t
    # Retunrs Homogeneos and decomposition.
    return T
