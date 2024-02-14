import copy
import yaml
import numpy as np
from pathlib import Path
import pytransform3d.transformations as pt
import open3d as o3d

from . import register_utils


def plot_and_save_trajectory(
    poses,
    axlen=1,
    subsampling_factor=1,
    save_name="trajectory.ply",
    draw_connections=True,
    connection_color=[1, 0, 0],
    show=False,
):
    tm = []  # Temporal transformations
    transformation_matrices = np.empty((len(poses), 4, 4))
    points = []
    for j, cam_pose in enumerate(poses):
        if j % subsampling_factor != 0:
            continue
        # rot = np.transpose(cam_pose[0:3, 0:3])
        rot = cam_pose[0:3, 0:3]
        transl = cam_pose[:3, 3]
        # transl = np.matmul(-np.transpose(rot), cam_pose[:3, 3])
        points.append(transl)
        tm.append(pt.transform_from(R=rot, p=transl))
    transformation_matrices = np.asarray(tm)
    trajectory = None
    for i, pose in enumerate(transformation_matrices):
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axlen)
        mesh_frame.transform(pose)
        if i == 0:
            trajectory = copy.deepcopy(mesh_frame)
        else:
            trajectory += copy.deepcopy(mesh_frame)
    if draw_connections:
        lines = []
        for i in range(len(points) - 1):
            lines.append([i, i + 1])
        colors = [connection_color for i in range(len(lines))]
        line_set = o3d.geometry.LineSet()
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
    if show:
        o3d.visualization.draw_geometries([trajectory, line_set])
    o3d.io.write_triangle_mesh(save_name, trajectory)


def load_trajectories(trajectory_path):
    if not Path(trajectory_path).exists():
        raise FileNotFoundError
    trajectory = np.genfromtxt(trajectory_path, delimiter=",")
    num_poses = int(len(trajectory) / 16)
    trajectory = trajectory.reshape((num_poses, 4, 4))
    return trajectory


def subsample_poses(poses, indexes, interval=1):
    start_idx, end_idx = indexes

    selected_poses = []
    idx_list = []
    for i, pose in enumerate(poses):
        if (i - start_idx) % interval == 0 and i >= start_idx and (i) <= end_idx:
            selected_poses.append(pose)
            idx_list.append(i)

    return np.asarray(selected_poses), np.asarray(idx_list)


def quaternion_matrix(quaternion):
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < np.finfo(float).eps * 4.0:
        return np.identity(4)
    q *= np.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array(
        [
            [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
            [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
            [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def get_extrinsic_matrix(poses):
    extrinsic_matrices = []
    visible_view_count = len(poses)
    for i in range(visible_view_count):
        rigid_transform = quaternion_matrix(
            [
                poses["poses[" + str(i) + "]"]["orientation"]["w"],
                poses["poses[" + str(i) + "]"]["orientation"]["x"],
                poses["poses[" + str(i) + "]"]["orientation"]["y"],
                poses["poses[" + str(i) + "]"]["orientation"]["z"],
            ]
        )
        rigid_transform[0][3] = poses["poses[" + str(i) + "]"]["position"]["x"]
        rigid_transform[1][3] = poses["poses[" + str(i) + "]"]["position"]["y"]
        rigid_transform[2][3] = poses["poses[" + str(i) + "]"]["position"]["z"]
        transform = np.asmatrix(rigid_transform)
        extrinsic_matrices.append(transform)

    return np.asarray(extrinsic_matrices)


def read_colmap_trajectory(trajectory_path):
    stream = open(trajectory_path, "r")
    doc = yaml.safe_load(stream)
    _, values = doc.items()
    poses = values[1]
    mat_poses = get_extrinsic_matrix(poses)

    return mat_poses


def invert_poses(poses):
    copy_poses = copy.deepcopy(poses)
    for i, pose in enumerate(copy_poses):
        pose[:3, :3] = np.transpose(poses[i, :3, :3])
        pose[:3, 3] = np.matmul(-np.transpose(poses[i, :3, :3]), poses[i, :3, 3])

    return copy_poses


def invert_pose(pose):
    inverted = copy.deepcopy(pose)
    inverted[:3, :3] = np.transpose(pose[:3, :3])
    inverted[:3, 3] = np.matmul(-np.transpose(pose[:3, :3]), pose[:3, 3])

    return inverted


def scale_poses(poses, scale):
    scaled_poses_list = []
    poses_copy = copy.deepcopy(poses)
    for pose in poses_copy:
        pose[:3, 3] *= scale
        scaled_poses_list.append(pose)
    scaled_poses_list = np.asarray(scaled_poses_list)

    return scaled_poses_list


def transform_poses(poses, transformation):
    transformed_pose = []
    for pose in poses:
        transformed_pose.append(transformation @ pose)
    transformed_pose = np.asarray(transformed_pose)
    return transformed_pose


def load_colmap_yaml(poses_path, invert=True):
    colmap_poses = read_colmap_trajectory(poses_path)
    if invert:
        colmap_poses = invert_poses(colmap_poses)
    return colmap_poses


def save_poses(poses, save_path):
    """
    Args:
        poses: n x 4 x 4 numpy array of camera poses
    """
    np.savetxt(save_path, poses.flatten(), delimiter=",")
    print(f"Camera poses saved to: {save_path}")


def find_nearest_pose(query, poses):
    """
    Finds the nearest pose in the sequence to the query

    ARGS
    query: 4 x 4 query camera trajectory
    poses: n x 4 x 4 camera trajectory sequence
    """

    seq_pos = poses[:, :3, 3]
    q_pos = query[:3, 3]

    dist = np.linalg.norm(seq_pos - q_pos, axis=1)
    # sorted_idxs = np.argsort(dist)

    # ## check view
    # axis_errs = []
    # for i in sorted_idxs:
    #     check = error_utils.axis_angle_err(poses[i, :3, :3], query[:3, :3])
    #     axis_errs.append(np.rad2deg(check))

    # breakpoint()

    return np.argmin(dist)


def register_poses(source, target):
    ## extract correspondences (camera centers)
    source_pts = np.array([pose[:3, 3] for pose in source])
    target_pts = np.array([pose[:3, 3] for pose in target])

    transform = register_utils.register_rigid(source=source_pts, target=target_pts)

    transformed_pts = np.array([transform @ np.hstack([pt, 1]) for pt in source_pts])
    transformed_pts = transformed_pts[:, :3]

    res_err = np.linalg.norm(transformed_pts - target_pts, axis=1)

    return transform, np.mean(res_err)


def calculate_pose_error(poses_1, poses_2):
    assert poses_1.shape == poses_2.shape

    rot_err_list = []
    tr_err_list = []

    for idx, p in enumerate(poses_1):
        rot = p[:3, :3] @ poses_2[idx, :3, :3]
        rot_err = np.arccos((np.trace(rot) - 1) / 2)
        rot_err_list.append(rot_err)

        tr_err = np.linalg.norm(p[:3, 3] - poses_2[idx, :3, 3])
        tr_err_list.append(tr_err)

    return np.array(rot_err_list), np.array(tr_err_list)
