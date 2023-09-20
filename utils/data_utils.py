import numpy as np
from pathlib import Path
import open3d as o3d
import cv2 as cv

from . import pose_utils


def extract_json(json_data, pose_in_m=False):
    base_dir = Path(json_data["base_dir"]).expanduser()

    img_dir = json_data["img_dir"] if "img_dir" in json_data else "images"
    img_list = sorted(Path(str(base_dir / img_dir)).glob("*.jpg"))

    seg = (
        o3d.io.read_triangle_mesh(str(base_dir / json_data["seg"]))
        if "seg" in json_data
        else None
    )

    pose_file = json_data["poses"] if "poses" in json_data else "trajectories.csv"
    poses = pose_utils.load_trajectories(str(base_dir / pose_file))
    poses, idxs = pose_utils.subsample_poses(
        poses=poses,
        indexes=(json_data["start_idx"], json_data["end_idx"]),
        interval=json_data["interval"],
    )
    if pose_in_m:
        poses = pose_utils.scale_poses(poses, 1000)

    mask = cv.imread(str(base_dir / "undistorted_mask.bmp"), cv.IMREAD_GRAYSCALE)

    ## check that poses were subsampled correctly
    assert len(img_list) == len(poses)

    return base_dir, img_list, mask, poses, idxs, seg


def load_intrinsics(intr_list):
    intrinsics = np.eye(3)
    intrinsics[0, 0] = intr_list[0]
    intrinsics[1, 1] = intr_list[1]
    intrinsics[0, 2] = intr_list[2]
    intrinsics[1, 2] = intr_list[3]

    return intrinsics
