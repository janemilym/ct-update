import json
import numpy as np
from pathlib import Path
import open3d as o3d
import cv2 as cv

from . import pose_utils
from . import dreco_utils as dreco
from . import render_utils


def extract_json(json_file):
    with open(json_file) as f:
        args = json.load(f)

    intrinsics = load_intrinsics([float(i) for i in args["intrinsics"].split(",")])
    data_dir = Path(args["data_dir"]).expanduser()

    preop_data = args["preop_data"]
    # intraop_data = args["intraop_data"]

    # intraop_data = []
    # for i in range(1, args["steps"] + 1):
    #     intraop_data.append(args[f"{i}"])

    preop = PatientData(preop_data, intrinsics)
    intraop = []
    for i in range(1, args["steps"] + 1):
        bite_data = args[f"{i}"]
        intraop.append(PatientData(bite_data, intrinsics, downsample=True))

    return data_dir, intrinsics, preop, intraop


class PatientData:
    def __init__(self, json_data, intrinsics, downsample=False):
        self.base_dir = Path(json_data["base_dir"]).expanduser()

        img_dir = json_data["img_dir"] if "img_dir" in json_data else "images"
        self.img_paths = sorted(Path(str(self.base_dir / img_dir)).glob("*.jpg"))
        self.images = [cv.imread(str(img_path)) for img_path in self.img_paths]

        self.seg_path = self.base_dir / json_data["seg"]

        self.seg = (
            o3d.io.read_triangle_mesh(str(self.seg_path))
            if "seg" in json_data
            else None
        )

        pose_file = json_data["poses"] if "poses" in json_data else "trajectories.csv"
        poses = pose_utils.load_trajectories(str(self.base_dir / pose_file))

        self.poses, self.idxs = pose_utils.subsample_poses(
            poses=poses,
            indexes=(json_data["start_idx"], json_data["end_idx"]),
            interval=json_data["interval"],
        )
        # if pose_in_m:
        #     poses = pose_utils.scale_poses(poses, 1000)

        # ! PROBLEM WITH ALIGNED POSES
        # if "gt_poses" in json_data:
        #     gt_poses = pose_utils.load_trajectories(
        #         str(self.base_dir / json_data["gt_poses"])
        #     )

        #     self.gt_poses, _ = pose_utils.subsample_poses(
        #         poses=gt_poses,
        #         indexes=(json_data["start_idx"], json_data["end_idx"]),
        #         interval=json_data["interval"],
        #     )
        #     breakpoint()

        #     # NOTE: registering here to use all poses
        #     self.T, _ = pose_utils.register_poses(
        #         source=gt_poses[: json_data["end_idx"]],
        #         target=poses[: json_data["end_idx"]],
        #     )

        # ! debug vis
        # transformed_poses = pose_utils.transform_poses(gt_poses, self.T)
        # pose_utils.plot_and_save_trajectory(
        #     transformed_poses, save_name="test_transformed_poses.ply"
        # )
        # pose_utils.plot_and_save_trajectory(
        #     poses, save_name="test_poses_in_preop.ply"
        # )
        # pose_utils.plot_and_save_trajectory(
        #     gt_poses, save_name="test_orig_intraop.ply"
        # )

        self.mask = cv.imread(
            str(self.base_dir / "undistorted_mask.bmp"), cv.IMREAD_GRAYSCALE
        )

        self.intrinsics = intrinsics

        if downsample:
            self.downsample()

    def downsample(
        self,
        downsampling_factor=4.0,
        divide=64,
        suggested_h=256,
        suggested_w=320,
    ):
        (
            self.mask,
            self.start_h,
            self.end_h,
            self.start_w,
            self.end_w,
        ) = dreco.downsample_and_crop_mask(
            self.mask,
            downsampling_factor=downsampling_factor,
            divide=divide,
            suggested_h=suggested_h,
            suggested_w=suggested_w,
        )

        self.intrinsics = dreco.modify_camera_intrinsic_matrix(
            self.intrinsics,
            self.start_h,
            self.start_w,
            downsampling_factor=downsampling_factor,
        )

        self.images = [
            dreco.downsample_image(
                img=img,
                start_h=self.start_h,
                end_h=self.end_h,
                start_w=self.start_w,
                end_w=self.end_w,
            )
            for img in self.images
        ]

    def generate_renders(self, seg=None, mask=None, output_dir=None, desc=None):
        if self.seg is None:
            assert seg is not None
        elif seg is None:
            seg = self.seg

        if mask is None:
            mask = self.mask

        renders = render_utils.generate_renders(
            mesh=seg,
            poses=self.poses,
            intrinsics=self.intrinsics,
            img_width=mask.shape[1],
            img_height=mask.shape[0],
            mask=mask,
        )

        if output_dir is not None:
            render_utils.save_render_video(
                img_list=self.images,
                mesh_render_list=renders,
                output_dir=output_dir,
                desc=desc,
            )

        return renders


def load_mesh(mesh_path):
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    return mesh


def load_intrinsics(intr_list):
    intrinsics = np.eye(3)
    intrinsics[0, 0] = intr_list[0]
    intrinsics[1, 1] = intr_list[1]
    intrinsics[0, 2] = intr_list[2]
    intrinsics[1, 2] = intr_list[3]

    return intrinsics
