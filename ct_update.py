import click
import json
import open3d as o3d
import cv2 as cv
from pathlib import Path

# local
from utils import *


@click.command()
@click.option("--input", required=True, help="input arguments in json for CT update")
def ct_update(input):
    with open(input) as f:
        args = json.load(f)

    intrinsics = data_utils.load_intrinsics(
        [float(i) for i in args["intrinsics"].split(",")]
    )
    output_dir = Path(args["output_dir"]).expanduser()

    preop_data = args["preop_data"]
    intraop_data = args["intraop_data"]
    # preop_depths = compute_preop_depths(preop_data, intrinsics, output_dir)

    warp_intraop_to_preop(preop_data, intraop_data, output_dir)

    return


def extract_info(json_data, pose_in_m=False):
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

    return img_list, mask, poses, idxs, seg


def compute_preop_depths(preop_json, intrinsics, output_dir=None):
    preop_img_paths, preop_mask, preop_poses, _, preop_seg = extract_info(
        preop_json["preop_data"]
    )

    preop_renders, preop_depths = render_utils.generate_renders(
        mesh=preop_seg,
        poses=preop_poses,
        intrinsics=intrinsics,
        img_width=preop_mask.shape[1],
        img_height=preop_mask.shape[0],
        mask=preop_mask,
    )

    if output_dir is not None:
        render_utils.save_render_video(
            img_list=preop_img_paths,
            mesh_render_list=preop_renders,
            output_dir=output_dir,
            desc="preop",
        )
    del preop_renders

    return preop_depths


def warp_intraop_to_preop(preop_json, intraop_json, output_dir=None):
    preop_img_paths, _, _, _, _ = extract_info(preop_json)
    intraop_img_paths, intraop_mask, intraop_poses, intraop_idxs, _ = extract_info(
        intraop_json, pose_in_m=True
    )
    image_utils.extract_keypoints(
        intraop_img_paths, mask=intraop_mask, output_dir=output_dir, desc="intraop"
    )
    return


if __name__ == "__main__":
    ct_update()
