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

    preop_img_paths, preop_mask, preop_poses, preop_idxs, seg = extract_info(
        args["preop_data"]
    )
    intraop_img_paths, intraop_mask, intraop_poses, intraop_idxs, _ = extract_info(
        args["intraop_data"]
    )
    intrinsics = data_utils.load_intrinsics(
        [float(i) for i in args["intrinsics"].split(",")]
    )

    output_dir = Path(args["output_dir"]).expanduser()

    preop_render_dir = output_dir / "preop_renders"
    preop_render_dir.mkdir(parents=True, exist_ok=True)
    preop_renders = render_utils.generate_renders(
        mesh=seg,
        poses=preop_poses,
        intrinsics=intrinsics,
        img_width=preop_mask.shape[1],
        img_height=preop_mask.shape[0],
        mask=preop_mask,
    )
    render_utils.save_render_video(
        img_list=preop_img_paths,
        mesh_render_list=preop_renders,
        output_dir=preop_render_dir,
        desc="preop",
    )

    ## generate preop fused mesh based on CT
    ## render depths at camera poses + depth fusion

    return


def extract_info(json_data):
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

    mask = cv.imread(str(base_dir / "undistorted_mask.bmp"), cv.IMREAD_GRAYSCALE)

    ## check that poses were subsampled correctly
    assert len(img_list) == len(poses)

    return img_list, mask, poses, idxs, seg


if __name__ == "__main__":
    ct_update()
