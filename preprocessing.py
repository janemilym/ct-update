import click
import json
from pathlib import Path
import cv2 as cv

from utils import *


@click.command()
@click.option(
    "--input",
    default="./data/p04_left/input.json",
    required=True,
    help="input json file",
)
@click.option(
    "--downsample",
    type=bool,
    default=True,
    help="whether or not downsample images from original",
)
def preprocess(input, downsample):
    with open(input) as f:
        args = json.load(f)

    intrinsics = data_utils.load_intrinsics(
        [float(i) for i in args["intrinsics"].split(",")]
    )
    preop_data = args["preop_data"]
    # intraop_data = args["intraop_data"]

    (
        preop_dir,
        preop_img_paths,
        preop_mask,
        preop_poses,
        preop_idxs,
        preop_seg,
    ) = data_utils.extract_json(preop_data)

    (
        ds_preop_mask,
        _,
        _,
        _,
        _,
    ) = dreco_utils.downsample_and_crop_mask(
        preop_mask, suggested_h=256, suggested_w=320
    )

    if downsample:
        mask = ds_preop_mask
        cv.imwrite(str(preop_dir / "downsampled_mask.bmp"), ds_preop_mask)
    else:
        mask = preop_mask

    render_utils.generate_renders(
        mesh=preop_seg,
        poses=preop_poses,
        intrinsics=intrinsics,
        img_width=mask.shape[1],
        img_height=mask.shape[0],
        mask=mask,
        save_dir=preop_dir,
        save_option=["depth_map"],
    )


if __name__ == "__main__":
    preprocess()
