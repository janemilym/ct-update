import click
import json
import open3d as o3d
import cv2 as cv
from pathlib import Path
from tqdm import tqdm

# local
from utils import *

SRC_IDX = 727
DST_IDX = 80


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

    preop_img_paths, preop_mask, _, _, _ = extract_info(preop_data)
    intraop_img_paths, intraop_mask, _, _, _ = extract_info(intraop_data)

    # preop_idxs = image_utils.find_best_matches(
    #     src_img_paths=preop_img_paths,
    #     dst_img_paths=intraop_img_paths,
    #     src_mask=preop_mask,
    #     dst_mask=intraop_mask,
    #     output_dir=output_dir,
    # )

    # preop_idxs = get_closest_preop(preop_data, intraop_data)
    preop_depths = compute_preop_depths(preop_data, intrinsics, idxs=[727])

    # ct depths at intraop frames
    warp_preop_to_intraop(preop_data, intraop_data, preop_depths, output_dir)

    img = cv.imread(str(intraop_img_paths[DST_IDX]))
    cv.imwrite(f"test_intraop_{intraop_img_paths[DST_IDX].stem}.png", img)

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


def get_closest_preop(preop_json, intraop_json):
    #! does not work
    ## may refine later to return multiple preop idxs per intraop pose
    _, _, preop_poses, _, _ = extract_info(preop_json)
    _, _, intraop_poses, _, _ = extract_info(intraop_json, pose_in_m=True)

    preop_idxs = []
    for pose in intraop_poses:
        idx_tmp = pose_utils.find_nearest_pose(pose, preop_poses)
        preop_idxs.append(idx_tmp)

    return preop_idxs


def compute_preop_depths(preop_json, intrinsics, idxs, output_dir=None):
    preop_img_paths, preop_mask, preop_poses, _, preop_seg = extract_info(preop_json)

    preop_renders = render_utils.generate_renders(
        mesh=preop_seg,
        poses=preop_poses,
        intrinsics=intrinsics,
        img_width=preop_mask.shape[1],
        img_height=preop_mask.shape[0],
        mask=preop_mask,
    )

    print("Generating preop depth renders...")
    preop_depths = dict()
    for i in tqdm(idxs):
        color, depth, depth_disp = preop_renders[i]
        preop_depths[i] = depth_disp

    if output_dir is not None:
        render_utils.save_render_video(
            img_list=preop_img_paths,
            mesh_render_list=preop_renders,
            output_dir=output_dir,
            desc="preop",
        )

    return preop_depths


def warp_preop_to_intraop(preop_json, intraop_json, preop_depths, output_dir=None):
    preop_img_paths, preop_mask, _, _, _ = extract_info(preop_json)
    intraop_img_paths, intraop_mask, _, _, _ = extract_info(
        intraop_json, pose_in_m=True
    )
    # selected_preop_imgs = [preop_img_paths[i] for i in set(preop_idxs)]

    # homography with SIFT
    # preop_kps, preop_des = image_utils.extract_keypoints_dict(
    #     selected_preop_imgs, mask=preop_mask
    # )
    # intraop_kps, intraop_des = image_utils.extract_keypoints_dict(
    #     intraop_img_paths, mask=intraop_mask
    # )

    # print("Warping preop to intraop...")
    # test_imgs = []
    # assert len(preop_idxs) == len(intraop_img_paths)
    # for dst_idx in tqdm(range(len(intraop_img_paths))):
    #     src_idx = preop_idxs[dst_idx]

    #     src_img_path = preop_img_paths[src_idx]
    #     dst_img_path = intraop_img_paths[dst_idx]

    #     src_kps, dst_kps = image_utils.match_keypoints(
    #         preop_kps[src_img_path.stem],
    #         preop_des[src_img_path.stem],
    #         intraop_kps[dst_img_path.stem],
    #         intraop_des[dst_img_path.stem],
    #     )
    #     warped, dst_img = image_utils.warp_image(
    #         src_img_path, dst_img_path, src_kps, dst_kps, intraop_mask
    #     )

    #     img_tmp = cv.hconcat([warped, dst_img])

    #     # src_img = cv.imread(str(src_img_path))
    #     # dst_img = cv.imread(str(dst_img_path))
    #     # img_tmp = cv.hconcat([src_img, dst_img])

    #     test_imgs.append(img_tmp)

    # image_utils.save_video(test_imgs, str(output_dir / "warped.mp4"))

    src_img_path = preop_img_paths[SRC_IDX]
    dst_img_path = intraop_img_paths[DST_IDX]
    src_img = cv.imread(str(src_img_path))
    dst_img = cv.imread(str(dst_img_path))

    _, preop_kps, preop_des = image_utils.extract_keypoints(
        src_img_path, mask=preop_mask
    )
    _, intra_kps, intra_des = image_utils.extract_keypoints(
        dst_img_path, mask=intraop_mask
    )

    matches, src_kps, dst_kps = image_utils.match_keypoints(
        src_kps=preop_kps, src_des=preop_des, dst_kps=intra_kps, dst_des=intra_des
    )

    match_img = cv.drawMatches(
        src_img,
        preop_kps,
        dst_img,
        intra_kps,
        matches[:50],
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(0, 255, 0),
    )
    cv.imwrite("test_correspondences.png", match_img)
    test_warped = image_utils.warp_image(
        src_img, dst_img, src_kps, dst_kps, intraop_mask
    )
    cv.imwrite("test_warped.png", test_warped)
    overlay = cv.addWeighted(dst_img, 0.5, test_warped, 0.5, 0.0)
    cv.imwrite("test_overlay.png", overlay)

    warped_depth = image_utils.warp_image(
        preop_depths[SRC_IDX], dst_img, src_kps, dst_kps, intraop_mask
    )
    cv.imwrite("test_warped_depth.png", warped_depth)

    return


if __name__ == "__main__":
    ct_update()
