import click
import json
import open3d as o3d
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
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

    (
        preop_dir,
        preop_img_paths,
        preop_mask,
        preop_poses,
        preop_idxs,
        preop_seg,
    ) = extract_info(preop_data)
    (
        intraop_dir,
        intraop_img_paths,
        intraop_mask,
        intraop_poses,
        intraop_idxs,
        _,
    ) = extract_info(intraop_data)
    # * preop CT renders at intraop poses
    intraop_renders = generate_intraop_renders(
        preop_seg, intrinsics, intraop_img_paths, intraop_poses, intraop_mask
    )

    # * get indexes of preop keyframes corresponding to intraop seq
    preop_key_idxs = get_closest_preop(preop_poses, intraop_poses)
    # save_closest_preop_vid(
    #     preop_img_paths, preop_key_idxs, intraop_img_paths, output_dir
    # )
    # np.savetxt(
    #     str(output_dir / "preop_key_idxs.txt"), preop_key_idxs, fmt="%d", delimiter="\n"
    # )

    # * load estimated depth maps (dreco pipeline)
    depth_map_paths = sorted((intraop_dir / "estimated_depths").glob("*.npy"))
    depth_map_imgs = sorted((intraop_dir / "estimated_depths").glob("*.jpg"))

    init_h, init_w, _ = cv.imread(str(depth_map_imgs[0])).shape
    (
        ds_intraop_mask,
        start_h,
        end_h,
        start_w,
        end_w,
    ) = dreco_utils.downsample_and_crop_mask(
        intraop_mask, suggested_h=init_h, suggested_w=init_w
    )

    # # ! DIRECT RENDERS FR INTRAOP POSES
    test_errs = []
    test_img_list = []
    idx = 0
    for intraop_p, depth_p, depth_img_p in zip(
        intraop_img_paths, depth_map_paths, depth_map_imgs
    ):
        if idx > 190:
            break
        intraop_img = cv.imread(str(intraop_p))

        # estimated INTRAOP depth
        est_intraop_depth = np.load(depth_p, allow_pickle=True)

        # PREOP ct depths at INTRAOP poses
        color, depth, depth_disp = intraop_renders[idx]
        depth = dreco_utils.downsample_image(depth, start_h, end_h, start_w, end_w)

        # compute error
        err_tmp, _ = error_utils.scale_invariant_MSE(est_intraop_depth, depth)
        test_errs.append(err_tmp)

        # * VISUALIZATION
        # estimated intraop depth display
        intraop_depth_img = cv.imread(str(depth_img_p))
        intraop_depth_img = cv.applyColorMap(intraop_depth_img, cv.COLORMAP_JET)
        intraop_depth_img = image_utils.apply_mask(intraop_depth_img, ds_intraop_mask)

        # downsample everything
        ds_intraop_img = dreco_utils.downsample_image(
            intraop_img, start_h, end_h, start_w, end_w
        )
        ds_color = dreco_utils.downsample_image(color, start_h, end_h, start_w, end_w)
        ds_depth_disp = dreco_utils.downsample_image(
            depth_disp, start_h, end_h, start_w, end_w
        )

        # build frames for vid
        top = cv.hconcat([ds_intraop_img, ds_color])
        bottom = cv.hconcat([intraop_depth_img, ds_depth_disp])
        head_w = top.shape[1]
        header = np.zeros((50, head_w, 3), dtype=np.uint8)
        header = cv.putText(
            header,
            f"Index:{idx}",
            (head_w // 3, 40),
            cv.FONT_HERSHEY_PLAIN,
            3,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )
        idx += 1

        img_tmp = cv.vconcat([header, top, bottom])
        test_img_list.append(img_tmp)

    breakpoint()
    plt.plot(test_errs)
    plt.savefig(str(output_dir / "render_comparison.png"))

    image_utils.save_video(
        test_img_list, save_path=str(output_dir / "direct_render.mp4")
    )

    # # TODO: INTEGRATE RAFT HERE
    # # * load independently computed flow maps (RAFT)
    # flow_map_paths = sorted((output_dir / "raft" / "flow").glob("*.npy"))

    # # * preop depth renders from CT
    # preop_depth_maps, preop_depth_disp = compute_preop_depths(
    #     preop_data, intrinsics, idxs=preop_key_idxs
    # )

    # # TODO: put this in helper function this is so ugly
    # assert len(preop_key_idxs) == len(flow_map_paths)
    # errs = []
    # warped_img_list = []  # visualization
    # count = 0
    # for idx, intraop_p, flow_p, depth_p, depth_img_p in zip(
    #     preop_key_idxs,
    #     intraop_img_paths,
    #     flow_map_paths,
    #     depth_map_paths,
    #     depth_map_imgs,
    # ):
    #     flow = np.load(flow_p, allow_pickle=True)
    #     flow = np.moveaxis(flow, 0, -1)

    #     intraop_img = cv.imread(str(intraop_p))
    #     # corresponding preop keyframe
    #     preop_img = cv.imread(str(preop_img_paths[idx]))
    #     preop_depth = preop_depth_maps[idx]
    #     preop_disp = preop_depth_disp[idx]

    #     # estimated INTRAOP depth
    #     intraop_depth = np.load(depth_p, allow_pickle=True)

    #     # warping PREOP depth maps from CT
    #     warped_depth_map = image_utils.apply_flow(preop_depth, flow)
    #     warped_depth_map = dreco_utils.downsample_image(
    #         warped_depth_map, start_h, end_h, start_w, end_w
    #     )
    #     warped_depth_map = image_utils.apply_mask(warped_depth_map, ds_intraop_mask)

    #     # compute error
    #     err_tmp, _ = error_utils.scale_invariant_MSE(intraop_depth, warped_depth_map)
    #     errs.append(err_tmp)

    #     # * VISUALIZATION
    #     intraop_depth_img = cv.imread(str(depth_img_p))
    #     intraop_depth_img = cv.applyColorMap(intraop_depth_img, cv.COLORMAP_JET)
    #     intraop_depth_img = image_utils.apply_mask(intraop_depth_img, ds_intraop_mask)

    #     # warping images
    #     pre_to_intra = image_utils.apply_flow(preop_img, flow)
    #     pre_to_intra = dreco_utils.downsample_image(
    #         pre_to_intra, start_h, end_h, start_w, end_w
    #     )
    #     pre_to_intra = image_utils.apply_mask(pre_to_intra, ds_intraop_mask)

    #     # warping depth display image (normalized)
    #     warped_depth_disp = image_utils.apply_flow(preop_disp, flow)
    #     warped_depth_disp = dreco_utils.downsample_image(
    #         warped_depth_disp, start_h, end_h, start_w, end_w
    #     )
    #     warped_depth_disp = image_utils.apply_mask(warped_depth_disp, ds_intraop_mask)

    #     # downsampling original images (to match depth maps)
    #     ds_preop_disp = dreco_utils.downsample_image(
    #         preop_disp, start_h, end_h, start_w, end_w
    #     )
    #     ds_intraop_img = dreco_utils.downsample_image(
    #         intraop_img, start_h, end_h, start_w, end_w
    #     )
    #     ds_preop_img = dreco_utils.downsample_image(
    #         preop_img, start_h, end_h, start_w, end_w
    #     )

    #     # constructing frames for video
    #     top = cv.hconcat([ds_intraop_img, pre_to_intra, ds_preop_img])
    #     bottom = cv.hconcat([intraop_depth_img, warped_depth_disp, ds_preop_disp])
    #     head_w = top.shape[1]
    #     header = np.zeros((50, head_w, 3), dtype=np.uint8)
    #     header = cv.putText(
    #         header,
    #         f"Index:{count}",
    #         (head_w // 3, 40),
    #         cv.FONT_HERSHEY_PLAIN,
    #         3,
    #         (255, 255, 255),
    #         2,
    #         cv.LINE_AA,
    #     )
    #     count += 1

    #     img_tmp = cv.vconcat([header, top, bottom])
    #     warped_img_list.append(img_tmp)

    # plt.plot(errs)
    # plt.savefig(str(output_dir / "raft_comparison.png"))

    # image_utils.save_video(
    #     warped_img_list, save_path=str(output_dir / "raft_warped.mp4")
    # )


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

    return base_dir, img_list, mask, poses, idxs, seg


def generate_intraop_renders(
    preop_seg,
    intrinsics,
    intraop_img_paths,
    intraop_poses,
    intraop_mask,
    output_dir=None,
):
    intraop_renders = render_utils.generate_renders(
        mesh=preop_seg,
        poses=intraop_poses,
        intrinsics=intrinsics,
        img_width=intraop_mask.shape[1],
        img_height=intraop_mask.shape[0],
        mask=intraop_mask,
    )
    if output_dir is not None:
        render_utils.save_render_video(
            img_list=intraop_img_paths,
            mesh_render_list=intraop_renders,
            output_dir=output_dir,
            desc="intraop",
        )

    return intraop_renders


def get_closest_preop(preop_poses, intraop_poses):
    #! refine later to return multiple preop idxs per intraop pose
    preop_idxs = []
    for pose in intraop_poses:
        idx_tmp = pose_utils.find_nearest_pose(pose, preop_poses)
        preop_idxs.append(idx_tmp)

    return preop_idxs


def save_closest_preop_vid(
    preop_img_paths, preop_key_idxs, intraop_img_paths, output_dir
):
    img_list = []
    for idx, intraop_path in zip(preop_key_idxs, intraop_img_paths):
        closest_preop_path = preop_img_paths[idx]
        preop_img = cv.imread(str(closest_preop_path))
        intraop_img = cv.imread(str(intraop_path))

        img_tmp = cv.hconcat([preop_img, intraop_img])
        img_list.append(img_tmp)

    image_utils.save_video(
        img_list, save_path=str(output_dir / "closest_preop_pose.mp4")
    )


def compute_preop_depths(preop_json, intrinsics, idxs, output_dir=None):
    (
        preop_dir,
        preop_img_paths,
        preop_mask,
        preop_poses,
        preop_idxs,
        preop_seg,
    ) = extract_info(preop_json)
    preop_renders = render_utils.generate_renders(
        mesh=preop_seg,
        poses=preop_poses,
        intrinsics=intrinsics,
        img_width=preop_mask.shape[1],
        img_height=preop_mask.shape[0],
        mask=preop_mask,
    )

    print("Generating preop depth renders...")
    preop_depth_maps = dict()
    preop_depth_disp = dict()
    for i in tqdm(idxs):
        color, depth, depth_disp = preop_renders[i]
        preop_depth_maps[i] = depth
        preop_depth_disp[i] = depth_disp

    if output_dir is not None:
        render_utils.save_render_video(
            img_list=preop_img_paths,
            mesh_render_list=preop_renders,
            output_dir=output_dir,
            desc="preop",
        )

    return preop_depth_maps, preop_depth_disp


# def warp_preop_to_intraop(preop_json, intraop_json, preop_depths, output_dir=None):
#     preop_img_paths, preop_mask, _, _, _ = extract_info(preop_json)
#     intraop_img_paths, intraop_mask, _, _, _ = extract_info(
#         intraop_json, pose_in_m=True
#     )
#     # selected_preop_imgs = [preop_img_paths[i] for i in set(preop_idxs)]

#     # homography with SIFT
#     # preop_kps, preop_des = image_utils.extract_keypoints_dict(
#     #     selected_preop_imgs, mask=preop_mask
#     # )
#     # intraop_kps, intraop_des = image_utils.extract_keypoints_dict(
#     #     intraop_img_paths, mask=intraop_mask
#     # )

#     # print("Warping preop to intraop...")
#     # test_imgs = []
#     # assert len(preop_idxs) == len(intraop_img_paths)
#     # for dst_idx in tqdm(range(len(intraop_img_paths))):
#     #     src_idx = preop_idxs[dst_idx]

#     #     src_img_path = preop_img_paths[src_idx]
#     #     dst_img_path = intraop_img_paths[dst_idx]

#     #     src_kps, dst_kps = image_utils.match_keypoints(
#     #         preop_kps[src_img_path.stem],
#     #         preop_des[src_img_path.stem],
#     #         intraop_kps[dst_img_path.stem],
#     #         intraop_des[dst_img_path.stem],
#     #     )
#     #     warped, dst_img = image_utils.warp_image(
#     #         src_img_path, dst_img_path, src_kps, dst_kps, intraop_mask
#     #     )

#     #     img_tmp = cv.hconcat([warped, dst_img])

#     #     # src_img = cv.imread(str(src_img_path))
#     #     # dst_img = cv.imread(str(dst_img_path))
#     #     # img_tmp = cv.hconcat([src_img, dst_img])

#     #     test_imgs.append(img_tmp)

#     # image_utils.save_video(test_imgs, str(output_dir / "warped.mp4"))

#     src_img_path = preop_img_paths[SRC_IDX]
#     dst_img_path = intraop_img_paths[DST_IDX]
#     src_img = cv.imread(str(src_img_path))
#     dst_img = cv.imread(str(dst_img_path))

#     _, preop_kps, preop_des = image_utils.extract_keypoints(
#         src_img_path, mask=preop_mask
#     )
#     _, intra_kps, intra_des = image_utils.extract_keypoints(
#         dst_img_path, mask=intraop_mask
#     )

#     matches, src_kps, dst_kps = image_utils.match_keypoints(
#         src_kps=preop_kps, src_des=preop_des, dst_kps=intra_kps, dst_des=intra_des
#     )

#     match_img = cv.drawMatches(
#         src_img,
#         preop_kps,
#         dst_img,
#         intra_kps,
#         matches[:50],
#         None,
#         matchColor=(0, 255, 0),
#         singlePointColor=(0, 255, 0),
#     )
#     cv.imwrite("test_correspondences.png", match_img)
#     test_warped = image_utils.warp_image(
#         src_img, dst_img, src_kps, dst_kps, intraop_mask
#     )
#     cv.imwrite("test_warped.png", test_warped)
#     overlay = cv.addWeighted(dst_img, 0.5, test_warped, 0.5, 0.0)
#     cv.imwrite("test_overlay.png", overlay)

#     warped_depth = image_utils.warp_image(
#         preop_depths[SRC_IDX], dst_img, src_kps, dst_kps, intraop_mask
#     )
#     cv.imwrite("test_warped_depth.png", warped_depth)

#     return


if __name__ == "__main__":
    ct_update()
