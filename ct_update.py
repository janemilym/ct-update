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

# ! manually matched frames
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

    # * build preop TSDF volume
    preop_renders = generate_renders(
        seg=preop_seg,
        intrinsics=intrinsics,
        img_paths=preop_img_paths,
        poses=preop_poses,
        mask=preop_mask,
    )
    tsdf_vol, overall_mean_depth_value = build_tsdf(
        render_list=preop_renders,
        img_path_list=preop_img_paths,
        poses=preop_poses,
        intrinsics=intrinsics,
        mask=preop_mask,
    )
    del preop_renders
    verts, faces, norms, colors, _ = tsdf_vol.get_mesh(only_visited=True)
    tsdf.meshwrite(
        str(output_dir / "initial_fused_mesh.ply"), verts, faces, -norms, colors
    )

    # * preop CT renders at intraop poses
    intraop_renders = generate_renders(
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

    # * compare preop CT renders fr intraop poses with est depth
    # compare_intraop_renders(intraop_dir, intraop_img_paths, intraop_renders, intraop_mask, output_dir)

    # * compare warped preop CT to intraop poses with est depth
    intraop_max_depth = get_max_depth_fr_renders(intraop_renders)
    del intraop_renders
    depth_errs, warped_depth_maps = compare_warped_intraop(
        preop_data,
        preop_img_paths,
        intrinsics,
        preop_key_idxs,
        intraop_dir,
        intraop_img_paths,
        intraop_mask,
        intraop_max_depth,
        output_dir,
    )

    # # ! MANUALLY SELECTED - STABLE WARPING
    IDXS = (28, 66)
    change_masks = extract_change_mask(depth_errs, idxs=IDXS, output_dir=output_dir)

    # * feed into tsdf
    print("trying to re-integrate TSDF...")
    for i, idx in enumerate(range(IDXS[0], IDXS[1])):
        #     color_img = cv.imread(str(intraop_img_paths[idx]))
        #     color_img = cv.cvtColor(color_img, cv.COLOR_BGR2RGB)

        #     depth_img = warped_depth_maps[idx]
        color_img = np.zeros((1080, 1920, 3))
        # depth_img = np.ones((1080, 1920, 1))
        depth_img = warped_depth_maps[idx]
        # test_mask = np.ones_like(depth_img)
        test_mask = cv.resize(
            change_masks[i],
            (1080, 1920),
            interpolation=cv.INTER_AREA,
        )
        test_mask = np.expand_dims(test_mask, axis=-1)

        tsdf_vol.integrate(
            color_img,
            depth_img,
            intrinsics,
            intraop_poses[idx],
            min_depth=1.0e-3 * overall_mean_depth_value,
            std_im=np.zeros_like(depth_img),
            obs_weight=1.0,
            mask=test_mask,
        )
    verts, faces, norms, colors, _ = tsdf_vol.get_mesh(only_visited=True)
    tsdf.meshwrite(str(output_dir / "updated_mesh.ply"), verts, faces, -norms, colors)
    print("done")

    # ! quick render comparison
    init_mesh = o3d.io.read_triangle_mesh(str(output_dir / "initial_fused_mesh.ply"))
    init_renders = generate_renders(
        seg=init_mesh,
        intrinsics=intrinsics,
        img_paths=preop_img_paths,
        poses=intraop_poses,
        mask=intraop_mask,
    )
    init_imgs = []
    for i in range(len(init_renders)):
        color_img, _, _ = init_renders[i]
        color_img = cv.cvtColor(color_img, cv.COLOR_BGR2RGB)
        init_imgs.append(color_img)
    del init_renders

    updated_mesh = o3d.io.read_triangle_mesh(str(output_dir / "updated_mesh.ply"))
    updated_renders = generate_renders(
        seg=updated_mesh,
        intrinsics=intrinsics,
        img_paths=preop_img_paths,
        poses=intraop_poses,
        mask=intraop_mask,
    )
    updated_imgs = []
    for i in range(len(updated_renders)):
        color_img, _, _ = updated_renders[i]
        color_img = cv.cvtColor(color_img, cv.COLOR_BGR2RGB)
        updated_imgs.append(color_img)
    del updated_renders

    disp_list = []
    for idx, p in enumerate(intraop_img_paths):
        img = cv.imread(str(p))

        disp = cv.hconcat([init_imgs[idx], updated_imgs[idx], img])
        disp_list.append(disp)
    image_utils.save_video(disp_list, str(output_dir / "update_renders.mp4"))


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


def generate_renders(
    seg, intrinsics, img_paths, poses, mask, output_dir=None, desc=None
):
    render_list = render_utils.generate_renders(
        mesh=seg,
        poses=poses,
        intrinsics=intrinsics,
        img_width=mask.shape[1],
        img_height=mask.shape[0],
        mask=mask,
    )

    if output_dir is not None:
        render_utils.save_render_video(
            img_list=img_paths,
            mesh_render_list=render_list,
            output_dir=output_dir,
            desc=desc,
        )

    return render_list


def build_tsdf(
    render_list,
    img_path_list,
    poses,
    intrinsics,
    mask,
    trunc_margin_multiplier=10.0,
    max_voxel_count=64e6,
):
    print("Estimating voxel volume bounds...")
    vol_bnds = np.zeros((3, 2))
    overall_mean_depth_value = 0.0
    render_list.set_mask_usage(False)

    for i in tqdm(range(len(render_list))):
        _, depth_img, _ = render_list[i]
        masked_depth_img = image_utils.apply_mask(depth_img, mask)
        view_frust_pts = tsdf.get_view_frustum(depth_img, intrinsics, poses[i])
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))

        if i == 0:
            overall_mean_depth_value = np.sum(masked_depth_img) / np.sum(mask).astype(
                np.float32
            )
        else:
            overall_mean_depth_value = overall_mean_depth_value * (
                i / (i + 1.0)
            ) + np.sum(masked_depth_img) / np.sum(mask) * (1 / (i + 1.0))

    voxel_size = 0.1
    vol_dim = (vol_bnds[:, 1] - vol_bnds[:, 0]) / voxel_size
    # Adaptively change the size of one voxel to fit into the GPU memory
    volume = vol_dim[0] * vol_dim[1] * vol_dim[2]
    factor = (volume / max_voxel_count) ** (1.0 / 3.0)
    voxel_size *= factor
    print("Voxel Size: {}".format(voxel_size))

    tsdf_vol = tsdf.TSDFVolume(
        vol_bnds,
        voxel_size=voxel_size,
        trunc_margin=voxel_size * trunc_margin_multiplier,
    )

    # Loop through images and fuse them together
    tq = tqdm(total=len(render_list))
    tq.set_description("Depth fusion")
    for i in range(len(render_list)):
        # Read RGB-D images
        color_img = cv.imread(str(img_path_list[i]))
        color_img = cv.cvtColor(color_img, cv.COLOR_BGR2RGB)

        _, depth_img, _ = render_list[i]
        depth_img = np.expand_dims(depth_img, axis=-1)

        # Integrate observation into voxel volume (assume color aligned with depth)
        # We have changed the slope of the truncated distance function based on the depth std values
        # ! ASSUMES ALL CT DEPTH RENDERS HAVE ZERO UNCERTAINTY
        # NOTE: integrate function was modified to ignore uncertainty
        tsdf_vol.integrate(
            color_img,
            depth_img,
            intrinsics,
            poses[i],
            min_depth=1.0e-3 * overall_mean_depth_value,
            std_im=np.zeros_like(depth_img),
            obs_weight=1.0,
        )
        tq.update(1)

    tq.close()

    return tsdf_vol, overall_mean_depth_value


def extract_change_mask(depth_diff_list, idxs=None, output_dir=None):
    # ! this takes multiple intraop frames to isolate region of change
    if idxs is not None:
        depth_diff_list = depth_diff_list[idxs[0] : idxs[1]]

    # accumulate differences across seq
    err_vis = []
    masks = []
    count = 0 if idxs is None else idxs[0]
    for err_map in depth_diff_list:
        abs_err = np.abs(err_map)
        threshold = 0.75 * np.max(abs_err)

        err_mask = np.zeros_like(abs_err)
        err_mask[abs_err > threshold] = 255
        masks.append(err_mask)

        # ! visualization
        err_mask = np.uint8(np.repeat(err_mask[:, :, np.newaxis], 3, axis=2))

        err_img = abs_err / np.max(abs_err) * 255
        err_img = cv.applyColorMap(np.uint8(err_img), cv.COLORMAP_HOT)

        row = cv.hconcat([err_img, err_mask])
        head_w = row.shape[1]
        header = np.zeros((50, head_w, 3), dtype=np.uint8)
        header = cv.putText(
            header,
            f"Index:{count}",
            (head_w // 3, 40),
            cv.FONT_HERSHEY_PLAIN,
            3,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )
        count += 1
        disp = cv.vconcat([header, row])
        err_vis.append(disp)

    image_utils.save_video(err_vis, save_path=str(output_dir / "depth_diff_mask.mp4"))

    return masks


def compare_intraop_renders(
    intraop_dir, intraop_img_paths, intraop_renders, intraop_mask, output_dir
):
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
        #! bc of misalignment in p04 intraop seq (last part out of anatomy)
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

    plt.plot(test_errs)
    plt.savefig(str(output_dir / "render_comparison.png"))

    image_utils.save_video(
        test_img_list, save_path=str(output_dir / "direct_render.mp4")
    )


def compare_warped_intraop(
    preop_data,
    preop_img_paths,
    intrinsics,
    preop_key_idxs,
    intraop_dir,
    intraop_img_paths,
    intraop_mask,
    intraop_max_depth,
    output_dir,
):
    # * load estimated depth maps (dreco pipeline)
    depth_map_paths = sorted((intraop_dir / "estimated_depths").glob("*.npy"))
    depth_map_imgs = sorted((intraop_dir / "estimated_depths").glob("*.jpg"))

    est_max_depth = get_max_depth_fr_maps(depth_map_paths)

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

    # TODO: INTEGRATE RAFT HERE
    # * load independently computed flow maps (RAFT)
    flow_map_paths = sorted((output_dir / "raft" / "flow").glob("*.npy"))

    # * preop depth renders from CT
    preop_depth_maps, preop_depth_disp = compute_preop_depths(
        preop_data, intrinsics, idxs=np.unique(preop_key_idxs)
    )

    assert len(preop_key_idxs) == len(flow_map_paths)
    errs = []
    depth_diff_list = []
    warped_img_list = []  # visualization
    warped_depths = []
    count = 0
    for idx, intraop_p, flow_p, depth_p, depth_img_p in zip(
        preop_key_idxs,
        intraop_img_paths,
        flow_map_paths,
        depth_map_paths,
        depth_map_imgs,
    ):
        flow = np.load(flow_p, allow_pickle=True)
        flow = np.moveaxis(flow, 0, -1)

        intraop_img = cv.imread(str(intraop_p))
        # corresponding preop keyframe
        preop_img = cv.imread(str(preop_img_paths[idx]))
        preop_depth = preop_depth_maps[idx]
        preop_disp = preop_depth_disp[idx]

        # estimated INTRAOP depth
        intraop_depth = np.load(depth_p, allow_pickle=True)

        # warping PREOP depth maps from CT
        warped_depth_map = image_utils.apply_flow(preop_depth, flow)
        warped_depths.append(warped_depth_map)
        warped_depth_map = dreco_utils.downsample_image(
            warped_depth_map, start_h, end_h, start_w, end_w
        )
        warped_depth_map = image_utils.apply_mask(warped_depth_map, ds_intraop_mask)

        # compute error
        # err_tmp, _ = error_utils.scale_invariant_MSE(intraop_depth, warped_depth_map)
        # errs.append(err_tmp)

        tmp_mask_1 = np.zeros_like(warped_depth_map)
        tmp_mask_1[warped_depth_map > 0.0] = 1

        tmp_mask_2 = np.zeros_like(intraop_depth)
        tmp_mask_2[intraop_depth > 0.0] = 1

        tmp_mask = np.logical_and(tmp_mask_1, tmp_mask_2)

        depth_diff = warped_depth_map - (
            intraop_depth * intraop_max_depth / est_max_depth
        )
        depth_diff = image_utils.apply_mask(depth_diff, tmp_mask)
        depth_diff_list.append(depth_diff)
        # # ! for visualization
        # depth_diff_img = render_utils.display_depth_map(
        #     depth_diff, colormode=cv.COLORMAP_HOT
        # )
        # depth_diff_img = image_utils.apply_mask(depth_diff_img, tmp_mask).astype(
        #     np.uint8
        # )

        # # * VISUALIZATION
        # intraop_depth_img = cv.imread(str(depth_img_p))
        # intraop_depth_img = cv.applyColorMap(intraop_depth_img, cv.COLORMAP_JET)
        # intraop_depth_img = image_utils.apply_mask(intraop_depth_img, ds_intraop_mask)

        # # warping input images
        # # pre_to_intra = image_utils.apply_flow(preop_img, flow)
        # # pre_to_intra = dreco_utils.downsample_image(
        # #     pre_to_intra, start_h, end_h, start_w, end_w
        # # )
        # # pre_to_intra = image_utils.apply_mask(pre_to_intra, ds_intraop_mask)

        # # warping depth display image (normalized)
        # warped_depth_disp = image_utils.apply_flow(preop_disp, flow)
        # warped_depth_disp = dreco_utils.downsample_image(
        #     warped_depth_disp, start_h, end_h, start_w, end_w
        # )
        # warped_depth_disp = image_utils.apply_mask(warped_depth_disp, ds_intraop_mask)

        # # downsampling original images (to match depth maps)
        # ds_preop_disp = dreco_utils.downsample_image(
        #     preop_disp, start_h, end_h, start_w, end_w
        # )
        # # ds_intraop_img = dreco_utils.downsample_image(
        # #     intraop_img, start_h, end_h, start_w, end_w
        # # )
        # # ds_preop_img = dreco_utils.downsample_image(
        # #     preop_img, start_h, end_h, start_w, end_w
        # # )

        # # constructing frames for video
        # # top = cv.hconcat([ds_intraop_img, pre_to_intra, ds_preop_img])
        # # bottom = cv.hconcat([intraop_depth_img, warped_depth_disp, ds_preop_disp])
        # # head_w = top.shape[1]
        # row = cv.hconcat(
        #     [ds_preop_disp, warped_depth_disp, intraop_depth_img, depth_diff_img]
        # )
        # head_w = row.shape[1]
        # header = np.zeros((50, head_w, 3), dtype=np.uint8)
        # header = cv.putText(
        #     header,
        #     f"Index:{count}",
        #     (head_w // 3, 40),
        #     cv.FONT_HERSHEY_PLAIN,
        #     3,
        #     (255, 255, 255),
        #     2,
        #     cv.LINE_AA,
        # )
        # count += 1

        # # img_tmp = cv.vconcat([header, top, bottom])
        # img_tmp = cv.vconcat([header, row])
        # warped_img_list.append(img_tmp)

    # plt.plot(errs)
    # plt.savefig(str(output_dir / "raft_comparison.png"))

    # image_utils.save_video(
    #     warped_img_list, save_path=str(output_dir / "scaled_depth_diff.mp4")
    # )

    return np.asarray(depth_diff_list), np.asarray(warped_depths)


def get_max_depth_fr_renders(render_list):
    max_depth = 0.0
    print("Computing max depth from renders...")
    for i in tqdm(range(len(render_list))):
        _, depth_img, _ = render_list[i]
        max_depth = np.maximum(max_depth, np.amax(depth_img))

    return max_depth


def get_max_depth_fr_maps(map_path_list):
    max_depth = 0.0
    for p in map_path_list:
        depth_map = np.load(p, allow_pickle=True)
        max_depth = np.maximum(max_depth, np.amax(depth_map))

    return max_depth


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


if __name__ == "__main__":
    ct_update()
