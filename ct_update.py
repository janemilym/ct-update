import click
import copy
import time

import cv2 as cv
import numpy as np
from tqdm import tqdm


# local
from utils import *

@click.command()
@click.option("--input", default="./data/p04_left/input.json", required=True, help="input arguments in json for CT update")
@click.option("--desc", default="output", help="description for experiments")
def ct_update(input, desc):
    runtime = []

    data_dir, intrinsics, preop, intraop = data_utils.extract_json(input)

    output_dir = data_dir / desc
    output_dir.mkdir(parents=True, exist_ok=True)

    # * create directories for outputs
    mesh_dir = output_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    change_dir = output_dir / "changes"
    change_dir.mkdir(parents=True, exist_ok=True)

    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    preop_start = time.time()
    # * build preop TSDF volume
    preop_renders = preop.generate_renders()
    tsdf_vol, overall_mean_depth_value = build_tsdf(
        render_list=preop_renders,
        img_path_list=preop.img_paths,
        poses=preop.poses,
        intrinsics=intrinsics,
        mask=preop.mask,
    )
    del preop_renders
    verts, faces, norms, colors, _ = tsdf_vol.get_mesh(only_visited=True)
    tsdf.meshwrite(str(mesh_dir / "initial_mesh.ply"), verts, faces, -norms, colors)
    runtime.append(time.time() - preop_start)

    for bite, intra in enumerate(intraop):
        bite_start = time.time()

        # * compare warped preop CT to intraop poses with est depth
        if bite == 0:
            (scaled_intraop_depths, change_masks) = compare_warped_intraop(
                preop,
                intra,
                output_dir=change_dir,
                desc=f"bite{bite + 1}",
            )
        else:
            prev_update = data_utils.load_mesh(new_mesh_path)
            (scaled_intraop_depths, change_masks) = compare_warped_intraop(
                intraop[bite - 1],
                intra,
                last_seg=prev_update,
                output_dir=change_dir,
                desc=f"bite{bite + 1}",
            )

        # * feed into tsdf
        bite_mask_dir = mask_dir / f"bite_{bite + 1}"
        bite_mask_dir.mkdir(parents=True, exist_ok=True)

        print("Re-integrating TSDF...")
        for idx, (new_depth, new_mask) in tqdm(
            enumerate(zip(scaled_intraop_depths, change_masks)),
            total=len(scaled_intraop_depths),
        ):
            color_img = cv.imread(str(intra.img_paths[idx]))
            color_img = cv.cvtColor(color_img, cv.COLOR_BGR2RGB)
            color_img = downsample_utils.downsample_image(
                color_img,
                intra.start_h,
                intra.end_h,
                intra.start_w,
                intra.end_w,
            )

            tsdf_vol.integrate(
                color_img,
                new_depth,
                intra.intrinsics,
                intra.poses[idx],
                min_depth=1.0e-3 * overall_mean_depth_value,
                std_im=np.zeros_like(new_depth),
                obs_weight=1.0,
                mask=new_mask,
            )
            cv.imwrite(str(bite_mask_dir / f"mask_{idx}.png"), new_mask)
        verts, faces, norms, colors, _ = tsdf_vol.get_mesh(only_visited=True)
        new_mesh_path = mesh_dir / f"updated_mesh_bite{bite + 1}.ply"
        tsdf.meshwrite(
            str(new_mesh_path),
            verts,
            faces,
            -norms,
            colors,
        )
        print("Done")

        runtime.append(time.time() - bite_start)

    np.savetxt(str(output_dir / "runtime.txt"), runtime, fmt="%f")


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


def compare_warped_intraop(preop, intraop, output_dir, last_seg=None, desc=""):
    # * load estimated depth maps
    depth_map_paths = sorted((intraop.base_dir / "depths").glob("*.npy"))
    if last_seg:
        intraop_renders = intraop.generate_renders(seg=last_seg)
    else:
        intraop_renders = intraop.generate_renders(seg=preop.seg)

    new_depths = []
    change_masks = []
    preop_depths = []
    vid_frames = []
    # for visualization
    all_render_pts = []
    all_est_pts_reg = []
    for i, (intraop_pose, depth_p) in tqdm(
        enumerate(
            zip(
                intraop.poses,
                depth_map_paths,
            )
        ),
        total=len(intraop.poses),
    ):
        # * intraop info
        # estimated INTRAOP depth
        intraop_depth = np.load(depth_p, allow_pickle=True)
        intraop_depth = downsample_utils.downsample_image(
            intraop_depth,
            intraop.start_h,
            intraop.end_h,
            intraop.start_w,
            intraop.end_w,
        )

        mask = np.zeros_like(intraop_depth)
        mask[intraop_depth != 0] = 255

        _, preop_depth, _ = intraop_renders[i]
        preop_depth = image_utils.apply_mask(preop_depth, mask)
        preop_depths.append(preop_depth)

        # * project points to 3d to extract scale
        height, width = preop_depth.shape
        est_3d = []
        render_3d = []
        for u in range(height):
            for v in range(width):
                if mask[u, v] != 0:
                    est_pt = image_utils.project_2d_to_3d(
                        u, v, intraop_depth, intraop_pose, intraop.intrinsics
                    )
                    est_3d.append(est_pt)

                    render_pt = image_utils.project_2d_to_3d(
                        u, v, preop_depth, intraop_pose, intraop.intrinsics
                    )
                    render_3d.append(render_pt)

                    all_render_pts.append(render_pt)

        _, _, new_scale = register_utils.find_scale(
            source_pts=np.asarray(est_3d), target_pts=np.asarray(render_3d)
        )
        new_pts = new_scale * np.asarray(est_3d)

        # * register and transform to preop model
        transform = register_utils.register_3d(
            source_pts=new_pts,
            target_pts=np.asarray(render_3d),
            dist_thresh=5.0,
            confidence=0.6,
        )

        est_reg = np.array([transform @ np.hstack([pt, 1]) for pt in new_pts])
        est_reg = est_reg[:, :3]

        # * project back to intraop frame
        scaled_intraop_depth = copy.deepcopy(preop_depth)
        count = 0
        for pt_3d in est_reg:
            all_est_pts_reg.append(pt_3d)

            pt_2d, depth = image_utils.project_3d_to_2d(
                pt_3d, intraop_pose, intraop.intrinsics
            )

            if (
                (pt_2d[0] < 0)
                or (pt_2d[1] < 0)
                or (pt_2d[0] >= height)
                or (pt_2d[1] >= width)
            ):
                count += 1
            else:
                scaled_intraop_depth[pt_2d[0], pt_2d[1]] = depth
        scaled_intraop_depth = image_utils.apply_mask(
            scaled_intraop_depth, intraop.mask
        )
        new_depths.append(scaled_intraop_depth)

        # * computing mask with thresholding
        depth_diff = scaled_intraop_depth - preop_depth
        depth_diff[scaled_intraop_depth == 0] = 0
        depth_diff[preop_depth == 0] = 0

        change_mask = np.zeros_like(depth_diff)
        threshold = 1.0  # ! here
        change_mask[depth_diff > threshold] = 255
        change_masks.append(change_mask)

        # * visualization
        imgs = change_detection_frame(
            preop_depth, intraop_depth, scaled_intraop_depth, depth_diff, change_mask
        )
        header = build_header(row=imgs, count=i)
        frame = cv.vconcat([header, imgs])
        vid_frames.append(frame)

    image_utils.save_video(vid_frames, str(output_dir / f"detected_changes_{desc}.mp4"))

    return (np.asarray(new_depths), np.asarray(change_masks))


def change_detection_frame(
    preop_depth,
    warped_preop_depth,
    scaled_intraop_depth,
    depth_diff,
    change_mask,
    scale=20.0,
):
    preop_depth_img = render_utils.display_depth_map(preop_depth, scale=scale)
    warped_preop_depth = (warped_preop_depth + np.min(warped_preop_depth)) / (
        np.max(warped_preop_depth) - np.min(warped_preop_depth)
    )
    warped_preop_img = render_utils.display_depth_map(warped_preop_depth)
    warped_preop_img[warped_preop_depth == 0] = 0
    scaled_intraop_img = render_utils.display_depth_map(
        scaled_intraop_depth, scale=scale
    )
    depth_diff = render_utils.display_depth_map(depth_diff, scale=scale)
    depth_diff[change_mask != 0] = 255

    frame = cv.hconcat(
        [preop_depth_img, warped_preop_img, scaled_intraop_img, depth_diff]
    )

    return frame


def build_header(row, count):
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
    return header


if __name__ == "__main__":
    ct_update()
