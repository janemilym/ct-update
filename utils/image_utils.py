import numpy as np
import cv2 as cv
from tqdm import tqdm
import open3d as o3d

from . import pose_utils
from . import register_utils


def apply_mask(img, mask):
    """
    Apply mask to image

    Args:
        mask: grayscale or binary
        image: color or grayscale
    """
    # convert to binary if grayscale
    if mask.max() > 0:
        mask[mask > 0] = 1

    if len(img.shape) == len(mask.shape):
        masked_img = img * mask
    elif len(img.shape) == 3:
        mask_rgb = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        masked_img = img * mask_rgb

    return masked_img


def canny_edge_detection(img):
    # Convert the frame to grayscale for edge detection
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and smoothen edges
    blurred = cv.GaussianBlur(src=gray, ksize=(5, 5), sigmaX=0)

    # Perform Canny edge detection
    edges = cv.Canny(blurred, 50, 125)

    return edges


def sobel_edge_detection(img):
    grad_x = cv.Sobel(img, cv.CV_64F, 1, 0)
    grad_y = cv.Sobel(img, cv.CV_64F, 0, 1)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    grad_norm = (grad * 255 / grad.max()).astype(np.uint8)

    return grad_norm


# def apply_optical_flow(prev_frame, flow):
#     height, width, _ = flow.shape
#     R2 = np.dstack(np.meshgrid(np.arange(width), np.arange(height)))

#     pixel_map = R2 - flow
#     pixel_map = pixel_map.astype("float32")
#     new_frame = cv.remap(
#         prev_frame, pixel_map[:, :, 0:1], pixel_map[:, :, 1:], cv.INTER_LINEAR
#     )

#     return new_frame


def apply_scene_flow(depth_map, orig_pose, new_pose, intrinsics, flow):
    height, width = depth_map.shape

    warped_depth = np.zeros_like(depth_map)
    pts = []
    pre_flow = []
    post_flow = []
    colors = []
    for u in range(height):
        for v in range(width):
            # extract 3d points in world frame
            z = depth_map[u, v]
            if z == 0:
                # skipping zero depths
                continue
            x = (v - intrinsics[0, 2]) * z / intrinsics[0, 0]
            y = (u - intrinsics[1, 2]) * z / intrinsics[1, 1]

            orig_pt_3d = orig_pose @ np.array([x, y, z, 1])
            new_pt_3d = orig_pose @ (flow[u, v] @ np.array([x, y, z, 1]))

            pre_flow.append(orig_pt_3d[:3])
            post_flow.append(new_pt_3d[:3])

            # project back to new camera
            new_pt = pose_utils.invert_pose(new_pose) @ new_pt_3d
            new_pt = new_pt[:2] / new_pt[2]

            new_v = new_pt[0] * intrinsics[0, 0] + intrinsics[0, 2]  # x -> v
            new_u = new_pt[1] * intrinsics[1, 1] + intrinsics[1, 2]  # y -> u

            new_v = np.round(new_v).astype(int)
            new_u = np.round(new_u).astype(int)

            if (new_v < 0) or (new_u < 0) or (new_u >= height) or (new_v >= width):
                continue
            else:
                warped_depth[new_u, new_v] = z
                pts.append((np.array([new_u, new_v]), new_pt_3d[:3]))

    # err_norm = np.linalg.norm(np.asarray(pre_flow) - np.asarray(post_flow), axis=1)
    # colors_pre = np.zeros((len(err_norm), 3))
    # colors_pre[err_norm > 15] = [0, 255, 0]
    # colors_post = np.ones((len(err_norm), 3)) * 128
    # colors_post[err_norm > 15] = [255, 0, 0]
    # save_3d_points(np.asarray(pre_flow), "test_3d_projected.ply", colors=colors)
    # save_3d_points(np.asarray(post_flow), "test_3d_flow.ply", colors=colors)

    # adjust_T = register_utils.register_3d(
    #     source_pts=np.asarray(post_flow), target_pts=np.asarray(pre_flow)
    # )
    # adjusted_pose = adjust_T @ new_pose
    # new_pts = np.array([adjust_T @ np.hstack([pt, 1]) for pt in post_flow])
    # new_pts = new_pts[:, :3]
    # save_3d_points(new_pts, "test_flow_transform.ply")

    return warped_depth


def project_2d_to_3d(u, v, depth_map, pose, intrinsics):
    z = depth_map[u, v]
    x = (v - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (u - intrinsics[1, 2]) * z / intrinsics[1, 1]

    pt_3d = pose @ np.array([x, y, z, 1])
    return pt_3d[:3]


def project_3d_to_2d(pt, pose, intrinsics):
    new_pt = pose_utils.invert_pose(pose) @ np.hstack([pt, 1])
    depth = new_pt[2]
    new_pt = new_pt[:2] / depth

    new_v = new_pt[0] * intrinsics[0, 0] + intrinsics[0, 2]  # x -> v
    new_u = new_pt[1] * intrinsics[1, 1] + intrinsics[1, 2]  # y -> u

    new_v = np.round(new_v).astype(int)
    new_u = np.round(new_u).astype(int)

    return [new_u, new_v], depth


def save_3d_points(point_array, save_path, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_array)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(str(save_path), pcd)


def save_keypoint_video(kp_imgs, output_dir, desc=None):
    height, width, _ = kp_imgs[0].shape

    desc = desc + "_" if desc is not None else ""
    output_vid = cv.VideoWriter(
        str(output_dir / f"{desc}sift_kps.mp4"),
        cv.VideoWriter_fourcc(*"mp4v"),
        15,
        (width, height),
        True,
    )

    print(f"Writing video...")
    for img in tqdm(kp_imgs):
        output_vid.write(img)
    output_vid.release()

    print(f"Saved keypoint video to: {output_dir}.")


def save_video(img_list, save_path="video.mp4"):
    height, width, _ = img_list[0].shape

    output_vid = cv.VideoWriter(
        str(save_path),
        cv.VideoWriter_fourcc(*"mp4v"),
        15,
        (width, height),
        True,
    )

    print(f"Writing video...")
    for img in tqdm(img_list):
        output_vid.write(img)
    output_vid.release()

    print(f"Saved video to: {save_path}")


def interpolate(img):
    h, w = img.shape
    imgResult = np.zeros((h, w), np.uint8)
    h_mask, w_mask = (1, 1)
    threshold = 0.0
    for j in range(h):
        for i in range(w):
            if img[j, i] == threshold:
                ymin = (j - h_mask) if (j - h_mask) >= 0 else 0
                ymax = (j + h_mask) if (j + h_mask) < h else (h - 1)
                xmin = (i - w_mask) if (j - w_mask) >= 0 else 0
                xmax = (i + w_mask) if (j + w_mask) < w else (w - 1)

                vals = img[ymin:ymax, xmin:xmax].flatten()
                imgResult[j, i] = np.mean(vals[vals != 0])
                # imgResult[j, i] = cv.mean(img[ymin:ymax, xmin:xmax])[0]
            else:
                imgResult[j, i] = img[j, i]

    return imgResult


# * unused functions
# def extract_keypoints_dict(img_path_list, mask=None, output_dir=None, desc=None):
#     kp_imgs = []
#     keypoints = dict()
#     descriptors = dict()
#     print("Extracting SIFT keypoints from sequence...")
#     for img_path in tqdm(img_path_list):
#         img, kps, des = extract_keypoints(img_path, mask=mask)
#         kp_img = cv.drawKeypoints(img, kps, None, color=(0, 255, 0))
#         kp_imgs.append(kp_img)

#         # keypoints[img_path.stem] = [k.pt for k in kps]
#         keypoints[img_path.stem] = kps
#         descriptors[img_path.stem] = des

#     if output_dir is not None:
#         save_keypoint_video(kp_imgs, output_dir, desc)

#     return keypoints, descriptors


# def extract_keypoints(img_path, mask=None):
#     if mask is not None:
#         ## eroding mask to eliminate kps at border
#         mask = cv.erode(
#             mask, cv.getStructuringElement(cv.MORPH_ELLIPSE, (60, 60)), iterations=3
#         )
#     img = cv.imread(str(img_path))
#     sift = cv.SIFT_create()
#     kps, des = sift.detectAndCompute(img, mask=mask)

#     return img, kps, des


# def match_keypoints(src_kps, src_des, dst_kps, dst_des):
#     bf = cv.BFMatcher()
#     matches = bf.match(src_des, dst_des)
#     matches = sorted(matches, key=lambda val: val.distance)

#     new_src_kps = []
#     new_dst_kps = []
#     for m in matches:
#         src_idx = m.queryIdx
#         dst_idx = m.trainIdx

#         new_src_kps.append(src_kps[src_idx].pt)
#         new_dst_kps.append(dst_kps[dst_idx].pt)
#     assert len(new_src_kps) == len(new_dst_kps)

#     return matches, np.asarray(new_src_kps), np.asarray(new_dst_kps)


# def find_best_matches(
#     src_img_paths, dst_img_paths, src_mask=None, dst_mask=None, output_dir=None
# ):
#     """
#     Finds indexes of source img paths with largest number of matches to each dest img path
#     """
#     src_kps, src_des = extract_keypoints_dict(src_img_paths, mask=src_mask)
#     dst_kps, dst_des = extract_keypoints_dict(dst_img_paths[0:5], mask=dst_mask)

#     print("Finding best matches...")
#     src_idxs = []
#     for dst_path in tqdm(dst_img_paths[0:5]):
#         matches = []
#         for src_path in src_img_paths:
#             n, _, _ = match_keypoints(
#                 src_kps[src_path.stem],
#                 src_des[src_path.stem],
#                 dst_kps[dst_path.stem],
#                 dst_des[dst_path.stem],
#             )
#             matches.append(n)
#         matches = np.asarray(matches)

#         src_idxs.append(np.argmax(matches))
#         breakpoint()
#     if output_dir is not None:
#         img_list = []

#         for idx, img_path in zip(src_idxs, dst_img_paths):
#             dst_img = cv.imread(str(img_path))
#             src_img = cv.imread(str(src_img_paths[idx]))

#             tmp_img = cv.hconcat([src_img, dst_img])
#             img_list.append(tmp_img)

#         save_video(img_list, save_path=str(output_dir / "matched_frames.mp4"))

#     return src_idxs


# def warp_image(src_img, dst_img, src_kps, dst_kps, dst_mask, threshold=10.0):
#     # src_img = cv.imread(str(src_img_path))
#     # dst_img = cv.imread(str(dst_img_path))

#     height, width, _ = dst_img.shape
#     h, _ = cv.findHomography(src_kps, dst_kps, cv.RANSAC, threshold)
#     warped = cv.warpPerspective(src_img, h, (width, height))
#     warped = apply_mask(warped, dst_mask)

#     return warped
