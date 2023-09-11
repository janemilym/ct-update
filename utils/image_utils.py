import numpy as np
import cv2 as cv
from tqdm import tqdm


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


def apply_flow(prev_frame, flow):
    height, width, _ = flow.shape
    R2 = np.dstack(np.meshgrid(np.arange(width), np.arange(height)))

    pixel_map = R2 - flow
    pixel_map = pixel_map.astype("float32")
    new_frame = cv.remap(
        prev_frame, pixel_map[:, :, 0:1], pixel_map[:, :, 1:], cv.INTER_LINEAR
    )

    return new_frame


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
