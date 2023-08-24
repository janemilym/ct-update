import numpy as np
import cv2 as cv
from tqdm import tqdm


def extract_keypoints(img_path_list, mask, output_dir=None, desc=None):
    ## eroding mask to eliminate kps at border
    new_mask = cv.erode(
        mask, cv.getStructuringElement(cv.MORPH_ELLIPSE, (60, 60)), iterations=3
    )

    kp_imgs = []
    keypoints = dict()
    print("Extracting SIFT keypoints...")
    for img_path in tqdm(img_path_list):
        img = cv.imread(str(img_path))
        sift = cv.SIFT_create()
        kps, des = sift.detectAndCompute(img, mask=new_mask)

        kp_img = cv.drawKeypoints(img, kps, None, color=(0, 255, 0))
        kp_imgs.append(kp_img)

        keypoints[img_path.stem] = [k.pt for k in kps]

    if output_dir is not None:
        save_keypoint_video(kp_imgs, output_dir, desc)

    return keypoints


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
