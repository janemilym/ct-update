import numpy as np
import cv2 as cv


def downsample_and_crop_mask(
    mask, downsampling_factor=4.0, divide=64, suggested_h=None, suggested_w=None
):
    downsampled_mask = cv.resize(
        mask, (0, 0), fx=1.0 / downsampling_factor, fy=1.0 / downsampling_factor
    )
    end_h_index = downsampled_mask.shape[0]
    end_w_index = downsampled_mask.shape[1]
    # divide is related to the pooling times of the teacher model
    if np.max(downsampled_mask > 1):
        indexes = np.where(downsampled_mask >= 200)
    elif np.max(downsampled_mask == 1):
        indexes = np.where(downsampled_mask == 1)
    else:
        print("mask is broken")
        exit(0)
    h = indexes[0].max() - indexes[0].min()
    w = indexes[1].max() - indexes[1].min()

    remainder_h = h % divide
    remainder_w = w % divide

    increment_h = divide - remainder_h
    increment_w = divide - remainder_w

    target_h = h + increment_h
    target_w = w + increment_w

    start_h = max(indexes[0].min() - increment_h // 2, 0)
    end_h = start_h + target_h

    start_w = max(indexes[1].min() - increment_w // 2, 0)
    end_w = start_w + target_w

    if suggested_h is not None:
        if suggested_h != h:
            remain_h = suggested_h - target_h
            start_h = max(start_h - remain_h // 2, 0)
            end_h = min(suggested_h + start_h, end_h_index)
            start_h = end_h - suggested_h

    if suggested_w is not None:
        if suggested_w != w:
            remain_w = suggested_w - target_w
            start_w = max(start_w - remain_w // 2, 0)
            end_w = min(suggested_w + start_w, end_w_index)
            start_w = end_w - suggested_w

    kernel = np.ones((5, 5), np.uint8)
    downsampled_mask_erode = cv.erode(downsampled_mask, kernel, iterations=1)
    cropped_mask = downsampled_mask_erode[start_h:end_h, start_w:end_w]

    return cropped_mask, start_h, end_h, start_w, end_w


def downsample_image(img, start_h, end_h, start_w, end_w, downsampling_factor=4.0):
    # img = cv.imread(str(img_path))
    downsampled_img = cv.resize(
        img, (0, 0), fx=1.0 / downsampling_factor, fy=1.0 / downsampling_factor
    )
    if len(downsampled_img.shape) == 2:
        downsampled_img = downsampled_img[start_h:end_h, start_w:end_w]
    else:
        downsampled_img = downsampled_img[start_h:end_h, start_w:end_w, :]
        # downsampled_img = cv.cvtColor(downsampled_img, cv.COLOR_BGR2RGB)

    return downsampled_img
