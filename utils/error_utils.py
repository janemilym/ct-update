import numpy as np


## based on https://arxiv.org/pdf/1406.2283.pdf
def scale_invariant_MSE(depth_map_1, depth_map_2):
    assert depth_map_1.shape == depth_map_2.shape
    di_1 = np.log(depth_map_1)
    di_1 = np.nan_to_num(di_1, nan=0, posinf=0, neginf=0)
    di_2 = np.log(depth_map_2)
    di_2 = np.nan_to_num(di_2, nan=0, posinf=0, neginf=0)

    di = di_1 - di_2
    # di = np.log(depth_map_1) - np.log(depth_map_2)

    n = np.sum(di[di != 0])
    result = np.mean(di[di != 0] ** 2) - (1 / (n**2)) * (np.sum(di[di != 0]) ** 2)

    # n = np.sum(~np.isnan(di)) + 1e-8  # number of pixels
    # result = np.mean(di[~np.isnan(di)] ** 2) - (1 / (n**2)) * (
    #     np.sum(di[~np.isnan(di)]) ** 2
    # )
    if np.any(np.isnan(result)):
        print("result is nan")
        breakpoint()

    return result, n


def axis_angle_err(r1, r2):
    rot = r1 @ np.transpose(r2)
    rot_err = np.arccos((np.trace(rot) - 1) / 2)

    return rot_err
