import numpy as np


## based on https://arxiv.org/pdf/1406.2283.pdf
def scale_invariant_MSE(depth_map_1, depth_map_2):
    assert depth_map_1.shape == depth_map_2.shape
    di = np.log(depth_map_1) - np.log(depth_map_2)

    n = np.sum(~np.isnan(di)) + 1e-8  # number of pixels
    result = np.mean(di[~np.isnan(di)] ** 2) - (1 / (n**2)) * (
        np.sum(di[~np.isnan(di)]) ** 2
    )
    # if np.any(np.isnan(result)):
    #     breakpoint()

    return result, n


def axis_angle_err(r1, r2):
    rot = r1 @ np.transpose(r2)
    rot_err = np.arccos((np.trace(rot) - 1) / 2)

    return rot_err
