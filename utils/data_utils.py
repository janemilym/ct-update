import numpy as np

def load_intrinsics(intr_list):
    intrinsics = np.eye(3)
    intrinsics[0, 0] = intr_list[0]
    intrinsics[1, 1] = intr_list[1]
    intrinsics[0, 2] = intr_list[2]
    intrinsics[1, 2] = intr_list[3]

    return intrinsics