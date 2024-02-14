import pyvista as pv
import matplotlib.pyplot as plt
import numpy as np


def plot_correspondences(point_num):
    times = np.fromiter(range(point_num), dtype=float)
    norm = plt.Normalize()
    colors = plt.cm.hsv(norm(times))[np.newaxis, :, :3]
    u_col = np.round(colors[0] * 255)
    u_col = u_col.astype(np.uint8)

    return u_col


def save_point_array(point_array, save_name, colors=None):
    pset = pv.PointSet(point_array)
    pdata = pset.cast_to_polydata()

    pdata.save(save_name, texture=colors)
