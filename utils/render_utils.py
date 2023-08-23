import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from . import pose_utils


def generate_renders(
    mesh,
    poses,
    intrinsics,
    img_width,
    img_height,
    save_dir=None,
    idx_list=None,
):
    """
    Generate color and depth renders of mesh based on camera poses

    Args:
        mesh: o3d triangle mesh
        poses: n x 4 x 4 numpy array
        intrinsics: 3 x 3 array of camera intrinsics from checkerboard calibration
        img_width: int
        img_height: int
        save_dir: Path

    Returns:
        depth_maps: np.array of depth values per pose
        depth_display: np.array of normalized depth map for visualization
        color_renders: image seen by camera
    """
    # set material
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"

    # set up scene
    renderer_o3d = o3d.visualization.rendering.OffscreenRenderer(img_width, img_height)
    renderer_o3d.scene.set_background(np.array([0, 0, 0, 0]))
    renderer_o3d.scene.add_geometry("mesh", mesh, mat)

    # set camera properties
    intrinsics_o3d = o3d.camera.PinholeCameraIntrinsic(
        img_width, img_height, intrinsics
    )
    inverted_poses = pose_utils.invert_poses(poses)[0:2]

    depth_maps = []
    depth_display = []
    color_render = []
    for idx, p in enumerate(tqdm(inverted_poses)):
        renderer_o3d.setup_camera(intrinsics_o3d, p)

        # add light based on position #! this does not do anything idk why
        # renderer_o3d.scene.scene.add_point_light(
        #     "light", [1, 1, 1], p[:3, 3], 1e6, 1e4, True
        # )

        # depth map with values
        depth_image = np.asarray(renderer_o3d.render_to_depth_image())
        depth_maps.append(depth_image)

        # depth image for visualization
        normalized_image = (depth_image - depth_image.min()) / (
            depth_image.max() - depth_image.min()
        )
        depth_display.append(normalized_image)

        # rendered color image
        color_image = np.asarray(renderer_o3d.render_to_image())
        color_render.append(color_image)

        # remove light before next render
        # renderer_o3d.scene.scene.remove_light("light")

        # save individual images (based on original seq idx values)
        if save_dir is not None:
            depth_dir = save_dir / "depth"
            depth_dir.mkdir(parents=True, exist_ok=True)

            color_dir = save_dir / "color"
            color_dir.mkdir(parents=True, exist_ok=True)

            if idx_list is None:
                depth_save = depth_dir / f"depth_{idx:06d}.png"
                color_save = color_dir / f"color_{idx:06d}.png"
            else:
                depth_save = depth_dir / f"depth_{idx_list[idx]:06d}.png"
                color_save = color_dir / f"color_{idx_list[idx]:06d}.png"

            plt.imsave(str(depth_save), normalized_image)
            plt.imsave(str(color_save), color_image)

    return np.asarray(color_render), np.asarray(depth_maps), np.asarray(depth_display)


def surface_mesh_global_scale(surface_mesh):
    max_bound = np.max(surface_mesh.vertices, axis=0)
    min_bound = np.min(surface_mesh.vertices, axis=0)

    return (
        np.linalg.norm(max_bound - min_bound, ord=2),
        np.linalg.norm(min_bound, ord=2),
        np.abs(max_bound[2] - min_bound[0]),
    )
