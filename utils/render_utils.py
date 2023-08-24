import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from tqdm import tqdm
from torch.utils.data import Dataset

from . import pose_utils
from . import image_utils

class MeshRender(Dataset):
    def __init__(
        self,
        mesh,
        poses,
        intrinsics,
        height,
        width,
        mask,
        near_plane=None,
        far_plane=None,
    ) -> None:
        self.poses = pose_utils.invert_poses(poses)
        self.n = len(poses)
        self.scale, _, _ = surface_mesh_global_scale(mesh)

        self.intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, intrinsics)
        self.mask = mask

        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"

        self.scene = o3d.visualization.rendering.OffscreenRenderer(width, height)
        self.scene.scene.set_background(np.array([0, 0, 0, 0]))
        self.scene.scene.add_geometry("mesh", mesh, mat)

        ## set camera FoV
        near_plane = 1e-3 if near_plane is None else near_plane
        far_plane = self.scale if far_plane is None else far_plane
        self.scene.scene.camera.set_projection(
            intrinsics, near_plane, far_plane, width, height
        )

    def __len__(self):
        return self.n

    def __getitem__(self, index) -> any:
        current_pose = self.poses[index]

        # add light based on position #! this does not do anything idk why
        # renderer_o3d.scene.scene.add_point_light(
        #     "light", [1, 1, 1], p[:3, 3], 1e6, 1e4, True
        # )
        self.scene.setup_camera(self.intrinsics, current_pose)

        depth_image = np.asarray(self.scene.render_to_depth_image(z_in_view_space=True))
        depth_image = np.nan_to_num(depth_image, posinf=0.0, neginf=0.0)
        depth_image = image_utils.apply_mask(depth_image, self.mask)

        normalized_image = display_depth_map(depth_image, scale=self.scale)
        normalized_image = image_utils.apply_mask(normalized_image, self.mask)

        color_image = np.asarray(self.scene.render_to_image())
        color_image = image_utils.apply_mask(color_image, self.mask)

        # remove light from scene to update in next render
        # renderer_o3d.scene.scene.remove_light("light")

        return color_image, depth_image, normalized_image


def generate_renders(
    mesh,
    poses,
    intrinsics,
    img_width,
    img_height,
    mask,
    save_dir=None,
    idx_list=None,
):
    mesh_render_list = MeshRender(mesh, poses, intrinsics, img_height, img_width, mask)

    depth_maps = []
    for idx in tqdm(range(len(mesh_render_list))):
        color_img, depth_map, depth_disp = mesh_render_list[idx]
        depth_maps.append(depth_map)

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

            plt.imsave(str(depth_save), depth_disp, mask)
            plt.imsave(str(color_save), color_img, mask)

    return mesh_render_list, depth_maps


def get_max_depth(mesh_render_list):
    max_depth = -np.inf
    for i in range(len(mesh_render_list)):
        _, depth_img, _ = mesh_render_list[i]

        if np.max(depth_img) > max_depth:
            max_depth = np.max(depth_img)

    return max_depth


def display_depth_map(
    depth_map, min_value=None, max_value=None, colormode=cv.COLORMAP_JET, scale=None
):
    if (min_value is None or max_value is None) and scale is None:
        if len(depth_map[depth_map > 0]) > 0:
            min_value = np.min(depth_map[depth_map > 0])
        else:
            min_value = 0.0

        if max_value is None:
            max_value = np.max(depth_map)
    elif scale is not None:
        min_value = 0.0
        max_value = scale
    else:
        pass

    depth_map_visualize = np.abs(
        (depth_map - min_value) / (max_value - min_value + 1.0e-8) * 255
    )
    depth_map_visualize[depth_map_visualize > 255] = 255
    depth_map_visualize[depth_map_visualize <= 0.0] = 0
    depth_map_visualize = cv.applyColorMap(np.uint8(depth_map_visualize), colormode)

    return depth_map_visualize


def save_render_video(img_list, mesh_render_list, output_dir, desc):
    assert len(img_list) == len(mesh_render_list)

    ## scale to visible max depth
    # max_depth = get_max_depth(mesh_render_list)
    # mesh_render_list.set_scale(max_depth)

    height, width, _ = mesh_render_list[0][0].shape

    output_vid = cv.VideoWriter(
        str(output_dir / f"{desc}_renders.mp4"),
        cv.VideoWriter_fourcc(*"mp4v"),
        15,
        (width * 3, height),
        True,
    )

    print(f"Writing video...")
    for idx in tqdm(range(len(img_list))):
        img = cv.imread(str(img_list[idx]))
        render, _, depth = mesh_render_list[idx]
        frame = np.concatenate([img, render, depth], axis=1)

        output_vid.write(frame)
    output_vid.release()

    print(f"Saved render video to: {output_dir}.")


def surface_mesh_global_scale(surface_mesh):
    max_bound = np.max(surface_mesh.vertices, axis=0)
    min_bound = np.min(surface_mesh.vertices, axis=0)

    return (
        np.linalg.norm(max_bound - min_bound, ord=2),
        np.linalg.norm(min_bound, ord=2),
        np.abs(max_bound[2] - min_bound[0]),
    )
