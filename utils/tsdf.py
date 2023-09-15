import cv2
import numpy as np
from skimage import measure

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule

except Exception as err:
    print("Warning: %s" % (str(err)))
    print("Failed to import PyCUDA.")
    exit()


# Get corners of 3D camera view frustum of depth image
def get_view_frustum(depth_im, cam_intr, cam_pose):
    im_h = depth_im.shape[0]
    im_w = depth_im.shape[1]
    max_depth = np.max(depth_im)
    view_frust_pts = np.array(
        [
            (np.array([0, 0, 0, im_w, im_w]) - cam_intr[0, 2])
            * np.array([0, max_depth, max_depth, max_depth, max_depth])
            / cam_intr[0, 0],
            (np.array([0, 0, im_h, 0, im_h]) - cam_intr[1, 2])
            * np.array([0, max_depth, max_depth, max_depth, max_depth])
            / cam_intr[1, 1],
            np.array([0, max_depth, max_depth, max_depth, max_depth]),
        ]
    )
    view_frust_pts = np.dot(cam_pose[:3, :3], view_frust_pts) + np.tile(
        cam_pose[:3, 3].reshape(3, 1), (1, view_frust_pts.shape[1])
    )  # from camera to world coordinates
    return view_frust_pts


# Save 3D mesh to a polygon .ply file
def meshwrite(filename, verts, faces, norms, colors):
    # Write header
    ply_file = open(filename, "w")
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property float nx\n")
    ply_file.write("property float ny\n")
    ply_file.write("property float nz\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("element face %d\n" % (faces.shape[0]))
    ply_file.write("property list uchar int vertex_index\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        ply_file.write(
            "%f %f %f %f %f %f %d %d %d\n"
            % (
                verts[i, 0],
                verts[i, 1],
                verts[i, 2],
                norms[i, 0],
                norms[i, 1],
                norms[i, 2],
                colors[i, 0],
                colors[i, 1],
                colors[i, 2],
            )
        )

    # Write face list
    for i in range(faces.shape[0]):
        ply_file.write("3 %d %d %d\n" % (faces[i, 0], faces[i, 1], faces[i, 2]))

    ply_file.close()


class TSDFVolume(object):
    def __init__(self, vol_bnds, voxel_size, trunc_margin):
        # Define voxel volume parameters
        self._vol_bnds = (
            vol_bnds  # rows: x,y,z columns: min,max in world coordinates in meters
        )
        self._voxel_size = voxel_size
        self._trunc_margin = trunc_margin

        # Adjust volume bounds
        self._vol_dim = (
            np.ceil((self._vol_bnds[:, 1] - self._vol_bnds[:, 0]) / self._voxel_size)
            .copy(order="C")
            .astype(int)
        )  # ensure C-order contiguous
        self._vol_bnds[:, 1] = self._vol_bnds[:, 0] + self._vol_dim * self._voxel_size
        self._vol_origin = (
            self._vol_bnds[:, 0].copy(order="C").astype(np.float32)
        )  # ensure C-order contiguous
        print(
            "Voxel volume size: %d x %d x %d"
            % (self._vol_dim[0], self._vol_dim[1], self._vol_dim[2])
        )

        # Initialize pointers to voxel volume in CPU memory
        # Assign oversized tsdf volume
        self._tsdf_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)  # -2.0 *
        self._weight_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
        self._weight_vol_color_cpu = np.zeros(self._vol_dim).astype(np.float32)
        self._uncertainty_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
        self._color_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)

        # Copy voxel volumes to GPU
        self._tsdf_vol_gpu = cuda.mem_alloc(self._tsdf_vol_cpu.nbytes)
        cuda.memcpy_htod(self._tsdf_vol_gpu, self._tsdf_vol_cpu)
        self._weight_vol_gpu = cuda.mem_alloc(self._weight_vol_cpu.nbytes)
        cuda.memcpy_htod(self._weight_vol_gpu, self._weight_vol_cpu)

        self._weight_vol_color_gpu = cuda.mem_alloc(self._weight_vol_color_cpu.nbytes)
        cuda.memcpy_htod(self._weight_vol_color_gpu, self._weight_vol_color_cpu)

        self._uncertainty_vol_gpu = cuda.mem_alloc(self._uncertainty_vol_cpu.nbytes)
        cuda.memcpy_htod(self._uncertainty_vol_gpu, self._uncertainty_vol_cpu)
        self._color_vol_gpu = cuda.mem_alloc(self._color_vol_cpu.nbytes)
        cuda.memcpy_htod(self._color_vol_gpu, self._color_vol_cpu)

        # Original cuda kernel function (C++)
        self._cuda_src_mod_with_confidence_map = SourceModule(
            """
          __global__ void integrate(float * tsdf_vol,
                                    float * weight_vol,
                                    float * uncertainty_vol,
                                    float * color_vol,
                                    float * vol_dim,
                                    float * vol_origin,
                                    float * cam_intr,
                                    float * cam_pose,
                                    float * other_params,
                                    float * color_im,
                                    float * depth_im,
                                    float * std_im,
                                    float * mask) {
            // Get voxel index
            int gpu_loop_idx = (int) other_params[0];
            int max_threads_per_block = blockDim.x;
            int block_idx = blockIdx.z*gridDim.y*gridDim.x+blockIdx.y*gridDim.x+blockIdx.x;
            int voxel_idx = gpu_loop_idx*gridDim.x*gridDim.y*gridDim.z*max_threads_per_block+block_idx*max_threads_per_block+threadIdx.x;
            int vol_dim_x = (int) vol_dim[0];
            int vol_dim_y = (int) vol_dim[1];
            int vol_dim_z = (int) vol_dim[2];
            if (voxel_idx > vol_dim_x*vol_dim_y*vol_dim_z)
                return;
            // Get voxel grid coordinates (note: be careful when casting)
            float voxel_x = floorf(((float)voxel_idx)/((float)(vol_dim_y*vol_dim_z)));
            float voxel_y = floorf(((float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z))/((float)vol_dim_z));
            float voxel_z = (float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z-((int)voxel_y)*vol_dim_z);
            // Voxel grid coordinates to world coordinates
            float voxel_size = other_params[1];
            float pt_x = vol_origin[0]+voxel_x*voxel_size;
            float pt_y = vol_origin[1]+voxel_y*voxel_size;
            float pt_z = vol_origin[2]+voxel_z*voxel_size;
            // World coordinates to camera coordinates
            float tmp_pt_x = pt_x-cam_pose[0*4+3];
            float tmp_pt_y = pt_y-cam_pose[1*4+3];
            float tmp_pt_z = pt_z-cam_pose[2*4+3];
            float cam_pt_x = cam_pose[0*4+0]*tmp_pt_x+cam_pose[1*4+0]*tmp_pt_y+cam_pose[2*4+0]*tmp_pt_z;
            float cam_pt_y = cam_pose[0*4+1]*tmp_pt_x+cam_pose[1*4+1]*tmp_pt_y+cam_pose[2*4+1]*tmp_pt_z;
            float cam_pt_z = cam_pose[0*4+2]*tmp_pt_x+cam_pose[1*4+2]*tmp_pt_y+cam_pose[2*4+2]*tmp_pt_z;
            // Because of the long tube of endoscope, the minimum depth to consider is not zero
            float min_depth = other_params[6];
            if (cam_pt_z < min_depth) {
                return;
            }
            // Camera coordinates to image pixels
            int pixel_x = (int) roundf(cam_intr[0*3+0]*(cam_pt_x/cam_pt_z)+cam_intr[0*3+2]);
            int pixel_y = (int) roundf(cam_intr[1*3+1]*(cam_pt_y/cam_pt_z)+cam_intr[1*3+2]);
            // Skip if outside view frustum
            int im_h = (int) other_params[2];
            int im_w = (int) other_params[3];
            if (pixel_x < 0 || pixel_x >= im_w || pixel_y < 0 || pixel_y >= im_h)
                return;
            // Skip invalid depth
            float depth_value = depth_im[pixel_y*im_w+pixel_x];
            float std_value = std_im[pixel_y*im_w + pixel_x];
            // This was modified to assume all values are certain (using CT ground-truth)
            //if (depth_value <= 0 || std_value <= 0) {
            //    return;
            //}
            if (depth_value <= 0) {
                return;
            }
            // Get std value for the current observation
            float trunc_margin = other_params[4];
            float depth_diff = depth_value - cam_pt_z;
            if (depth_diff < -trunc_margin)
                return;
            //float dist = fmin(1.0f, depth_diff / std_value);
            float dist = fmin(1.0f,depth_diff/trunc_margin);
            float w_old = weight_vol[voxel_idx];
            float obs_weight = other_params[5];
            // TESTING HERE: USING MASK
            float invalid = mask[pixel_y * im_w + pixel_x];
            if (invalid > 0) {
                //tsdf_vol[voxel_idx] = 0;
                //weight_vol[voxel_idx] = 0;
                w_old = 0;
                dist = 0;
                //obs_weight = 0.000001;
            }
            float w_new = w_old + obs_weight;
            tsdf_vol[voxel_idx] = (tsdf_vol[voxel_idx] * w_old + dist * obs_weight) / w_new;
            weight_vol[voxel_idx] = w_new;
            // Integrate color
            if (invalid > 0) {
                color_vol[voxel_idx] = 0;
                return;
            }
            float new_color = color_im[pixel_y * im_w + pixel_x];
            float new_b = floorf(new_color / (256 * 256));
            float new_g = floorf((new_color - new_b * 256 * 256) / 256);
            float new_r = new_color - new_b * 256 * 256 - new_g * 256;
            float old_color = color_vol[voxel_idx];
            float old_b = floorf(old_color / (256 * 256));
            float old_g = floorf((old_color - old_b * 256 * 256) / 256);
            float old_r = old_color - old_b * 256 * 256 - old_g * 256;
            new_b = fmin(roundf((old_b * w_old + new_b * obs_weight) / w_new), 255.0f);
            new_g = fmin(roundf((old_g * w_old + new_g * obs_weight) / w_new), 255.0f);
            new_r = fmin(roundf((old_r * w_old + new_r * obs_weight) / w_new), 255.0f);
            color_vol[voxel_idx] = new_b * 256 * 256 + new_g * 256 + new_r;
          }"""
        )

        self._cuda_integrate = self._cuda_src_mod_with_confidence_map.get_function(
            "integrate"
        )

        # Determine block/grid size on GPU
        gpu_dev = cuda.Device(0)
        self._max_gpu_threads_per_block = gpu_dev.MAX_THREADS_PER_BLOCK
        n_blocks = int(
            np.ceil(
                float(np.prod(self._vol_dim)) / float(self._max_gpu_threads_per_block)
            )
        )
        grid_dim_x = min(gpu_dev.MAX_GRID_DIM_X, int(np.floor(np.cbrt(n_blocks))))
        grid_dim_y = min(
            gpu_dev.MAX_GRID_DIM_Y, int(np.floor(np.sqrt(n_blocks / grid_dim_x)))
        )
        grid_dim_z = min(
            gpu_dev.MAX_GRID_DIM_Z,
            int(np.ceil(float(n_blocks) / float(grid_dim_x * grid_dim_y))),
        )
        self._max_gpu_grid_dim = np.array([grid_dim_x, grid_dim_y, grid_dim_z]).astype(
            int
        )
        # _n_gpu_loops specifies how many loops for the GPU to process the entire volume
        self._n_gpu_loops = int(
            np.ceil(
                float(np.prod(self._vol_dim))
                / float(
                    np.prod(self._max_gpu_grid_dim) * self._max_gpu_threads_per_block
                )
            )
        )

    def integrate(
        self,
        color_im,
        depth_im,
        cam_intr,
        cam_pose,
        min_depth,
        std_im,
        obs_weight=1.0,
        mask=None,
    ):
        im_h = depth_im.shape[0]
        im_w = depth_im.shape[1]

        # Fold RGB color image into a single channel image
        color_im = color_im.astype(np.float32)
        color_im = np.floor(
            color_im[:, :, 2] * 256 * 256 + color_im[:, :, 1] * 256 + color_im[:, :, 0]
        )

        if mask is None:
            mask = np.zeros_like(depth_im)

        # integrate voxel volume (calls CUDA kernel)
        for gpu_loop_idx in range(self._n_gpu_loops):
            self._cuda_integrate(
                self._tsdf_vol_gpu,
                self._weight_vol_gpu,
                self._uncertainty_vol_gpu,
                self._color_vol_gpu,
                cuda.InOut(self._vol_dim.astype(np.float32)),
                cuda.InOut(self._vol_origin.astype(np.float32)),
                cuda.InOut(cam_intr.reshape(-1).astype(np.float32)),
                cuda.InOut(cam_pose.reshape(-1).astype(np.float32)),
                cuda.InOut(
                    np.asarray(
                        [
                            gpu_loop_idx,
                            self._voxel_size,
                            im_h,
                            im_w,
                            self._trunc_margin,
                            obs_weight,
                            min_depth,
                        ],
                        np.float32,
                    )
                ),
                cuda.InOut(color_im.reshape(-1).astype(np.float32)),
                cuda.InOut(depth_im.reshape(-1).astype(np.float32)),
                cuda.InOut(std_im.reshape(-1).astype(np.float32)),
                cuda.InOut(mask.reshape(-1).astype(np.float32)),
                block=(self._max_gpu_threads_per_block, 1, 1),
                grid=(
                    int(self._max_gpu_grid_dim[0]),
                    int(self._max_gpu_grid_dim[1]),
                    int(self._max_gpu_grid_dim[2]),
                ),
            )

    # Copy voxel volume to CPU
    def get_volume(self):
        cuda.memcpy_dtoh(self._tsdf_vol_cpu, self._tsdf_vol_gpu)
        cuda.memcpy_dtoh(self._color_vol_cpu, self._color_vol_gpu)
        cuda.memcpy_dtoh(self._weight_vol_cpu, self._weight_vol_gpu)
        return self._tsdf_vol_cpu, self._color_vol_cpu, self._weight_vol_cpu

    # Get mesh of voxel volume via marching cubes
    def get_mesh(self, only_visited=False):
        tsdf_vol, color_vol, weight_vol = self.get_volume()

        # verts, faces, norms, vals = measure.marching_cubes_lewiner(tsdf_vol, level=0, gradient_direction='ascent')
        verts, faces, norms, vals = measure.marching_cubes(
            volume=tsdf_vol, level=0, gradient_direction="ascent", method="lewiner"
        )

        verts_ind = np.round(verts).astype(int)
        verts = (
            verts * self._voxel_size + self._vol_origin
        )  # voxel grid coordinates to world coordinates

        # Get vertex colors
        std_vals = weight_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        std_vals = np.uint8(std_vals / np.max(std_vals) * 255)
        std_colors = std_vals.astype(np.uint8).reshape(-1, 1)
        std_colors = cv2.cvtColor(
            cv2.applyColorMap(std_colors, cv2.COLORMAP_HOT), cv2.COLOR_BGR2RGB
        ).reshape(-1, 3)

        rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors_b = np.floor(rgb_vals / (256 * 256))
        colors_g = np.floor((rgb_vals - colors_b * 256 * 256) / 256)
        colors_r = rgb_vals - colors_b * 256 * 256 - colors_g * 256
        colors = np.transpose(
            np.uint8(np.floor(np.asarray([colors_r, colors_g, colors_b])))
        ).reshape(-1, 3)

        # valid_face_indexes = []
        # invalid_face_indexes = []
        if only_visited:
            verts_indxes = (
                verts_ind[:, 0] * weight_vol.shape[1] * weight_vol.shape[2]
                + verts_ind[:, 1] * weight_vol.shape[2]
                + verts_ind[:, 2]
            )
            weight_vol = weight_vol.reshape((-1))
            valid_vert_indexes = np.nonzero(weight_vol[verts_indxes] >= 1)[0]
            valid_vert_indexes = set(valid_vert_indexes)
            # set_0 = set(faces[:, 0])
            # set_1 = set(faces[:, 1])
            # set_2 = set(faces[:, 2])

            indicators = (
                np.array([face in valid_vert_indexes for face in faces[:, 0]])
                & np.array([face in valid_vert_indexes for face in faces[:, 1]])
                & np.array([face in valid_vert_indexes for face in faces[:, 2]])
            )
            # for idx, face in enumerate(faces):
            #     if face[0] not in valid_vert_indexes or face[1] not in valid_vert_indexes or face[
            #         2] not in valid_vert_indexes:
            #         invalid_face_indexes.append(idx)
            #     else:
            #         valid_face_indexes.append(idx)

            return verts, faces[indicators], norms, colors, std_colors

        return verts, faces, norms, colors, std_colors
