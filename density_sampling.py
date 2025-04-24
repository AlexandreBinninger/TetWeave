import torch
import kaolin as kal
from tetweave import neighbor_active_points, get_active_points
import numpy as np
from util import *
import tqdm
import render
from blue_noise import BlueNoise

class Grid():
    def __init__(self, resolution, bottom, size, device="cuda"):
        self.device = device
        if isinstance(resolution, int):
            resolution = (resolution, resolution, resolution)
        if isinstance(bottom, float):
            bottom = torch.tensor([bottom, bottom, bottom], device=device)
        if isinstance(size, float):
            size = torch.tensor([size, size, size], device=device)
            
        self.resolution = resolution
        self.bottom = bottom.to(device)
        self.size = size.to(device)
        self.density = torch.zeros(resolution, device=device)
        self.device = device
        self.bn=BlueNoise()
    
    def get_frequencies(self):
        frequencies = self.density.clone()
        frequencies[frequencies <=0] = 0
        total = frequencies.sum()
        frequencies /= total
        return frequencies
    
    def get_voxel_center(self, idx):
        voxel_size = self.size / torch.tensor(self.resolution, device=self.device)
        return self.bottom.unsqueeze(0) + voxel_size.unsqueeze(0) * (idx + 0.5)
    
    def get_voxel_centers(self):
        x, y, z = torch.meshgrid(torch.arange(self.resolution[0]), torch.arange(self.resolution[1]), torch.arange(self.resolution[2]), indexing='ij')
        idx = torch.stack([x, y, z], dim=0).reshape(3, -1).permute(1, 0).to("cuda")
        return self.get_voxel_center(idx)

    def flatten_idx_to_3d(self, idx):
        z = idx % self.resolution[2]
        y = (idx // self.resolution[2]) % self.resolution[1]
        x = idx // (self.resolution[2] * self.resolution[1])
        return torch.stack([x, y, z], dim=-1)
    
    def three_d_to_flatten_idx(self, idx):
        return idx[:, 0] * self.resolution[1] * self.resolution[2] + idx[:, 1] * self.resolution[2] + idx[:, 2]

    def sample_points(self, num_points):
        frequencies = self.get_frequencies()
        sampled_idx = torch.multinomial(frequencies.view(-1), num_points, replacement=True)
        idx_resampled_points, resampled_counts = torch.unique(sampled_idx, return_counts=True)
        
        points = []
        idx_resampled_points_3d = self.flatten_idx_to_3d(idx_resampled_points)
        voxel_centers = self.get_voxel_center(idx_resampled_points_3d)
        for i, _ in enumerate(idx_resampled_points_3d):
            nb_pts = resampled_counts[i].item()
            noise = self.bn.aggregate_scaled_targeted_noise(nb_points=nb_pts)
            center = voxel_centers[i]
            pt = center + noise * self.size / torch.tensor(self.resolution, device=self.device)
            points.append(pt)
        
        points = torch.cat(points, dim=0)
        random_epsilon = torch.rand(points.shape[0], 3, device=self.device) * 1e-6
        points += random_epsilon
        return points
    
    def set_functional_density(self, f):
        all_points = self.get_voxel_centers()
        all_points = all_points.reshape(self.resolution[0], self.resolution[1], self.resolution[2], 3)
        self.density = f(all_points)
    
    def point_voxel_idx(self, points):
        voxel_size = self.size / torch.tensor(self.resolution, device=self.device)
        idx = ((points - self.bottom) / voxel_size).floor().long()
        return idx
    
    def filter_idx(self, idx):
        idx = idx[idx.min(dim=1)[0] >= 0]
        idx = idx[idx[:, 0] < self.resolution[0]]
        idx = idx[idx[:, 1] < self.resolution[1]]
        idx = idx[idx[:, 2] < self.resolution[2]]
        return idx
    
    def containing_voxels(self, points):
        """
        Compute the voxel indices that contain the given points.
        Args:
            points (torch.Tensor): Tensor of shape (N, 3) representing 3D points.
        Returns:
            torch.Tensor: Tensor of shape (M, 3) containing the 3D voxel indices that contain the points.
        """
        voxel_size = self.size / torch.tensor(self.resolution, device=self.device)
        idx = ((points - self.bottom) / voxel_size).floor().long()
        idx_flatten = self.three_d_to_flatten_idx(idx)
        idx_flatten = torch.unique(idx_flatten)
        idx = self.flatten_idx_to_3d(idx_flatten)
        idx = self.filter_idx(idx)
        return idx
    
    def mask_density(self, idx):
        flattened_density = self.density.view(-1)
        resolution = torch.tensor(self.density.shape, device=self.device)
        linear_idx = idx[:, 0] * resolution[1] * resolution[2] + idx[:, 1] * resolution[2] + idx[:, 2]
        new_flattened_density = torch.zeros_like(flattened_density)
        new_flattened_density[linear_idx] = flattened_density[linear_idx]
        self.density = new_flattened_density.view(self.density.shape)

        
def x_abs(points):
    return torch.abs(points[:, :, :, 0])

def x_plus_one(points):
    return (points[:, :, :, 0] + 1)**2

def x_constant(points):
    return torch.ones_like(points[:, :, :, 0])

def x_squared(points):
    return (points[:, :, :, 0])**2

def y_squared(points):
    p = torch.abs((points[:, :, :, 1] - 1.0))**3 + 0.05
    return p
    
def z_squared(points):
    return points[:, :, :, 2]**2

def radius_squared(points, r=0.0):
    return (points[:, :, :, 0]**2 + points[:, :, :, 1]**2 + points[:, :, :, 2]**2) - r**2    


@torch.no_grad()
def points_in_tetrahedra_chunked(points, vertices, tetrahedra, chunk_size=10):
    """
    Check if points are inside a set of tetrahedra using barycentric coordinates, 
    processing in chunks for large datasets.
    
    Args:
    - points: Tensor of shape (P, 3), where P is the number of points.
    - vertices: Tensor of shape (V, 3), where V is the number of vertices.
    - tetrahedra: Tensor of shape (T, 4), where T is the number of tetrahedra.
    - chunk_size: Number of points to process in each chunk.
    
    Returns:
    - inside_tet: Tensor of shape (P,) with the index of the first tetrahedron that contains each point, or -1 if no tetrahedron contains the point.
    - barycentric_coords: Tensor of shape (P, 4) with the barycentric coordinates of each point in the first tetrahedron it is inside, or zeros if none found.
    """
    num_points = points.shape[0]
    inside_tet = torch.full((num_points,), -1, device=points.device, dtype=torch.long)
    barycentric_coords = torch.zeros((num_points, 4), device=points.device)

    pbar = tqdm.tqdm(total=num_points//chunk_size, desc="Processing points in tetrahedra")
    for start in range(0, num_points, chunk_size):
        end = min(start + chunk_size, num_points)
        chunk_points = points[start:end]

        tets_vertices = vertices[tetrahedra]
        v0, v1, v2, v3 = tets_vertices[:, 0], tets_vertices[:, 1], tets_vertices[:, 2], tets_vertices[:, 3]

        a = v0 - v3
        b = v1 - v3
        c = v2 - v3
        d = chunk_points.unsqueeze(1) - v3.unsqueeze(0)

        mat = torch.stack([a, b, c], dim=-1)
        chunk_bary_coords = torch.linalg.solve(mat, d.permute(1, 2, 0)).permute(2, 0, 1)
        lambda_0, lambda_1, lambda_2 = chunk_bary_coords.unbind(dim=-1)
        lambda_3 = 1 - lambda_0 - lambda_1 - lambda_2
        inside = (lambda_0 >= 0) & (lambda_1 >= 0) & (lambda_2 >= 0) & (lambda_3 >= 0)

        valid_tet_exists = inside.any(dim=1)
        chunk_inside_tet = torch.where(valid_tet_exists, inside.float().argmax(dim=1), torch.tensor(-1, device=points.device))

        chunk_valid_mask = valid_tet_exists.nonzero(as_tuple=True)[0]
        if chunk_valid_mask.numel() > 0:
            valid_tet_idx = chunk_inside_tet[chunk_valid_mask]
            barycentric_coords[start:end, 0][chunk_valid_mask] = lambda_0[chunk_valid_mask, valid_tet_idx]
            barycentric_coords[start:end, 1][chunk_valid_mask] = lambda_1[chunk_valid_mask, valid_tet_idx]
            barycentric_coords[start:end, 2][chunk_valid_mask] = lambda_2[chunk_valid_mask, valid_tet_idx]
            barycentric_coords[start:end, 3][chunk_valid_mask] = lambda_3[chunk_valid_mask, valid_tet_idx]

        inside_tet[start:end] = chunk_inside_tet
        pbar.update(1)

    return inside_tet, barycentric_coords

def resample_point_method(sdf, tets, x_nx3, method="partial"):
    if method == "partial":
        neighbor_active_points_ids = neighbor_active_points(sdf, tets)
        neighbor_active_points_mask = torch.zeros_like(sdf, dtype=torch.bool)
        neighbor_active_points_mask[neighbor_active_points_ids] = True
        nb_resampled_points = (~neighbor_active_points_mask).sum()
    elif method == "percentage":
        percentage = 0.9
        active_points_ids = get_active_points(sdf, tets)
        neighbor_active_points_ids = neighbor_active_points(sdf, tets)
        neighbor_active_points_mask = torch.zeros_like(sdf, dtype=torch.bool)
        neighbor_active_points_mask[active_points_ids] = True
        true_indices = torch.where(neighbor_active_points_mask)[0]
        neighbor_active_points_mask[neighbor_active_points_ids] = True
        shuffled_indices = true_indices[torch.randperm(true_indices.size(0))]
        num_to_flip = int(len(true_indices) * percentage)
        indices_to_flip = shuffled_indices[:num_to_flip]
        neighbor_active_points_mask[indices_to_flip] = False
        nb_resampled_points = (~neighbor_active_points_mask).sum()
    elif method == "full":
        nb_resampled_points = x_nx3.shape[0]
        neighbor_active_points_mask = torch.zeros_like(sdf, dtype=torch.bool)
        
    return nb_resampled_points, neighbor_active_points_mask

def aggregation(updated_x_nx3, neighbor_active_points_mask, x_nx3, sdf, tets, features = None, method="interpolation"):
    if method == "interpolation":
        updated_sdf = sdf[~neighbor_active_points_mask].clone()
        inside_tets, barycentric_coords = points_in_tetrahedra_chunked(updated_x_nx3, x_nx3, tets)
        mask_inside_tets = inside_tets != -1
        updated_sdf[mask_inside_tets] = torch.einsum("ij,ij->i", barycentric_coords[mask_inside_tets], sdf[tets[inside_tets[mask_inside_tets]]])
        sdf[~neighbor_active_points_mask] = updated_sdf
        if features is not None:
            updated_features = features[~neighbor_active_points_mask].clone()
            updated_features[mask_inside_tets] = torch.einsum("ij,ijk->ik", barycentric_coords[mask_inside_tets], features[tets[inside_tets[mask_inside_tets]]])
            features[~neighbor_active_points_mask] = updated_features
    
    elif method == "zero":
        sdf[~neighbor_active_points_mask] = 0
        if features is not None:
            features[~neighbor_active_points_mask] = 0
    x_nx3[~neighbor_active_points_mask] = updated_x_nx3
    
    return x_nx3, sdf, features

def sample_block_functional_density(x_nx3, sdf, tets, kal_mesh, func = y_squared, features = None, aggregation_method="interpolation", resampled_points_method="partial"):
    nb_resampled_points, neighbor_active_points_mask = resample_point_method(sdf, tets, x_nx3, method=resampled_points_method)
    if nb_resampled_points == 0:
        return x_nx3, sdf, features
    grid = Grid(32, -1.0, 2.0)
    grid.set_functional_density(func)
    containing_voxels_mesh = grid.containing_voxels(kal_mesh.vertices)
    grid.mask_density(containing_voxels_mesh)
    updated_x_nx3 = grid.sample_points(nb_resampled_points)
    return aggregation(updated_x_nx3, neighbor_active_points_mask, x_nx3, sdf, tets, features=features, method=aggregation_method)

from render import render_pos_mask
def sample_error_density(x_nx3, sdf, tets, features, kal_mesh, error_maps, cameras, resolution, voxel_resolution, aggregation_method="interpolation", resampled_points_method="partial", device="cuda"):
    nb_resampled_points, neighbor_active_points_mask = resample_point_method(sdf, tets, x_nx3, method=resampled_points_method)
    if nb_resampled_points == 0:
        return x_nx3, sdf, features
    grid = Grid(voxel_resolution, -1.0, 2.0)
    
    grid_point_errors = torch.zeros(grid.density.shape, device=device)
    grid_point_counts = torch.zeros(grid.density.shape, device=device, dtype=torch.long)
    
    for i, current_camera in enumerate(cameras):
        current_position, current_mask = render_pos_mask(kal_mesh, current_camera, resolution)
        current_error = error_maps[i].unsqueeze(0)
        
        mask_squeeze = current_mask.squeeze(-1) > 0.5
        masked_position = current_position[mask_squeeze].to(device)
        masked_error = current_error[mask_squeeze].to(device)
        
        voxel_indices = grid.point_voxel_idx(masked_position)
        
        # Ensure voxel indices are within bounds
        valid_voxel_mask = (voxel_indices >= 0).all(dim=-1) & (voxel_indices < torch.tensor(grid.resolution, device=device)).all(dim=-1)
        voxel_indices = voxel_indices[valid_voxel_mask]
        masked_error = masked_error[valid_voxel_mask]
        
        # Flatten indices and filter valid ones
        flattened_voxel_indices = grid.three_d_to_flatten_idx(voxel_indices)
        valid_flattened_mask = (flattened_voxel_indices >= 0) & (flattened_voxel_indices < grid_point_errors.numel())
        flattened_voxel_indices = flattened_voxel_indices[valid_flattened_mask]
        masked_error = masked_error[valid_flattened_mask]
        
        # Create flattened tensors for scatter_add_
        grid_point_errors_flat = torch.zeros(grid.resolution[0] * grid.resolution[1] * grid.resolution[2], device=device)
        grid_points_counts_flat = torch.zeros_like(grid_point_errors_flat, dtype=torch.long)
        
        # Accumulate values
        grid_point_errors_flat.scatter_add_(0, flattened_voxel_indices, masked_error)
        grid_points_counts_flat.scatter_add_(0, flattened_voxel_indices, torch.ones_like(masked_error, dtype=torch.long))
        
        # Reshape back to 3D
        current_grid_point_errors = grid_point_errors_flat.view(grid.resolution[0], grid.resolution[1], grid.resolution[2])
        current_grid_point_counts = grid_points_counts_flat.view(grid.resolution[0], grid.resolution[1], grid.resolution[2])
        
        # Aggregate into the grid
        grid_point_errors += current_grid_point_errors
        grid_point_counts += current_grid_point_counts
    
    grid_point_errors = torch.where(grid_point_counts > 0, grid_point_errors / grid_point_counts, grid_point_errors)
    
    grid.density = grid_point_errors
    updated_x_nx3 = grid.sample_points(nb_resampled_points)
    return aggregation(updated_x_nx3, neighbor_active_points_mask, x_nx3, sdf, tets, features=features, method=aggregation_method)

def sample_error_density_gt(x_nx3, sdf, tets, kal_mesh, gt_kal_mesh, features = None, num_samples = 100, resolution = [1024, 1024], voxel_resolution = 32, aggregation_method="interpolation", resampled_points_method="partial", device="cuda"):
    camera_pos = fibonacci_sphere_sampling(num_samples, radius=3.0)
    fovy = np.deg2rad(45)
    cam_near_far=[0.1, 1000.0]
    pbar = tqdm.tqdm(total=num_samples)
    cameras = kal.render.camera.camera.Camera.from_args(
        eye=torch.tensor(camera_pos),
        at = torch.zeros(num_samples, 3),
        up = torch.tensor([0, 1, 0]),
        fov = fovy,
        near = cam_near_far[0],
        far = cam_near_far[1],
        height = resolution[0],
        width = resolution[1],
        device = device
    )

    error_maps = []
    for _, camera in enumerate(cameras):
        pred_buffers = render.render_mesh(kal_mesh, camera, resolution, return_types=["depth", "mask", "normals_face"], white_bg=False)
        pred_depth = pred_buffers["depth"]
        pred_depth = pred_depth / (pred_buffers["depth"] + 1e-8)
        pred_normals = pred_buffers["normals_face"]
        gt_buffer = render.render_mesh(gt_kal_mesh, camera, resolution, return_types=["depth", "mask", "normals_face"], white_bg=False)
        gt_depth = gt_buffer["depth"]
        gt_depth = gt_depth / (gt_buffer["depth"] + 1e-8)
        gt_normals = gt_buffer["normals_face"]
        error = torch.abs((pred_normals - gt_normals))
        error=error.sum(dim=-1)
        error_maps.append(error)
        pbar.update(1)
        
    error_maps = torch.cat(error_maps, dim=0)
    
    return sample_error_density(x_nx3, sdf, tets, features, kal_mesh, error_maps, cameras, resolution, voxel_resolution, device=device, aggregation_method=aggregation_method, resampled_points_method=resampled_points_method)
