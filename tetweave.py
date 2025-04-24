import torch
import numpy as np
import tetgen
import sh

triangle_table = torch.tensor([
    [-1, -1, -1, -1, -1, -1],
    [1, 0, 2, -1, -1, -1],
    [4, 0, 3, -1, -1, -1],
    [1, 4, 2, 1, 3, 4],
    [3, 1, 5, -1, -1, -1],
    [2, 3, 0, 2, 5, 3],
    [1, 4, 0, 1, 5, 4],
    [4, 2, 5, -1, -1, -1],
    [4, 5, 2, -1, -1, -1],
    [4, 1, 0, 4, 5, 1],
    [3, 2, 0, 3, 5, 2],
    [1, 3, 5, -1, -1, -1],
    [4, 1, 2, 4, 3, 1],
    [3, 0, 4, -1, -1, -1],
    [2, 0, 1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1]
], dtype=torch.long)

#      3
#     /|\
#    / 2 \ 
#   / / \ \
#  0-------1

num_triangles_table = torch.tensor([0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0], dtype=torch.long)
base_tet_edges = torch.tensor([0, 1, 
                               0, 2, 
                               0, 3, 
                               1, 2, 
                               1, 3, 
                               2, 3], dtype=torch.long)
v_id = torch.pow(2, torch.arange(4, dtype=torch.long))

def tet_is_inside_oriented(vertices, tets):
    """
    Check if the tetrahedra are oriented correctly, i.e. if the normals of the faces point outwards. 
    This is tested by checking if the dot product of the normals and the vector from the barycenter of the tetrahedron to the barycenter of the face is negative.
    """
    faces = torch.tensor([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=torch.long)
    tets_combinations = tets[:, faces]
    normals_per_tets = torch.cross(vertices[tets_combinations[:, :, 1]] - vertices[tets_combinations[:, :, 0]], vertices[tets_combinations[:, :, 2]] - vertices[tets_combinations[:, :, 0]])
    tet_barycenter = vertices[tets].mean(dim=1)
    faces_barycenter = vertices[tets_combinations].mean(dim=2)
    faces_b_to_tet_b = faces_barycenter - tet_barycenter.unsqueeze(1)
    dot = torch.sum(normals_per_tets * faces_b_to_tet_b, dim=2)
    return dot < 0

def reorient_tetrahedra(vertices, tets):
    inside_oriented_triangles = tet_is_inside_oriented(vertices, tets)
    to_flip = inside_oriented_triangles.all(dim=1)
    tets[to_flip] = tets[to_flip][:, [0, 2, 1, 3]]
    return tets

def get_active_tets(tets, sdf):
    occ_n = sdf > 0
    occ_fx4 = occ_n[tets.reshape(-1)].reshape(-1, 4)
    occ_sum = torch.sum(occ_fx4, -1)
    active_tets = (occ_sum > 0) & (occ_sum < 4)
    return active_tets

def get_active_edges(edges, sdf):
    occ_n = sdf > 0
    occ_fx2 = occ_n[edges.reshape(-1)].reshape(-1, 2)
    active_edges_mask = occ_fx2.sum(-1) == 1
    active_edges = edges.reshape(-1, 2)[active_edges_mask]
    return active_edges

def get_active_points(sdf, tets):
    active_tets = get_active_tets(tets, sdf)
    active_points = torch.unique(tets[active_tets])
    return active_points

def neighbor_active_points(sdf, tets):
    active_tets = get_active_tets(tets, sdf)
    active_points = torch.unique(tets[active_tets])
    active_points_tets = torch.isin(tets, active_points).any(dim=1)
    neighor_active_points = torch.unique(tets[active_points_tets])
    return neighor_active_points

def get_edges(tetrahedra):
    tet_edges = torch.tensor([
        [0, 1], [0, 2], [0, 3],
        [1, 2], [1, 3], [2, 3]
    ], device=tetrahedra.device)
    edges = tetrahedra[:, tet_edges]
    return edges

def find_near_duplicates_sort(points, epsilon=1e-7):
    """
    Approximate detection of near-duplicate points in a point cloud using sorting.
    
    Args:
    - points: Tensor of shape (N, 3), where N is the number of points.
    - epsilon: Distance threshold for considering points as duplicates.
    
    Returns:
    - duplicate_indices: A tensor of shape (M,) where M is the number of points who have a near-duplicate.
    """
    sorted_indices = torch.argsort(points[:, 0]) 
    points_sorted = points[sorted_indices]
    
    diffs = points_sorted[1:] - points_sorted[:-1] 
    distances = torch.norm(diffs, dim=-1)

    close_mask = distances < epsilon
    duplicate_indices_1, duplicate_indices_2 = sorted_indices[:-1][close_mask], sorted_indices[1:][close_mask]
    duplicate_indices = torch.cat((duplicate_indices_1, duplicate_indices_2))
    return duplicate_indices
    

class TetWeave():
    def __init__(self, device='cuda'):
        self.device = device
    
    @torch.no_grad()
    def delaunay_simplices_tetgen(self, points):
        _tetgen = tetgen.TetGen(points.cpu().numpy(), np.array([], dtype=int))
        _, tets = _tetgen.tetrahedralize(switches="Q")
        return torch.tensor(tets, device=self.device)
    
    def tetrahedralize(self, points):
        duplicates = find_near_duplicates_sort(points)
        if duplicates.shape[0] > 0:
            with torch.no_grad():
                points[duplicates] += torch.randn_like(points[duplicates]) * 1e-6
        tetrahedra = self.delaunay_simplices_tetgen(points)
        tetrahedra = reorient_tetrahedra(points, tetrahedra)
        return tetrahedra
    
    def __call__(self, points, sdf, sh_coefficients = None, sh_deg=-1):
        tetrahedra = self.tetrahedralize(points)
        vertices, faces = self.marching_tetrahedra(points, tetrahedra, sdf, sh_coefficients, sh_deg)
        return vertices, faces, tetrahedra
    
    def _sort_edges(self, edges):
        """
        Sort last dimension of edges of shape (E, 2). Inspired by DMTet implementation.
        """
        with torch.no_grad():
            order = (edges[:, 0] > edges[:, 1]).long()
            order = order.unsqueeze(dim=1)

            a = torch.gather(input=edges, index=order, dim=1)
            b = torch.gather(input=edges, index=1 - order, dim=1)

        return torch.stack([a, b], -1)
    
    def marching_tetrahedra(self, vertices, tets, sdf, sh_coefficients = None, sh_deg=-1):
        """
        Marching tetrahedra. Inspired by DMTet implementation.
        """
        
        with torch.no_grad():
            occ_n = sdf > 0
            occ_fx4 = occ_n[tets.reshape(-1)].reshape(-1, 4)
            occ_sum = torch.sum(occ_fx4, -1)
            valid_tets = (occ_sum > 0) & (occ_sum < 4)
            occ_sum = occ_sum[valid_tets]

            all_edges = tets[valid_tets][:, base_tet_edges.to(self.device)].reshape(-1, 2)
            all_edges = self._sort_edges(all_edges)
            unique_edges, _idx_map = torch.unique(all_edges, dim=0, return_inverse=True)

            unique_edges = unique_edges.long()
            mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1
            mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device=self.device) * -1
            mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long, device=self.device)
            idx_map = mapping[_idx_map]
            interp_v = unique_edges[mask_edges] # edges are sorted here
        edges_to_interp = vertices[interp_v.reshape(-1)].reshape(-1, 2, 3)
        edges_to_interp_sdf = sdf[interp_v.reshape(-1)].reshape(-1, 2, 1)
        
        if sh_deg > 0 and sh_coefficients is not None and sh_coefficients.shape[-1] > 0:
            dim_sh_coefficients = sh_coefficients.shape[-1]
            edges_sh_coefficients = sh_coefficients[interp_v.reshape(-1)].reshape(-1, 2, dim_sh_coefficients)
            edges_direction_1 = edges_to_interp[:, 0] - edges_to_interp[:, 1]
            edges_direction_1 = edges_direction_1 / edges_direction_1.norm(dim=-1, keepdim=True)
            edges_direction_2 = edges_to_interp[:, 1] - edges_to_interp[:, 0]
            edges_direction_2 = edges_direction_2 / edges_direction_2.norm(dim=-1, keepdim=True)
            edges_directions = torch.stack([edges_direction_1, edges_direction_2], dim=1)
            
            edges_sh_coefficients = edges_sh_coefficients.reshape(-1, dim_sh_coefficients)
            edges_directions = edges_directions.reshape(-1, 3)
            
            edges_sh = sh.eval_sh(sh_deg, edges_sh_coefficients, edges_directions)
            edges_sh = edges_sh.reshape(-1, 2).unsqueeze(-1)
            edges_sh = torch.tanh(edges_sh) + 1.0
            edges_to_interp_sdf *= edges_sh    
        edges_to_interp_sdf[:, -1] *= -1
        denominator = edges_to_interp_sdf.sum(1, keepdim=True)

        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1]) / denominator
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

        idx_map = idx_map.reshape(-1, 6)


        tetindex = (occ_fx4[valid_tets] * v_id.to(self.device).unsqueeze(0)).sum(-1)

        num_triangles = num_triangles_table.to(self.device)[tetindex]
        triangle_table_device = triangle_table.to(self.device)

        # Generate triangle indices
        faces = torch.cat((
            torch.gather(input=idx_map[num_triangles == 1], dim=1,
                        index=triangle_table_device[tetindex[num_triangles == 1]][:, :3]).reshape(-1, 3),
            torch.gather(input=idx_map[num_triangles == 2], dim=1,
                        index=triangle_table_device[tetindex[num_triangles == 2]][:, :6]).reshape(-1, 3),
        ), dim=0)

        return verts, faces
    
    def create_random_points_ball(self, n_points, radius=1.0):
        points = torch.randn(n_points, 3, device=self.device)
        radii = torch.rand(n_points, 1, device=self.device) ** (1/3) * radius
        points = (points / points.norm(dim=-1, keepdim=True)) * radii
        return points