import torch

def sdf_reg_loss(sdf, all_edges):
    sdf_f1x6x2 = sdf[all_edges.reshape(-1)].reshape(-1,2)
    mask = torch.sign(sdf_f1x6x2[...,0]) != torch.sign(sdf_f1x6x2[...,1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,0], (sdf_f1x6x2[...,1] > 0).float()) + \
            torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,1], (sdf_f1x6x2[...,0] > 0).float())
    return sdf_diff

def compute_angle(a, b, c):
    """
    Compute the angle between vectors a-b and c-b (in radians)
    """
    ab = a - b
    cb = c - b
    cosine_angle = torch.sum(ab * cb, dim=-1) / ((torch.norm(ab, dim=-1) * torch.norm(cb, dim=-1))+1e-6)
    angle = torch.acos(cosine_angle.clamp(-1, 1))  
    return angle

def triangle_angle_fairness(vertices, faces):
    """
    Compute the fairness loss for a mesh (vertices and faces).
    
    Args:
    - vertices: Tensor of shape (V, 3), where V is the number of vertices in the mesh.
    - faces: Tensor of shape (F, 3), where F is the number of triangular faces.

    Returns:
    - loss: A scalar value representing the fairness loss of the whole mesh.
    """
    tri_verts = vertices[faces]
    angles = torch.zeros((faces.shape[0], 3), device=vertices.device)
    angles[:, 0] = compute_angle(tri_verts[:, 1, :], tri_verts[:, 0, :], tri_verts[:, 2, :])
    angles[:, 1] = compute_angle(tri_verts[:, 0, :], tri_verts[:, 1, :], tri_verts[:, 2, :])
    angles[:, 2] = compute_angle(tri_verts[:, 0, :], tri_verts[:, 2, :], tri_verts[:, 1, :]) #TODO could be torch.pi - (angles[:, 0] + angles[:, 1])

    target_angle = torch.pi / 3
    target_angle_tensor = torch.ones_like(angles) * target_angle
    fairness_loss = torch.nn.functional.mse_loss(angles, target_angle_tensor)

    return fairness_loss

def compute_circumcenter(vertices, tets):
    """
    Compute the circumcenters of a set of tetrahedra.
    
    Args:
    - vertices: Tensor of shape (V, 3), where V is the number of vertices.
    - tets: Tensor of shape (F, 4), where F is the number of tetrahedra.
    
    Returns:
    - circumcenters: Tensor of shape (F, 3) representing the circumcenters.
    - circumradii: Tensor of shape (F,) representing the radii of the circumspheres.
    
    Source: https://mathworld.wolfram.com/Circumsphere.html or https://rodolphe-vaillant.fr/entry/127/find-a-tetrahedron-circumcenter
    """
    tet_vertices = vertices[tets]
    v0 = tet_vertices[:, 0]
    v1 = tet_vertices[:, 1]
    v2 = tet_vertices[:, 2]
    v3 = tet_vertices[:, 3]
    a = v1 - v0
    b = v2 - v0
    c = v3 - v0
    d = 2 * torch.det(torch.stack([a, b, c], dim=-1))  # Shape (F,)
    A = torch.linalg.norm(a, dim=-1) ** 2  # Shape (F,)
    B = torch.linalg.norm(b, dim=-1) ** 2  # Shape (F,)
    C = torch.linalg.norm(c, dim=-1) ** 2  # Shape (F,)
    circumcenters = v0 + (A.unsqueeze(-1) * torch.cross(b, c, dim=-1) +
                          B.unsqueeze(-1) * torch.cross(c, a, dim=-1) +
                          C.unsqueeze(-1) * torch.cross(a, b, dim=-1)) / d.unsqueeze(-1)
    circumradii = (v0 - circumcenters).norm(dim=-1)
    return circumcenters, circumradii


def compute_inertia_tensor(vertices, tets, circumcenters, volumes):
    """
    Compute the sum of principal moments of inertia (trace of inertia tensor) for tetrahedra using the formula in the referenced paper.

    Args:
    - vertices: Tensor of shape (V, 3), where V is the number of vertices.
    - tets: Tensor of shape (F, 4), where F is the number of tetrahedra.
    - circumcenters: Tensor of shape (F, 3), circumcenters for each tetrahedron.
    - volumes: Tensor of shape (F,), volumes of the tetrahedra.

    Returns:
    - M_T: Tensor of shape (F,) representing the sum of the principal moments of inertia for each tetrahedron.
    """

    tet_vertices = vertices[tets]  # Shape: (F, 4, 3)
    rel_vertices = tet_vertices - circumcenters.unsqueeze(1)  # Shape: (F, 4, 3)
    x = rel_vertices[..., 0]  # x-coordinates (F, 4)
    y = rel_vertices[..., 1]  # y-coordinates (F, 4)
    z = rel_vertices[..., 2]  # z-coordinates (F, 4)

    x_sum = x[..., 0] * x[..., 0] + x[..., 1] * x[..., 1] + x[..., 2] * x[..., 2] + x[..., 3] * x[..., 3] \
            + x[..., 0] * x[..., 1] + x[..., 0] * x[..., 2] + x[..., 0] * x[..., 3] \
            + x[..., 1] * x[..., 2] + x[..., 1] * x[..., 3] + x[..., 2] * x[..., 3]
    
    y_sum = y[..., 0] * y[..., 0] + y[..., 1] * y[..., 1] + y[..., 2] * y[..., 2] + y[..., 3] * y[..., 3] \
            + y[..., 0] * y[..., 1] + y[..., 0] * y[..., 2] + y[..., 0] * y[..., 3] \
            + y[..., 1] * y[..., 2] + y[..., 1] * y[..., 3] + y[..., 2] * y[..., 3]
    
    z_sum = z[..., 0] * z[..., 0] + z[..., 1] * z[..., 1] + z[..., 2] * z[..., 2] + z[..., 3] * z[..., 3] \
            + z[..., 0] * z[..., 1] + z[..., 0] * z[..., 2] + z[..., 0] * z[..., 3] \
            + z[..., 1] * z[..., 2] + z[..., 1] * z[..., 3] + z[..., 2] * z[..., 3]
    
    Ix = 6.0 * volumes * (y_sum + z_sum) / 60.0
    Iy = 6.0 * volumes * (x_sum + z_sum) / 60.0
    Iz = 6.0 * volumes * (x_sum + y_sum) / 60.0

    M_T = Ix + Iy + Iz
    return M_T

def compute_shell_moment(circumradii, volumes):
    """
    Compute the moment of inertia for a spherical shell with equivalent mass.
    
    Args:
    - circumradii: Tensor of shape (F,) representing the circumradii.
    - volumes: Tensor of shape (F,) representing the volumes of the tetrahedra.
    
    Returns:
    - M_S: Moment of inertia for each tetrahedron's circumshell.
    """
    M_S =  2.0/5.0*volumes * (circumradii ** 2)
    return M_S

def E_ODT(vertices, tets):
    """
    Compute the optimal Delaunay triangulation energy for a tetrahedral mesh.
    
    Args:
    - vertices: Tensor of shape (V, 3), where V is the number of vertices.
    - tets: Tensor of shape (F, 4), where F is the number of tetrahedra.
    
    Returns:
    - energy: A scalar value representing the ODT energy of the whole mesh.
    """
    circumcenters, circumradii = compute_circumcenter(vertices, tets)
    tet_vertices = vertices[tets]

    v0, v1, v2, v3 = tet_vertices[:, 0], tet_vertices[:, 1], tet_vertices[:, 2], tet_vertices[:, 3]
    volumes = torch.abs(torch.det(torch.stack([v1 - v0, v2 - v0, v3 - v0], dim=-1))) / 6.0

    M_T = compute_inertia_tensor(vertices, tets, circumcenters, volumes)
    M_S = compute_shell_moment(circumradii, volumes)
    energy = torch.mean(torch.abs(M_T - M_S))

    return energy
