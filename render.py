import numpy as np
import copy
import math
from ipywidgets import interactive, HBox, VBox, FloatLogSlider, IntSlider

import torch
import nvdiffrast.torch as dr
import kaolin as kal
import util

###############################################################################
# Functions adapted from https://github.com/NVlabs/nvdiffrec
###############################################################################

def get_random_camera_batch(batch_size, fovy = np.deg2rad(45), iter_res=[2048,2048], cam_near_far=[0.1, 1000.0], cam_radius=3.0, device="cuda"):
    camera_pos = torch.stack(kal.ops.coords.spherical2cartesian(
        *kal.ops.random.sample_spherical_coords((batch_size,), azimuth_low=0., azimuth_high=math.pi * 2,
                                                elevation_low=-math.pi / 2., elevation_high=math.pi / 2., device=device),
        cam_radius
    ), dim=-1)
    return kal.render.camera.Camera.from_args(
        eye=camera_pos + torch.rand((batch_size, 1), device=device) * 0.5 - 0.25,
        at=torch.zeros(batch_size, 3),
        up=torch.tensor([[0., 1., 0.]]),
        fov=fovy,
        near=cam_near_far[0], far=cam_near_far[1],
        height=iter_res[0], width=iter_res[1],
        device=device
    )

def get_rotate_camera(itr, fovy = np.deg2rad(45), iter_res=[512,512], cam_near_far=[0.1, 1000.0], cam_radius=3.0, device="cuda", ang_steps=10):
    ang = (itr / ang_steps) * np.pi * 2
    camera_pos = torch.stack(kal.ops.coords.spherical2cartesian(torch.tensor(0.4), torch.tensor(ang), -torch.tensor(cam_radius)))
    return kal.render.camera.Camera.from_args(
        eye=camera_pos,
        at=torch.zeros(3),
        up=torch.tensor([0., 1., 0.]),
        fov=fovy,
        near=cam_near_far[0], far=cam_near_far[1],
        height=iter_res[0], width=iter_res[1],
        device='cuda'
    )

glctx = dr.RasterizeCudaContext()
def render_mesh(mesh: kal.rep.SurfaceMesh, camera, iter_res, return_types = ["mask", "depth", "normals_face"], white_bg=False):
    '''
        return_types: List of strings. Possible values are "mask", "depth", "normals_face"
    '''
    # vertex_color is a tensor of shape (num_vertices, d).
    vertices_camera = camera.extrinsics.transform(mesh.vertices)

    # Projection: nvdiffrast take clip coordinates as input to apply barycentric perspective correction.
    # Using `camera.intrinsics.transform(vertices_camera) would return the normalized device coordinates.
    proj = camera.projection_matrix().unsqueeze(1)
    proj[:, :, 1, 1] = -proj[:, :, 1, 1]
    homogeneous_vecs = kal.render.camera.up_to_homogeneous(
        vertices_camera
    )
    vertices_clip = (proj @ homogeneous_vecs.unsqueeze(-1)).squeeze(-1)
    faces_int = mesh.faces.int()

    rast, _ = dr.rasterize(
        glctx, vertices_clip, faces_int, iter_res)

    out_dict = {}
    for type in return_types:
        if type == "mask" :
            img = dr.antialias((rast[..., -1:] > 0).float(), rast, vertices_clip, faces_int)
        elif type == "depth":
            depth = homogeneous_vecs[..., 2] / (homogeneous_vecs[..., 3]+1e-8)
            img = dr.interpolate(depth.unsqueeze(-1), rast, faces_int)[0]
        elif type == "normals_face" :
            img = dr.interpolate(
                mesh.face_normals.reshape(len(mesh), -1, 3), rast,
                torch.arange(mesh.faces.shape[0] * 3, device='cuda', dtype=torch.int).reshape(-1, 3)
            )[0]
        if white_bg:
            bg = torch.ones_like(img)
            alpha = (rast[..., -1:] > 0).float() 
            img = torch.lerp(bg, img, alpha)
        out_dict[type] = img

    return out_dict

def render_pos_mask(mesh: kal.rep.SurfaceMesh, camera, res):
    vertices_camera = camera.extrinsics.transform(mesh.vertices)
    
    # Projection: nvdiffrast take clip coordinates as input to apply barycentric perspective correction.
    # Using `camera.intrinsics.transform(vertices_camera) would return the normalized device coordinates.
    proj = camera.projection_matrix().unsqueeze(1)
    proj[:, :, 1, 1] = -proj[:, :, 1, 1]
    homogeneous_vecs = kal.render.camera.up_to_homogeneous(
        vertices_camera
    )
    vertices_clip = (proj @ homogeneous_vecs.unsqueeze(-1)).squeeze(-1)
    faces_int = mesh.faces.int()
    rast, _ = dr.rasterize(
        glctx, vertices_clip, faces_int, res)
    
    img_mask = dr.antialias((rast[..., -1:] > 0).float(), rast, vertices_clip, faces_int)
    img_position = dr.interpolate(mesh.vertices, rast, faces_int)[0]

    return img_position, img_mask
