# Evaluation functions for meshes and/or point clouds
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
import numpy as np
from sklearn.neighbors import KDTree
import trimesh
import pyvista as pv
import pymeshlab



def get_cd_nc_f1_ecd_ef1(pred_points, pred_normals, gt_points, gt_normals, f1_threshold=0.001, ef1_threshold=0.005, ef1_radius=0.004, ef1_dotproduct_threshold=0.2):
    # from gt to pred
    pred_tree = KDTree(pred_points)
    dist, inds = pred_tree.query(gt_points, k=1)
    recall = np.sum(dist < f1_threshold) / float(len(dist))
    gt2pred_mean_cd1 = np.mean(dist)
    dist = np.square(dist)
    gt2pred_mean_cd2 = np.mean(dist)
    neighbor_normals = pred_normals[np.squeeze(inds, axis=1)]
    dotproduct = (np.sum(gt_normals*neighbor_normals, axis=1)) / (np.linalg.norm(gt_normals, axis=1) * np.linalg.norm(neighbor_normals, axis=1))
    gt2pred_nc = np.mean(np.abs(dotproduct))
    gt2pred_in_5deg = np.mean( (dotproduct<=np.cos((5.0/180.0)*np.pi)).astype(np.float32) )
    

    # from pred to gt
    gt_tree = KDTree(gt_points)
    dist, inds = gt_tree.query(pred_points, k=1)
    precision = np.sum(dist < f1_threshold) / float(len(dist))
    pred2gt_mean_cd1 = np.mean(dist)
    dist = np.square(dist)
    pred2gt_mean_cd2 = np.mean(dist)
    neighbor_normals = gt_normals[np.squeeze(inds, axis=1)]
    dotproduct = (np.sum(pred_normals*neighbor_normals, axis=1))
    pred2gt_nc = np.mean(np.abs(dotproduct))
    pred2gt_in_5deg = np.mean( (dotproduct<=np.cos((5.0/180.0)*np.pi)).astype(np.float32) )

    cd1 = gt2pred_mean_cd1+pred2gt_mean_cd1
    cd2 = gt2pred_mean_cd2+pred2gt_mean_cd2
    nc = (gt2pred_nc+pred2gt_nc)/2
    if recall+precision > 0: f1 = 2 * recall * precision / (recall + precision)
    else: f1 = 0
    in_5deg = (gt2pred_in_5deg+pred2gt_in_5deg)/2

    #sample gt edge points
    indslist = gt_tree.query_radius(gt_points, ef1_radius)
    flags = np.zeros([len(gt_points)],bool)
    for p in range(len(gt_points)):
        inds = indslist[p]
        if len(inds)>0:
            this_normals = gt_normals[p:p+1]
            neighbor_normals = gt_normals[inds]
            dotproduct = np.abs(np.sum(this_normals*neighbor_normals, axis=1))
            if np.any(dotproduct < ef1_dotproduct_threshold):
                flags[p] = True
    gt_edge_points = np.ascontiguousarray(gt_points[flags])

    #sample pred edge points
    indslist = pred_tree.query_radius(pred_points, ef1_radius)
    flags = np.zeros([len(pred_points)],bool)
    for p in range(len(pred_points)):
        inds = indslist[p]
        if len(inds)>0:
            this_normals = pred_normals[p:p+1]
            neighbor_normals = pred_normals[inds]
            dotproduct = np.abs(np.sum(this_normals*neighbor_normals, axis=1))
            if np.any(dotproduct < ef1_dotproduct_threshold):
                flags[p] = True
    pred_edge_points = np.ascontiguousarray(pred_points[flags])

    #ecd ef1
    if len(pred_edge_points)==0: pred_edge_points=np.zeros([486,3],np.float32)
    if len(gt_edge_points)==0:
        ecd1 = 0
        ecd2 = 0
        ef1 = 1
    else:
        # from gt to pred
        tree = KDTree(pred_edge_points)
        dist, inds = tree.query(gt_edge_points, k=1)
        erecall = np.sum(dist < ef1_threshold) / float(len(dist))
        gt2pred_mean_ecd1 = np.mean(dist)
        dist = np.square(dist)
        gt2pred_mean_ecd2 = np.mean(dist)

        # from pred to gt
        tree = KDTree(gt_edge_points)
        dist, inds = tree.query(pred_edge_points, k=1)
        eprecision = np.sum(dist < ef1_threshold) / float(len(dist))
        pred2gt_mean_ecd1 = np.mean(dist)
        dist = np.square(dist)
        pred2gt_mean_ecd2 = np.mean(dist)

        ecd1 = gt2pred_mean_ecd1+pred2gt_mean_ecd1
        ecd2 = gt2pred_mean_ecd2+pred2gt_mean_ecd2
        if erecall+eprecision > 0: ef1 = 2 * erecall * eprecision / (erecall + eprecision)
        else: ef1 = 0

    return cd1, cd2, nc, f1, ecd1, ecd2, ef1, in_5deg

def get_ratio_metrics_mesh(mesh_comp):
    def create_mesh(vertices, faces):
        faces_prefaced = np.hstack([np.full((faces.shape[0], 1), 3), faces])
        mesh = pv.PolyData(vertices, faces_prefaced)
        return mesh
    vertices = mesh_comp.vertices.cpu().numpy()
    faces = mesh_comp.faces.cpu().numpy()   
    mesh_pv = create_mesh(vertices, faces)
    aspect_ratio = mesh_pv.compute_cell_quality(quality_measure='aspect_ratio')
    aspect_ratio = aspect_ratio.get_array(name='CellQuality')
    radius_ratio = mesh_pv.compute_cell_quality(quality_measure='radius_ratio')
    radius_ratio = radius_ratio.get_array(name='CellQuality')
    aspect_ratio_gt_4 = np.sum(aspect_ratio > 4) / len(aspect_ratio)
    radius_ratio_gt_4 = np.sum(radius_ratio > 4) / len(radius_ratio)
    return aspect_ratio_gt_4, radius_ratio_gt_4


def count_self_intersections_vertices(filename):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(filename)
    nb_faces = ms.current_mesh().face_number()
    ms.compute_selection_by_self_intersections_per_face()

    mesh = ms.current_mesh()
    f_selected = mesh.face_selection_array()
    percentage_faces = (f_selected.sum()/nb_faces)*100
    return percentage_faces


def get_regularity_metrics(mesh):
    vertices = mesh.vertices.cpu().numpy()
    faces = mesh.faces.cpu().numpy()
    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    angles = trimesh_mesh.face_angles * 180 / np.pi  # Convert radians to degrees
    min_angle = np.min(angles)
    max_angle = np.max(angles)
    sa_10 = np.any(min_angle < 10, axis=0)
    sa_10_percent = 100 * np.sum(sa_10) / trimesh_mesh.faces.shape[0]

    return min_angle, max_angle, sa_10_percent 


def compare_meshes(mesh_gt, mesh_comp, num_points=100000):
    meshgt_p3d = Meshes(verts=[mesh_gt.vertices], faces=[mesh_gt.faces])
    mesh_comp_p3d = Meshes(verts=[mesh_comp.vertices], faces=[mesh_comp.faces])
    sampled_points_gt, sampled_normals_gt = sample_points_from_meshes(meshgt_p3d, num_points, return_normals=True)
    sampled_points_comp, sampled_normals_comp = sample_points_from_meshes(mesh_comp_p3d, num_points, return_normals=True)
    
    f1_threshold = 0.001
    ef1_radius = 0.004
    ef1_dotproduct_threshold = 0.2
    ef1_threshold = 0.005
    cd1, cd2, nc, f1, ecd1, ecd2, ef1, in_5deg = get_cd_nc_f1_ecd_ef1(sampled_points_comp.squeeze(0).cpu().numpy(), sampled_normals_comp.squeeze(0).cpu().numpy(), sampled_points_gt.squeeze(0).cpu().numpy(), sampled_normals_gt.squeeze(0).cpu().numpy(), f1_threshold, ef1_threshold, ef1_radius, ef1_dotproduct_threshold)
    aspect_ratio_gt_4, radius_ratio_gt_4 = get_ratio_metrics_mesh(mesh_comp)
    min_angle, max_angle, sa_10_percent = get_regularity_metrics(mesh_comp)
    return {
        "cd": cd2,
        "nc": nc,
        "f1": f1,
        "ecd": ecd2,
        "ef1": ef1,
        "in_5deg": in_5deg,
        "aspect_ratio_gt_4": aspect_ratio_gt_4,
        "radius_ratio_gt_4": radius_ratio_gt_4,
        "min_angle": min_angle,
        "max_angle": max_angle,
        "sa_10_percent": sa_10_percent
    }

