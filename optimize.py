import math
import numpy as np
import torch
import trimesh
import os
from util import *
import render
import loss

from tetweave import TetWeave, get_edges, get_active_edges, get_active_tets
from density_sampling import sample_block_functional_density, sample_error_density_gt
import tqdm
import kaolin as kal
from omegaconf import DictConfig, OmegaConf

from dataset_gpu import DynamicDatasetOnGPU, collate_DynamicDatasetOnGPU
from torch.utils.data import DataLoader
import imageio

from transformers import get_scheduler
from evaluations import compare_meshes


def write_config_to_yaml(config: DictConfig, file_path: str):
    """Write the OmegaConf configuration to a YAML file."""
    try:
        with open(file_path, 'w') as file:
            yaml_str = OmegaConf.to_yaml(config)
            file.write(yaml_str)
            print(f"Configuration written to {file_path}")
    except Exception as e:
        print(f"Error writing config to YAML: {e}")
        

def scheduler(optimizer, warmup_steps, total_steps, num_cycles, type, func):
    '''
        - type can be {"LambdaLR", "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"} 
    '''
    if type == "LambdaLR":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: func(x))
    else:
        if type in ["cosine", "cosine_with_restarts"]:
            scheduler_specific_kwargs = {"num_cycles": num_cycles}
        else:
            scheduler_specific_kwargs = {}
            
        return get_scheduler(type, optimizer, num_training_steps=warmup_steps, num_warmup_steps=total_steps, scheduler_specific_kwargs=scheduler_specific_kwargs)
        
        
class Optimizer:
    
    def __init__(self, conf, mesh_filename, out_dir, device = 'cuda') -> None:
        self.conf = conf
        self.ref_mesh_filename = mesh_filename
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        write_config_to_yaml(conf, os.path.join(self.out_dir, 'config.yaml'))
        self.resampling_dict = {res.iter: res.n_points for res in self.conf.optimization.resampling.steps}
        self.resampling_iters = list(self.resampling_dict.keys())
        
        self.iter = 0
        self.total_iter = self.conf.optimization.main_stage.iter + self.conf.optimization.late_stage.iter
        self.device = device
        
        gt_mesh = load_mesh(self.ref_mesh_filename, device=device)
        self.gt_mesh_kal = kal.rep.SurfaceMesh(vertices=gt_mesh.vertices, faces=gt_mesh.faces)
    
        self.tetweave = TetWeave(device=device)
        self.radius_ball = math.sqrt(3)
        x_nx3 = self.tetweave.create_random_points_ball(self.conf.optimization.n_points, radius=self.radius_ball)
        self.return_types = ["mask", "depth", "normals_face"]
        
        sdf = torch.rand_like(x_nx3[:, 0]) - 0.1
        init_sdf_feature_magnitude = 1.0
        if self.conf.spherical_harmonics.use:
            assert self.conf.spherical_harmonics.degree >= 0, "Degree must be greater than 0 if spherical harmonics are used"
            assert self.conf.spherical_harmonics.degree <= 4, "Degree must be no greater than 4 for spherical harmonics (4 is the max supported by the current implementation)"
            num_coeff = (self.conf.spherical_harmonics.degree + 1)**2
            self.sdf_offset_dim = num_coeff
            init_sdf_feature_magnitude = 0.0
        else:
            self.sdf_offset_dim = 0
        sh_coefficients = torch.rand(x_nx3.shape[0], self.sdf_offset_dim, device=device)*init_sdf_feature_magnitude
        self.x_nx3 = torch.nn.Parameter(x_nx3.clone().detach(), requires_grad=True)
        self.sdf = torch.nn.Parameter(sdf.clone().detach(), requires_grad=True)
        self.sh_coefficients = torch.nn.Parameter(sh_coefficients.clone().detach(), requires_grad=True)
        self.features = None  
        self.optimizer_sdf = torch.optim.AdamW([self.sdf], lr=conf.optimization.lr_sdf)
        self.optimizer_x_nx3 = torch.optim.AdamW([self.x_nx3], lr=conf.optimization.lr_x_nx3)
        self.optimizer_sh = torch.optim.AdamW([self.sh_coefficients], lr=conf.optimization.lr_sh)
        
        self.scheduler_sdf = scheduler(self.optimizer_sdf, conf.optimization.scheduler_features.warmup_steps, self.total_iter, conf.optimization.scheduler_features.num_cycles, conf.optimization.scheduler_features.type, lambda x: self.lr_schedule(x))
        self.scheduler_x_nx3 = scheduler(self.optimizer_x_nx3, conf.optimization.scheduler_x_nx3.warmup_steps, self.total_iter, conf.optimization.scheduler_x_nx3.num_cycles, conf.optimization.scheduler_x_nx3.type, lambda x: self.lr_schedule(x))
        self.scheduler_sh = scheduler(self.optimizer_sh, conf.optimization.scheduler_sh.warmup_steps, self.total_iter, conf.optimization.scheduler_sh.num_cycles, conf.optimization.scheduler_sh.type, lambda x: self.lr_schedule(x))

        self.pbar = tqdm.tqdm(total=self.total_iter, desc="Progress", dynamic_ncols=True)
        
        self.setup_dataloader()
    
    def update_iter(self, loss):
        if (self.iter % self.conf.logs_iter.save_vis == 0):
            self.save_render(iter=self.iter, loss=loss)
        if (self.iter > 0 and self.iter % self.conf.logs_iter.save_model == 0 or (self.conf.logs_iter.last_iter and self.iter == self.total_iter-1)):
            self.save_model(iter=self.iter)
        if (self.iter > 0 and self.iter % self.conf.logs_iter.save_mesh == 0 or (self.conf.logs_iter.last_iter and self.iter == self.total_iter-1)):
            self.save_mesh(iter=self.iter)
        if (self.iter > 0 and self.iter % self.conf.logs_iter.interactive == 0 or (self.iter == self.total_iter-1 and self.conf.visualization.last_iter)):
            self.interactive_visualization()
        self.iter += 1
        self.pbar.update(1)
    
    def lr_schedule(self, iter):
        lr = max(10**(-(iter)*0.0002), 0.1)
        return lr
        
    def train(self):
        self.main_stage()
        self.late_stage()
        if self.conf.visualization.render_final:
            self.final_renders()
        self.launch_evaluations()

    
    def setup_dataloader(self):
        dataset = DynamicDatasetOnGPU(
            self.gt_mesh_kal, 
            self.conf.optimization.batch, 
            self.total_iter, 
            self.conf.optimization.train_res, 
            return_types=self.return_types,
            cam_radius=self.conf.optimization.camera_radius,
            device=self.device)
        self.dataloader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=False, collate_fn=collate_DynamicDatasetOnGPU)
        self.loader_iter = iter(self.dataloader)

    
    @torch.no_grad()
    def resample(self, target_nb_points):
        vertices, faces, tetrahedra = self.tetweave(self.x_nx3, self.sdf, sh_coefficients=self.sh_coefficients, sh_deg=self.conf.spherical_harmonics.degree)
        
        nb_new_points = target_nb_points - self.x_nx3.shape[0]
        if nb_new_points > 0:
            new_x_nx3 = torch.ones(nb_new_points, 3, device=self.device)*2
            self.x_nx3 = torch.nn.Parameter(torch.cat([self.x_nx3, new_x_nx3], dim=0), requires_grad=True)
            new_sdf = torch.ones(nb_new_points, device=self.device)
            self.sdf = torch.nn.Parameter(torch.cat([self.sdf, new_sdf], dim=0), requires_grad=True)
            new_sh_coefficients = torch.zeros(nb_new_points, self.sdf_offset_dim, device=self.device)
            self.sh_coefficients = torch.nn.Parameter(torch.cat([self.sh_coefficients, new_sh_coefficients], dim=0))
        
        self.optimizer_x_nx3.param_groups = []
        self.optimizer_x_nx3.add_param_group({'params': [self.x_nx3]})
        self.optimizer_sdf.param_groups = []
        self.optimizer_sdf.add_param_group({'params': [self.sdf]})
        self.optimizer_sh.param_groups = []
        self.optimizer_sh.add_param_group({'params': [self.sh_coefficients]})
        
        mesh_kal = kal.rep.SurfaceMesh(vertices, faces)
        if self.conf.optimization.resampling.sampling_method == "functional_density":
            sample_block_functional_density(self.x_nx3, self.sdf, tetrahedra, mesh_kal, features=self.sh_coefficients, aggregation_method=self.conf.optimization.resampling.aggregation, resampled_points_method=self.conf.optimization.resampling.resampled_points_method)
        elif self.conf.optimization.resampling.sampling_method == "error_density_gt":
            sample_error_density_gt(self.x_nx3, self.sdf, tetrahedra, mesh_kal, self.gt_mesh_kal, features=self.sh_coefficients, num_samples=self.conf.optimization.resampling.num_cameras, resolution=self.conf.optimization.resampling.rendering_resolution, voxel_resolution=self.conf.optimization.resampling.voxel_resolution, aggregation_method=self.conf.optimization.resampling.aggregation, resampled_points_method=self.conf.optimization.resampling.resampled_points_method)
        else:
            raise ValueError(f"Unknown resampling method: {self.conf.optimization.resampling.sampling_method}")
        
    def get_next_batch(self):
        try:
            targets, target_cameras = next(self.loader_iter)
            for key in targets:
                targets[key] = targets[key].to(self.device)
            target_cameras = target_cameras.to(self.device)
            return targets, target_cameras
        except StopIteration: # Should not be triggered if the dataloader is correctly initialized, but it's safer to handle it
            self.loader_iter = iter(self.dataloader)
            return self.get_next_batch()
            
    def main_stage(self):
        '''
            Main stage of optimization. 
            Optimize both the sdf/sh coefficients, and the points positions (every 'accumulation_delaunay' iterations for the latter).
        '''
        self.optimizer_sdf.zero_grad()
        self.optimizer_x_nx3.zero_grad()
        tetrahedra = self.tetweave.tetrahedralize(self.x_nx3)
        
        while self.iter < self.conf.optimization.main_stage.iter:
            targets, target_cameras = self.get_next_batch()
            target_mask = targets["mask"]
            target_depth = targets["depth"]
            target_normals = targets["normals_face"]
            if (self.iter % self.conf.optimization.main_stage.accumulation_delaunay == 0):
                vertices, faces, tetrahedra = self.tetweave(self.x_nx3, self.sdf, sh_coefficients=self.sh_coefficients, sh_deg=self.conf.spherical_harmonics.degree)
            else:
                vertices, faces = self.tetweave.marching_tetrahedra(self.x_nx3, tetrahedra, self.sdf, sh_coefficients=self.sh_coefficients, sh_deg=self.conf.spherical_harmonics.degree)
            
            kal_mesh = kal.rep.SurfaceMesh(vertices, faces)
                
            edges = get_edges(tetrahedra)
            buffers = render.render_mesh(kal_mesh, target_cameras, self.conf.optimization.train_res, return_types=self.return_types, white_bg=False)
            mask_loss = (buffers['mask'] - target_mask).abs().mean() * self.conf.optimization.main_stage.loss.mask_reg 
            depth_loss = (((((buffers["depth"] - (target_depth))* target_mask)**2).sum(-1)+1e-8)).sqrt().mean() * self.conf.optimization.main_stage.loss.depth_reg
            normal_loss = (((((buffers['normals_face'] - (target_normals))* target_mask)**2).sum(-1)+1e-8)).sqrt().mean() * self.conf.optimization.main_stage.loss.normal_reg
            
            t_iter = self.iter / self.total_iter
            sdf_weight = self.conf.optimization.main_stage.loss.sdf_reg - (self.conf.optimization.main_stage.loss.sdf_reg - self.conf.optimization.main_stage.loss.sdf_reg/20)*min(1.0, 4.0 * t_iter)
            reg_sdf_loss = loss.sdf_reg_loss(self.sdf, edges.reshape(-1, 2)).mean() * sdf_weight # Loss to eliminate internal floaters that are not visible
            
            odt_loss = loss.E_ODT(self.x_nx3, tetrahedra) * self.conf.optimization.main_stage.loss.ODT_reg
            triangle_angle_fairness_loss = loss.triangle_angle_fairness(vertices, faces) * self.conf.optimization.main_stage.loss.triangle_angle_fairness_reg
            
            total_loss = mask_loss + depth_loss + normal_loss + reg_sdf_loss + odt_loss + triangle_angle_fairness_loss
            
            total_loss.backward()
            self.optimizer_sdf.step()
            self.optimizer_sh.step()
            self.scheduler_sdf.step()
            self.scheduler_sh.step()
            self.scheduler_x_nx3.step()        
            self.optimizer_sdf.zero_grad()
            self.optimizer_sh.zero_grad()

            if ((self.iter+1) % self.conf.optimization.main_stage.accumulation_delaunay == 0):
                self.optimizer_x_nx3.step()
                self.optimizer_x_nx3.zero_grad()
            
            if (self.iter in self.resampling_iters):
                self.optimizer_x_nx3.step()
                self.optimizer_x_nx3.zero_grad()
                self.resample(self.resampling_dict[self.iter])
                tetrahedra = self.tetweave.tetrahedralize(self.x_nx3)

            self.update_iter(total_loss.item())
        print("Main stage done")
    
    def late_stage(self):
        '''
            Late stage of optimization. Acts as a fine-tuning stage.
            We assume that the current mesh is close to the target.
            Does not change the point position, hence the tetrahedral grid is assumed fixed.
        '''
        tetrahedra = self.tetweave.tetrahedralize(self.x_nx3)
        while self.iter < self.total_iter:
            self.optimizer_sdf.zero_grad()
            self.optimizer_sh.zero_grad()
            targets, target_cameras = self.get_next_batch()
            target_mask = targets["mask"]
            target_depth = targets["depth"]
            target_normals = targets["normals_face"]
            vertices, faces = self.tetweave.marching_tetrahedra(self.x_nx3, tetrahedra, self.sdf, sh_coefficients=self.sh_coefficients, sh_deg=self.conf.spherical_harmonics.degree)
            
            kal_mesh = kal.rep.SurfaceMesh(vertices, faces)
            edges = get_edges(tetrahedra)
            buffers = render.render_mesh(kal_mesh, target_cameras, self.conf.optimization.train_res, return_types=self.return_types, white_bg=False)
            mask_loss = (buffers['mask'] - target_mask).abs().mean() * self.conf.optimization.late_stage.loss.mask_reg
            depth_loss = (((((buffers["depth"] - (target_depth))* target_mask)**2).sum(-1)+1e-8)).sqrt().mean() * self.conf.optimization.late_stage.loss.depth_reg
            normal_loss = (((((buffers['normals_face'] - (target_normals))* target_mask)**2).sum(-1)+1e-8)).sqrt().mean() * self.conf.optimization.late_stage.loss.normal_reg
            
            t_iter = self.iter / self.total_iter
            sdf_weight = self.conf.optimization.late_stage.loss.sdf_reg - (self.conf.optimization.late_stage.loss.sdf_reg - self.conf.optimization.late_stage.loss.sdf_reg/20)*min(1.0, 4.0 * t_iter)
            reg_sdf_loss = loss.sdf_reg_loss(self.sdf, edges.reshape(-1, 2)).mean() * sdf_weight # Loss to eliminate internal floaters that are not visible

            total_loss = mask_loss + depth_loss + normal_loss + reg_sdf_loss 
            
            total_loss.backward()
            self.optimizer_sdf.step()
            self.scheduler_sdf.step()
            self.optimizer_sh.step()
            self.scheduler_sh.step()
            
            self.update_iter(total_loss.item())
        print("Late stage done")
    
    @torch.no_grad()
    def save_render(self, iter, loss):
        print(f"Iter {iter}, Loss: {loss}")
        vertices, faces, _ = self.tetweave(self.x_nx3, self.sdf, sh_coefficients=self.sh_coefficients, sh_deg=self.conf.spherical_harmonics.degree)
        camera = render.get_rotate_camera(iter//self.conf.logs_iter.save_vis, iter_res=self.conf.optimization.display_res, device=self.device, ang_steps=self.conf.visualization.angle_steps)
        mesh_kal = kal.rep.SurfaceMesh(vertices, faces)
        val_buffers = render.render_mesh(mesh_kal, camera, self.conf.optimization.display_res, return_types=["normals_face", "depth"], white_bg=True)
        val_image = ((val_buffers["normals_face"][0].detach().cpu().numpy()+1)/2*255).astype(np.uint8)
        gt_buffers = render.render_mesh(self.gt_mesh_kal, camera, self.conf.optimization.display_res, return_types=["normals_face"], white_bg=True)
        gt_image = ((gt_buffers["normals_face"][0].detach().cpu().numpy()+1)/2*255).astype(np.uint8)
        difference = torch.sqrt((val_buffers["normals_face"][0] - gt_buffers["normals_face"][0])**2)
        difference_img = ((difference.detach().cpu().numpy())*255).astype(np.uint8)
        images = [
            add_text_to_image(gt_image, "ground truth", font_size=20),
            add_text_to_image(val_image, f"reconstruction iter: {iter}", font_size=20),
            add_text_to_image(difference_img, "difference", font_size=20)]
        

        images = np.concatenate(images, axis=1)
        imageio.imwrite(os.path.join(self.out_dir, f"iter_{iter}.png"), images)


    @torch.no_grad()
    def save_model(self, iter):
        print(f"Saving model.")
        torch.save({
                        'it': iter,
                        'sdf': self.sdf,
                        'x_nx3': self.x_nx3,
                        'sh_deg': self.conf.spherical_harmonics.degree,
                        'sh_c': self.sh_coefficients,
                    }, os.path.join(self.out_dir, 'model_{:04d}.pt'.format(iter)))
    
    @torch.no_grad()
    def save_mesh(self, iter):
        print(f"Saving mesh.")
        vertices, faces, _ = self.tetweave(self.x_nx3, self.sdf, sh_coefficients=self.sh_coefficients, sh_deg=self.conf.spherical_harmonics.degree)
        mesh = trimesh.Trimesh(vertices=vertices.cpu().numpy(), faces=faces.cpu().numpy())
        mesh.export(os.path.join(self.out_dir, 'mesh_{:04d}.obj'.format(iter)))
    
    @torch.no_grad()
    def interactive_visualization(self):
        import polyscope as ps
        import polyscope.imgui as psim
        vertices, faces, tetrahedra = self.tetweave(self.x_nx3, self.sdf, sh_coefficients=self.sh_coefficients, sh_deg=self.conf.spherical_harmonics.degree)
        edges = get_edges(tetrahedra)
        active_edges = get_active_edges(edges, self.sdf)
        active_points_idx, inverse_idx = torch.unique(active_edges.flatten(), return_inverse=True)
        active_points = self.x_nx3[active_points_idx]
        active_sdf = self.sdf[active_points_idx]
        active_edges = inverse_idx.reshape(-1, 2)
        
        ps.init()
        ps.register_point_cloud("Grid points", self.x_nx3.detach().cpu().numpy(), radius=0.002, enabled=False)
        ps.register_volume_mesh("Delaunay triangulation", self.x_nx3.detach().cpu().numpy(), tetrahedra.detach().cpu().numpy(), enabled=False)
        edges_network = ps.register_curve_network("Edges", self.x_nx3.detach().cpu().numpy(), edges.reshape(-1, 2).detach().cpu().numpy(), radius=0.001, enabled=False)
        edges_network.add_scalar_quantity("sdf", self.sdf.detach().cpu().numpy())
        ps.register_surface_mesh("Reconstructed mesh", vertices.detach().cpu().numpy(), faces.detach().cpu().numpy(), edge_width=1.0)
        ps.register_surface_mesh("Reference mesh", self.gt_mesh_kal.vertices.cpu().numpy(), self.gt_mesh_kal.faces.cpu().numpy(), enabled=False)
        active_points_cloud = ps.register_point_cloud("Active grid points", active_points.cpu().detach().numpy(), radius=0.002)
        active_points_cloud.add_scalar_quantity("sdf", active_sdf.cpu().detach().numpy())
        ps.register_curve_network("Active edges", active_points.cpu().detach().numpy(), active_edges.cpu().detach().numpy(), radius=0.001)
        
        ps.show()
    
    @torch.no_grad()
    def final_renders(self):
        print("Saving final renders.")
        folder_name = os.path.join(self.out_dir, "final_renders")
        os.makedirs(folder_name, exist_ok=True)
        vertices, faces, _ = self.tetweave(self.x_nx3, self.sdf, sh_coefficients=self.sh_coefficients, sh_deg=self.conf.spherical_harmonics.degree)
        kal_mesh = kal.rep.SurfaceMesh(vertices, faces)

        camera_pos = smooth_sphere_sampling(self.conf.visualization.nb_lattitudes, self.conf.visualization.points_per_latitude)
        fovy = np.deg2rad(45)
        cam_near_far=[0.1, 1000.0]
        pbar = tqdm.tqdm(total=len(camera_pos), desc="Final renders in progress.", dynamic_ncols=True)
        for i, pos in enumerate(camera_pos):
            camera = kal.render.camera.camera.Camera.from_args(
                eye=torch.tensor(pos), 
                at=torch.tensor([0, 0, 0]), 
                up=torch.tensor([0, 1, 0]),
                fov=fovy,
                near=cam_near_far[0], far=cam_near_far[1],
                height=self.conf.visualization.final_res[0], width=self.conf.visualization.final_res[1], 
                device=self.device)
            buffers = render.render_mesh(kal_mesh, camera, self.conf.visualization.final_res, return_types=["normals_face"], white_bg=True)
            image = ((buffers["normals_face"][0].detach().cpu().numpy()+1)/2*255).astype(np.uint8)
            gt_buffer = render.render_mesh(self.gt_mesh_kal, camera, self.conf.visualization.final_res, return_types=["normals_face"], white_bg=True)
            gt_image = ((gt_buffer["normals_face"][0].detach().cpu().numpy()+1)/2*255).astype(np.uint8)
            images = [
                add_text_to_image(gt_image, "ground truth", font_size=self.conf.visualization.final_res[0]//20),
                add_text_to_image(image, f"reconstruction - {i}", font_size=self.conf.visualization.final_res[0]//20)]
            imageio.imwrite(os.path.join(folder_name, f"iter_{i}.png"), np.concatenate(images, axis=1))
            pbar.update(1)
    
    @torch.no_grad()
    def launch_evaluations(self):
        import json
        vertices, faces, _ = self.tetweave(self.x_nx3, self.sdf, sh_coefficients=self.sh_coefficients, sh_deg=self.conf.spherical_harmonics.degree)
        
        # For the evaluation, we only keep the largest connected component of the mesh
        mesh = trimesh.Trimesh(vertices=vertices.cpu().numpy(), faces=faces.cpu().numpy())
        components = mesh.split()
        sizes = [len(c.faces) for c in components]
        max_index = sizes.index(max(sizes))
        largest_component = components[max_index]
        vertices = torch.tensor(largest_component.vertices, device=self.device, dtype=torch.float)
        faces = torch.tensor(largest_component.faces, device=self.device, dtype=torch.long)
        kal_mesh = kal.rep.SurfaceMesh(vertices=vertices, faces=faces)
        
        # Launching evaluation with 1_000_000 points
        n_points = 1_000_000
        print("Launching evaluation metrics.")
        evaluations = compare_meshes(self.gt_mesh_kal, kal_mesh, num_points=n_points)
        with open(os.path.join(self.out_dir, "evaluations.json"), "w") as file:
            json.dump(evaluations, file)
        
        print("CD:", evaluations["cd"])
        print("F1:", evaluations["f1"])
        print("ECD:", evaluations["ecd"])
        print("EF1:", evaluations["ef1"])
        print("NC:", evaluations["nc"])
        print("IN>5deg(%)", evaluations["in_5deg"])
        print("Aspect ratio > 4:", evaluations["aspect_ratio_gt_4"])
        print("Radius ratio > 4:", evaluations["radius_ratio_gt_4"])
        print("SA<10(%)", evaluations["sa_10_percent"])
        print("#V:", vertices.shape[0])
        print("#F:", faces.shape[0])
        