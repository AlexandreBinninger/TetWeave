import torch
import numpy as np
from util import rotate_x


class BlueNoise():
    def __init__(self, path="./assets/blueNoise3d.npz", device="cuda") -> None:
        self.npz = np.load(path)
        self.device = device
        self.noise_levels = []
        for key in self.npz.keys():
            self.noise_levels.append((torch.tensor(self.npz[key], dtype=torch.float32, device=device) -0.5)*2.0)
    
    def aggregate_noise(self, level):
        '''
            Returns blue noise at a specified level. Level must be a positive integer. No effect after level 6.
            Amount of points is:
            - Level 0: 32
            - Level 1: 140
            - Level 2: 505
            - Level 3: 1738
            - Level 4: 5898
            - Level 5: 19938
            - Level 6: 67323
        '''
        return torch.cat(self.noise_levels[:level+1], dim=0)
    
    def aggregate_randomized_targeted_noise(self, nb_points):
        if nb_points <= 32:
            random_points_indx = torch.randperm(32, device=self.device)[:nb_points]
            return self.aggregate_noise(0)[random_points_indx]
        elif nb_points <= 140:
            random_points_indx = torch.randperm(140, device=self.device)[:nb_points]
            return self.aggregate_noise(1)[random_points_indx]
        elif nb_points <= 505:
            random_points_indx = torch.randperm(505, device=self.device)[:nb_points]
            return self.aggregate_noise(2)[random_points_indx]
        elif nb_points <= 1738:
            random_points_indx = torch.randperm(1738, device=self.device)[:nb_points]
            return self.aggregate_noise(3)[random_points_indx]
        elif nb_points <= 5898:
            random_points_indx = torch.randperm(5898, device=self.device)[:nb_points]
            return self.aggregate_noise(4)[random_points_indx]
        elif nb_points <= 19938:
            random_points_indx = torch.randperm(19938, device=self.device)[:nb_points]
            return self.aggregate_noise(5)[random_points_indx]
        elif nb_points <= 67323:
            random_points_indx = torch.randperm(67323, device=self.device)[:nb_points]
            return self.aggregate_noise(6)[random_points_indx]
        else:
            raise ValueError("Number of points must be less than 67323")
    
    def aggregate_scaled_targeted_noise(self, nb_points):
        if nb_points <= 32:
            noise = self.aggregate_noise(0)
        elif nb_points <= 140:
            noise = self.aggregate_noise(1)
        elif nb_points <= 505:
            noise = self.aggregate_noise(2)
        elif nb_points <= 1738:
            noise = self.aggregate_noise(3)
        elif nb_points <= 5898:
            noise = self.aggregate_noise(4)
        elif nb_points <= 19938:
            noise = self.aggregate_noise(5)
        elif nb_points <= 67323:
            noise = self.aggregate_noise(6)
        else:
            raise ValueError("Number of points must be less than 67323")
        max_abs_coords = noise.abs().max(dim=1).values
        scaling_factors = 1.0 / max_abs_coords
        sorted_scaling, _ = scaling_factors.sort()
        s = sorted_scaling[-nb_points]  # m-th largest scaling factor
        scaled_noise = noise * s

        # Filter points that are still inside the cube
        inside_mask = scaled_noise.abs().max(dim=1).values <= 1.0
        scaled_noise = scaled_noise[inside_mask]
        return scaled_noise

    def get_noise(self, level, scale=1.0, rotation_matrix = torch.eye(3, device="cuda")):
        noise = self.aggregate_noise(level)
        noise = noise * scale
        noise = torch.matmul(noise, rotation_matrix.T)
        return noise

    def get_packed_noise(self, level, pack = None, scale=1.0, rotation_matrix = torch.eye(3, device="cuda")):
        '''
            Pack is a 3D matrix of boolean that dispatches the noise to the right place.
            Eg, in 2D, if pack = [ True, False
                                    False, True]
            Then the noise is dispatched to the top left and bottom right corners
        '''
        if pack is None:
            pack = torch.tensor([[[True]]], device=self.device)  # default 1x1x1 pack if none is provided
        assert len(pack.shape) == 3, "Pack must be a 3D tensor"
        noise = self.aggregate_noise(level)
        pack_indices = torch.nonzero(pack, as_tuple=False) 
        expanded_noise = noise.unsqueeze(0) + pack_indices.unsqueeze(1)  
        packed_noise = expanded_noise.contiguous().view(-1, 3)
        mid_x = (pack.shape[0]-1)/2
        mid_y = (pack.shape[1]-1)/2
        mid_z = (pack.shape[2]-1)/2
        packed_noise = packed_noise - torch.tensor([mid_x, mid_y, mid_z], device=self.device)
        packed_noise = packed_noise * scale
        packed_noise = torch.matmul(packed_noise, rotation_matrix.T)
        return packed_noise

    def visualize(self):
        import polyscope as ps
        import polyscope.imgui as psim
        ps.init()
        for i in range(7):
            ps.register_point_cloud(f"Blue noise level {i}", self.aggregate_noise(i).cpu().numpy())
        
        scale=2.0
        angle = np.pi/10
        rotatation = rotate_x(angle, device="cuda")[:3, :3]
        noise = self.get_noise(6, scale, rotatation)
        ps.register_point_cloud("Rotated blue noise", noise.cpu().numpy())
        pack = torch.tensor([[[True, False], [True, False]]], device="cuda")
        print(pack.shape)
        packed_noise = self.get_packed_noise(3, pack, scale=scale, rotation_matrix=rotatation)
        ps.register_point_cloud("Packed blue noise", packed_noise.cpu().numpy())
        
        noise_64 = self.aggregate_scaled_targeted_noise(64)
        noise_450 = self.aggregate_scaled_targeted_noise(450)
        noise_1000 = self.aggregate_scaled_targeted_noise(1000)
        noise_12345 = self.aggregate_scaled_targeted_noise(12345)
        ps.register_point_cloud("64 points", noise_64.cpu().numpy())
        ps.register_point_cloud("450 points", noise_450.cpu().numpy())
        ps.register_point_cloud("1000 points", noise_1000.cpu().numpy())
        ps.register_point_cloud("12345 points", noise_12345.cpu().numpy())
        
        def callback():
            nonlocal angle
            changed_angle, angle = psim.InputFloat("Angle", angle, 0.01, 0.1)
            if changed_angle:
                rotatation = rotate_x(angle, device="cuda")[:3, :3]
                noise = self.get_noise(6, scale, rotatation)
                ps.register_point_cloud("Rotated blue noise", noise.cpu().numpy())

        x_axis = torch.tensor([[1, 0, 0]], device="cuda")
        y_axis = torch.tensor([[0, 1, 0]], device="cuda")
        z_axis = torch.tensor([[0, 0, 1]], device="cuda")
        ps.register_point_cloud("X axis", x_axis.cpu().numpy())
        ps.register_point_cloud("Y axis", y_axis.cpu().numpy())
        ps.register_point_cloud("Z axis", z_axis.cpu().numpy())
        ps.set_user_callback(callback)
        ps.show()

if __name__ == "__main__":
    bn = BlueNoise()
    bn.visualize()