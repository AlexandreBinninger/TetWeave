import torch
from torch.utils.data import Dataset
import render
import kaolin as kal

class DynamicDatasetOnGPU(Dataset):
    '''
        A dataset that generate the data on the fly directly on GPU.
    '''
    def __init__(self, mesh, batch_size, iterations, train_res, return_types=["mask", "depth", "normals_face"], cam_radius=3.0, device="cuda",):
        '''
            Args:
                - mesh (kaolin.rep.SurfaceMesh): The mesh to render.
                - batch_size (int): The batch size.
                - iterations (int): The number of iterations.
                - train_res [int, int]: The resolution of the training images.
                - return_types (list of str): The types of data to return. Possible values are "mask", "depth", "normals_face", and "vertex_color".
                
        '''
        self.mesh = mesh
        self.device = device
        self.batch_size = batch_size
        self.iterations = iterations
        self.train_res = train_res
        self.return_types = return_types
        self.cam_radius = cam_radius

    def __len__(self):
        return self.iterations

    def __getitem__(self, idx):
        cameras = render.get_random_camera_batch(self.batch_size, device=self.device, cam_radius=self.cam_radius)
        target = render.render_mesh(self.mesh, cameras, self.train_res, return_types=self.return_types, white_bg=False)
        return target, cameras

    def get_resolution(self):
        return self.train_res


def collate_DynamicDatasetOnGPU(batch):
    targets, cameras = zip(*batch)
    collated_targets = {}
    for key in targets[0].keys():
        collated_targets[key] = torch.stack([target[key] for target in targets], dim=0)
    collated_cameras = kal.render.camera.camera.Camera.cat(cameras)
    return collated_targets, collated_cameras