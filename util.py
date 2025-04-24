import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr

###############################################################################
# Functions adapted from https://github.com/NVlabs/nvdiffrec
###############################################################################

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def length(x: torch.Tensor, eps: float =1e-8) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-8) -> torch.Tensor:
    return x / length(x, eps)

def rotate_x(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[1,  0, 0, 0], 
                         [0,  c, s, 0], 
                         [0, -s, c, 0], 
                         [0,  0, 0, 1]], dtype=torch.float32, device=device)
    
class Mesh:
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces
        
    def auto_normals(self):
        v0 = self.vertices[self.faces[:, 0], :]
        v1 = self.vertices[self.faces[:, 1], :]
        v2 = self.vertices[self.faces[:, 2], :]
        nrm = safe_normalize(torch.linalg.cross(v1 - v0, v2 - v0))
        self.nrm = nrm

def load_mesh(path, device):
    mesh_np = trimesh.load(path, force='mesh')
    vertices = torch.tensor(mesh_np.vertices, device=device, dtype=torch.float)
    faces = torch.tensor(mesh_np.faces, device=device, dtype=torch.long)
    
    # Normalize
    vmin, vmax = vertices.min(dim=0)[0], vertices.max(dim=0)[0]
    scale = 1.8 / torch.max(vmax - vmin).item()
    vertices = vertices - (vmax + vmin) / 2 # Center mesh on origin
    vertices = vertices * scale # Rescale to [-0.9, 0.9]
    return Mesh(vertices, faces)

def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(
        attr, rast, attr_idx, rast_db=rast_db,
        diff_attrs=None if rast_db is None else 'all')

def print_gpu_memory():
    print("Total memory allocated:", torch.cuda.memory_allocated() / 1024**3, "GB")
    print("Max memory allocated:", torch.cuda.max_memory_allocated() / 1024**3, "GB")
    print("Total memory cached:", torch.cuda.memory_reserved() / 1024**3, "GB")
    print("Max memory cached:", torch.cuda.max_memory_reserved() / 1024**3, "GB")

def fibonacci_sphere_sampling(n, radius=3.0):
    points = np.zeros((n, 3))
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n):
        theta = 2 * np.pi * i / golden_ratio  
        psi = np.arccos(1-((2*i)/n))
        x, y, z = np.cos(theta) * np.sin(psi), np.sin(theta) * np.sin(psi), np.cos(psi)
        points[i] = [radius * x, radius * y, radius * z]
    return points


def smooth_sphere_sampling(nb_latitudes=10, points_per_latitude=20, radius=3.0):
    """
    Generates evenly spaced points on a sphere that move smoothly from top to bottom.
    
    Args:
        nb_latitudes (int): Number of latitudinal slices (from top to bottom).
        points_per_latitude (int): Number of points per latitude circle.
        
    Returns:
        np.ndarray: Array of shape (nb_latitudes * points_per_latitude, 3), where each row is an (x, y, z) point on the sphere.
    """
    points = []
    total_points = nb_latitudes * points_per_latitude
    theta = np.linspace(0, np.pi, total_points+2, endpoint=False)[1:-1]
    phi = np.linspace(0, 2*np.pi, points_per_latitude, endpoint=False)
    phi = np.tile(phi, nb_latitudes)
    z = np.sin(theta) * np.cos(phi)
    x = np.sin(theta) * np.sin(phi)
    y = np.cos(theta)
    points = np.stack([x, y, z], axis=-1)
    points *= radius
    
    return np.array(points)


from PIL import Image, ImageDraw, ImageFont
def add_text_to_image(image_np, text, font_path=None, font_size=20, text_color=(0, 0, 0)):
    """
    Adds text to the bottom of a numpy array image.

    Args:
        image_np (np.ndarray): Input image as a numpy array.
        text (str): The text to add at the bottom of the image.
        font_path (str, optional): Path to a .ttf font file. Defaults to None, which uses the default PIL font.
        font_size (int): Size of the font. Default is 20.
        text_color (tuple): RGB color for the text. Default is white (255, 255, 255).

    Returns:
        np.ndarray: The image with text added at the bottom.
    """
    image_pil = Image.fromarray(image_np)
    
    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default(size=font_size)
    
    draw = ImageDraw.Draw(image_pil)

    def textsize(text, font):
        # https://stackoverflow.com/a/77749307
        im = Image.new(mode="P", size=(0, 0))
        draw = ImageDraw.Draw(im)
        _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
        return width, height
    
    # Calculate text width and position
    text_width, text_height = textsize(text, font=font)
    image_width, image_height = image_pil.size
    text_x = (image_width - text_width) // 2
    text_y = image_height - text_height - 5  # Padding from the bottom
    draw.text((text_x, text_y), text, fill=text_color, font=font)
    return np.array(image_pil)