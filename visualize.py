import trimesh
import torch
from tetweave import TetWeave, get_edges
import polyscope as ps

@torch.no_grad()
def decode(input_path, device='cuda'):
    checkpoint = torch.load(input_path, map_location=device)

    x_nx3 = checkpoint['x_nx3'].to(torch.float32)
    sdf = checkpoint['sdf'].to(torch.float32)
    sh_deg = checkpoint['sh_deg']
    sh_c = checkpoint['sh_c'].to(torch.float32)
    
    tw = TetWeave(device)

    vertices, faces, tetrahedra = tw(x_nx3, sdf, sh_coefficients=sh_c, sh_deg=sh_deg)
    edges = get_edges(tetrahedra).reshape(-1, 2)
    
    output_path = input_path.replace(".pt", ".obj")
    mesh = trimesh.Trimesh(vertices = vertices.detach().cpu().numpy(), faces=faces.detach().cpu().numpy(), process=False)
    mesh.export(output_path)
    print(f"Saved mesh to {output_path}")

    ps.init()
    ps.register_surface_mesh("Result mesh", vertices.detach().cpu().numpy(), faces.detach().cpu().numpy(), enabled=True, edge_width=1.0)
    ps.register_point_cloud("Points", x_nx3.detach().cpu().numpy(), radius=0.002)
    ps.register_curve_network("Edges", x_nx3.detach().cpu().numpy(), edges.detach().cpu().numpy(), radius=0.0003)
    ps.register_volume_mesh("Delaunay Triangulation", x_nx3.detach().cpu().numpy(), tetrahedra.detach().cpu().numpy(), enabled=False)
    ps.show()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-mp", "--model_path", type=str, default="./assets/examples/crab_compressed.pt")
    args = parser.parse_args()
    model_path = args.model_path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    decode(model_path, device)

if __name__ == "__main__":
    main()