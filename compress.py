import torch
from tetweave import TetWeave, get_edges, get_active_edges

@torch.no_grad()
def compress_model(input_path, output_path, device='cuda'):
    checkpoint = torch.load(input_path, map_location=device)
    sdf = checkpoint['sdf'].to(torch.float32)
    x_nx3 = checkpoint['x_nx3'].to(torch.float32)
    sh_deg = checkpoint['sh_deg']
    sh_c = checkpoint['sh_c']
    
    tw = TetWeave(device)
    tetrahedra = tw.tetrahedralize(x_nx3)
    edges = get_edges(tetrahedra).reshape(-1, 2)
    active_edges = get_active_edges(edges, sdf)
    active_points_idx, inverse_idx = torch.unique(active_edges.flatten(), return_counts=False, return_inverse=True)
    active_points = x_nx3[active_points_idx]
    active_sdf = sdf[active_points_idx]
    active_edges = inverse_idx.reshape(-1, 2)
    if sh_c is not None:
        active_sh_coefficients = sh_c[active_points_idx]
        active_sh_coefficients = active_sh_coefficients.to(torch.float16)
    else:
        active_sh_coefficients = torch.empty((0, 0), device=device)
    active_points = active_points.to(torch.float16)
    active_sdf = active_sdf.to(torch.float16)
    torch.save({'x_nx3': active_points, 'sdf': active_sdf, 'sh_deg': sh_deg, 'sh_c': active_sh_coefficients}, output_path)
    

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-mp", "--model_path", type=str, default="assets/examples/GOAT.pt")
    parser.add_argument("-op", "--output_path", type=str, default="assets/examples/GOAT_compressed.pt")
    args = parser.parse_args()
    model_path = args.model_path
    output_path = args.output_path
    compress_model(model_path, output_path)


if __name__ == "__main__":
    main()