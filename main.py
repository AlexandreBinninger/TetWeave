import argparse
import omegaconf
from optimize import Optimizer
import torch
from util import print_gpu_memory

def main():
    parser = argparse.ArgumentParser(description='FlexiDelaunay optimization')
    parser.add_argument('-c', '--config', type=str, default="./assets/configs/default.yaml")
    parser.add_argument('-o', '--out_dir', type=str, default="out/default")
    parser.add_argument('-rm', '--ref_mesh', type=str, default="./assets/data/GOAT.obj")
    args = parser.parse_args()
    
    config = omegaconf.OmegaConf.load(args.config)
    device = 'cuda'
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    optimizer = Optimizer(config, args.ref_mesh, args.out_dir, device)
    optimizer.train()
    print_gpu_memory()
    
if __name__ == "__main__":
    main()