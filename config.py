import os, argparse
import numpy as np

desc="Vectorization Using GCN"
parser = argparse.ArgumentParser(description=desc)

OUTPUT_DIR = "./"

args = parser.parse_args()
num_gpus = int(
        os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
args.distributed = num_gpus > 1


parser.add_argument("--norm-features", action="store_true", default=False)
parser.add_argument("--adj-dropout", type=float, default=0)
parser.add_argument("--feature-layer", type=str, nargs="+", default=["classifier", "classifier", 0], help="feature extraction layer for GCN")
parser.add_argument("--lr", type=float, default=0.01, help="learning rate for the GCN network")
parser.add_argument("--weight-decay", type=float, default=5e-4)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--hidden", type=int, default=16)
parser.add_argument("--nfeat", type=int, default=256)
parser.add_argument("--nhood", type=int, choices=[4,8], default=4, help="adjacency matrix neighborhood")
parser.add_argument("--no-loop", action="store_true", help="Adjacency matrix: don't use loops")

def dir_struct(args, make_dirs=True):
    args.config = os.path.join(args.output_path, args.config)
    args.model_path = os.path.join(args.output_path, "checkpoints")
    
    if args.log_path is None: 
        args.log_path = os.path.join(args.output_path, 'logs')
    if args.image_path is None: 
        args.image_path = os.path.join(args.output_path, 'images')
    
    if make_dirs:
        print(f"| creating {args.model_path} ...")
        os.makedirs(args.model_path, exist_ok=True)
        print(f"| creating {args.log_path} ...")
        os.makedirs(args.log_path, exist_ok=True)
        print(f"| creating {args.image_path} ...")
        os.makedirs(args.image_path, exist_ok=True)
    return args