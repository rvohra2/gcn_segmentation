from asyncio.log import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import torchvision
import errno
import os

import config as cfg

from gcn_model import GCNs

torch.autograd.set_detect_anomaly(True)

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def get_rank():
    if not torch.distributed.is_available():
        return 0
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


def train():
    

def main(args):

    device = f"cuda:{args.gpu}" if not args.no_cuda else "cpu"
    print(f"{args.rank}: devices={torch.cuda.device_count()}")
    print(f"{args.rank}: device={device} (force no cuda={args.no_cuda})")

    logger.print_config_info(args)
    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    print(f"| setup loggers:")
    

    model = train(cfg, args.local_rank, args.distributed)

if __name__=='__main__':
    from config import args
    main(args)
