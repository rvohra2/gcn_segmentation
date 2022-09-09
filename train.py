# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)

import argparse

import os
import torch
# OSError: [Errno 24] Too many open files in multi processing
# https://github.com/pytorch/pytorch/issues/11201
import torch.multiprocessing

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.utils.comm import get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

torch.multiprocessing.set_sharing_strategy('file_system')


def train(cfg, local_rank, distributed):
    # model = build_detection_model(cfg)

    # device = torch.device(cfg.MODEL.DEVICE)
    # model.to(device)

    #optimizer = make_optimizer(cfg, model)
    #scheduler = make_lr_scheduler(cfg, optimizer)
    

    # if distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model, device_ids=[local_rank], output_device=local_rank,
    #         # this should be removed if we update BatchNorm stats
    #         broadcast_buffers=False,
    #     )

    arguments = {}
    arguments["iteration"] = 0
    # arguments["iteration"] = 10000

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    # checkpointer = DetectronCheckpointer(
    #     cfg, model, optimizer, scheduler, output_dir, save_to_disk
    # )
    # checkpointer.load()
    

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    
    
    do_train(
        data_loader,
        arguments,
    )
    print('MODEL PASS')
    
    
def main():
    parser = argparse.ArgumentParser(
        description="PyTorch Object Detection Training")
    parser.add_argument(
        "config",
        default="",
        metavar="FILE",
        help="path to configs file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    # parser.add_argument(
    #     "--skip-test",
    #     dest="skip_test",
    #     help="Do not test the final model",
    #     action="store_true",
    # )
    parser.add_argument(
        "opts",
        help="Modify configs options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(
        os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    # logger.info("Collecting env info (might take some time)")
    # logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config))
    with open(args.config, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with configs:\n{}".format(cfg))

    model = train(cfg, args.local_rank, args.distributed)
    # if not args.skip_test:
    #     test(cfg, model, args.distributed)


if __name__ == "__main__":
    main()
