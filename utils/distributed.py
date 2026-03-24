import os
from typing import Tuple
import torch
import torch.distributed as dist

def setup_distributed() -> Tuple[int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        from datetime import timedelta
        dist.init_process_group(backend="nccl", timeout = timedelta(hours=2))
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world_size, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True