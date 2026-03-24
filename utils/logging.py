import logging
import torch.distributed as dist

def create_logger(logging_dir):
    """Create a logger that writes to a log file and stdout (rank 0 only)."""
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if rank == 0:
        handlers = [logging.StreamHandler()]
        if logging_dir is not None:
            handlers.append(logging.FileHandler(f"{logging_dir}/log.txt"))
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=handlers,
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger
