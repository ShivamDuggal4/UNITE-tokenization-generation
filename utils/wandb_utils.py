import argparse
import hashlib
import math
import os

import torch
import wandb
from einops import rearrange
from torchvision.utils import make_grid
from utils.distributed import is_main_process



def _wandb_ready():
    return getattr(wandb, "run", None) is not None


def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }


def generate_run_id(exp_name):
    return str(int(hashlib.sha256(exp_name.encode("utf-8")).hexdigest(), 16) % 10**8)


def initialize(args, entity, exp_name, project_name):
    config_dict = namespace_to_dict(args)
    wandb.login(key=os.environ["WANDB_KEY"])
    wandb.init(
        entity=entity,
        project=project_name,
        name=exp_name,
        config=config_dict,
        id=generate_run_id(exp_name),
        resume="allow",
    )


def log(stats, step=None):
    if is_main_process() and _wandb_ready():
        wandb.log({k: v for k, v in stats.items()}, step=step)


def log_image(sample, step=None):
    if is_main_process() and _wandb_ready():
        sample = array2grid(sample)
        wandb.log({"samples": wandb.Image(sample), "train_step": step})


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x, nrow=nrow, normalize=True, value_range=(0, 1))
    x = x.clamp(0, 1).mul(255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return x


def log_visuals(input_samples, output_samples, global_step, wandb_log_key, num_images_to_log=16):
    if not _wandb_ready():
        return
    input_imgs = input_samples[:num_images_to_log]
    reconstructed_imgs = output_samples[:num_images_to_log].clamp(0.0, 1.0)
    vis_img = torch.cat([input_imgs.detach().cpu(), reconstructed_imgs.detach().cpu()], dim=0)
    vis_img = rearrange(vis_img, "(v h1 w1) c h w -> c (h1 h) (w1 v w)", w1=2, v=2)
    wandb_images = [wandb.Image(vis_img, caption="input; prediction;")]
    wandb.log({wandb_log_key: wandb_images}, step=global_step)


def log_visuals_triplet(
    input_samples,
    output_samples,
    output_samples_1,
    global_step,
    wandb_log_key,
    num_images_to_log=16,
    nrow=4,
    padding=1,
):
    if not _wandb_ready():
        return
    x = input_samples[:num_images_to_log].detach().clamp(0.0, 1.0).cpu()
    y0 = output_samples[:num_images_to_log].detach().clamp(0.0, 1.0).cpu()
    y1 = output_samples_1[:num_images_to_log].detach().clamp(0.0, 1.0).cpu()
    triplets = [torch.cat([x[i], y0[i], y1[i]], dim=2) for i in range(x.size(0))]
    triplets = torch.stack(triplets, dim=0)
    grid = make_grid(triplets, nrow=nrow, padding=padding)
    wandb_img = wandb.Image(grid, caption="input | decoded(x_pred) | decoded(x_t)")
    wandb.log({wandb_log_key: [wandb_img]}, step=global_step)
