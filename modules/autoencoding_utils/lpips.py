"""Learned Perceptual Image Patch Similarity (LPIPS) with VGG16 backbone."""

from __future__ import annotations

import os
from collections import namedtuple
from urllib.parse import urlparse
import uuid
import hashlib
import pathlib
from contextlib import contextmanager
from typing import Iterator, Optional
import requests
from tqdm import tqdm

import torch
import torch.hub
import torch.nn as nn
from torchvision import models

URL_MAP = {"vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"}
CKPT_MAP = {"vgg_lpips": "vgg.pth"}
MD5_MAP = {"vgg_lpips": "d507d7349b931f0638a25a48a722f98a"}


def dist_rank(default: int = 0) -> int:
    value = os.environ.get("RANK")
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def local_rank(default: int = 0) -> int:
    value = os.environ.get("LOCAL_RANK")
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def should_log() -> bool:
    return dist_rank(default=0) == 0


@contextmanager
def file_lock(lock_path: str) -> Iterator[None]:
    try:
        import fcntl
    except Exception:
        yield
        return

    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def md5_hash(path: str, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download(url: str, local_path: str, chunk_size: int = 1024 * 1024) -> None:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    tmp_path = f"{local_path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length") or 0)
        with tqdm(total=total_size, unit="B", unit_scale=True,
                  disable=not (should_log() and total_size > 0)) as pbar:
            with open(tmp_path, "wb") as f:
                for data in response.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(len(data))
    os.replace(tmp_path, local_path)


def get_ckpt_path(name: str, root: Optional[str] = None, check: bool = False) -> str:
    assert name in URL_MAP, f"Unknown checkpoint: {name}"
    if root is None:
        root = os.environ.get("LPIPS_CACHE_DIR")
    if root is None:
        root = os.path.join(str(pathlib.Path(__file__).parent.absolute()), ".caches")

    path = os.path.join(root, CKPT_MAP[name])
    expected_md5 = MD5_MAP.get(name)

    def is_valid() -> bool:
        if not os.path.exists(path):
            return False
        if not check or expected_md5 is None:
            return True
        return md5_hash(path) == expected_md5

    if is_valid():
        return path

    lock_path = f"{path}.lock"
    with file_lock(lock_path):
        if is_valid():
            return path

        if should_log():
            print(f"Downloading {name} model from {URL_MAP[name]} to {path}")

        tmp_path: Optional[str] = None
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
            with requests.get(URL_MAP[name], stream=True, timeout=60) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length") or 0)
                with tqdm(total=total, unit="B", unit_scale=True,
                          disable=not (should_log() and total > 0)) as pbar:
                    with open(tmp, "wb") as f:
                        for data in response.iter_content(chunk_size=1024 * 1024):
                            if data:
                                f.write(data)
                                pbar.update(len(data))
            tmp_path = tmp
            if expected_md5 is not None:
                actual = md5_hash(tmp_path)
                if actual != expected_md5:
                    raise RuntimeError(
                        f"LPIPS checkpoint MD5 mismatch: expected {expected_md5}, got {actual}"
                    )
            os.replace(tmp_path, path)
        finally:
            if tmp_path is not None and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    return path




def _ensure_torchvision_weights_cached(url: str) -> None:
    """Download VGG16 weights with file-locking for multi-process safety."""
    filename = os.path.basename(urlparse(url).path)
    hub_dir = torch.hub.get_dir()
    checkpoints_dir = os.path.join(hub_dir, "checkpoints")
    dst_path = os.path.join(checkpoints_dir, filename)
    lock_path = f"{dst_path}.lock"

    with file_lock(lock_path):
        if os.path.exists(dst_path):
            return
        torch.hub.load_state_dict_from_url(
            url,
            model_dir=checkpoints_dir,
            file_name=filename,
            check_hash=True,
            progress=local_rank() == 0,
        )


class ScalingLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("shift", torch.tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer("scale", torch.tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.shift) / self.scale


class NetLinLayer(nn.Module):
    def __init__(self, chn_in: int, chn_out: int = 1, use_dropout: bool = False) -> None:
        super().__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers.append(nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False))
        self.model = nn.Sequential(*layers)


class VGG16FeatureExtractor(nn.Module):
    def __init__(self, requires_grad: bool = False, pretrained: bool = True) -> None:
        super().__init__()
        weights = None
        if pretrained:
            weights = models.VGG16_Weights.DEFAULT
            _ensure_torchvision_weights_cached(weights.url)
        features = models.vgg16(weights=weights, progress=local_rank() == 0).features
        self.slice1 = nn.Sequential(*[features[x] for x in range(4)])
        self.slice2 = nn.Sequential(*[features[x] for x in range(4, 9)])
        self.slice3 = nn.Sequential(*[features[x] for x in range(9, 16)])
        self.slice4 = nn.Sequential(*[features[x] for x in range(16, 23)])
        self.slice5 = nn.Sequential(*[features[x] for x in range(23, 30)])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, tensor: torch.Tensor):
        h = self.slice1(tensor)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        outputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"])
        return outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)


def _normalize(tensor: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    norm_factor = torch.sqrt(torch.sum(tensor ** 2, dim=1, keepdim=True))
    return tensor / (norm_factor + eps)


def _spatial_average(tensor: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return tensor.mean([2, 3], keepdim=keepdim)


class LPIPS(nn.Module):
    """Learned perceptual metric (VGG-based) used by VQGAN.

    All parameters are frozen after loading pretrained weights.
    """

    def __init__(self, use_dropout: bool = True) -> None:
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]
        self.net = VGG16FeatureExtractor(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self._load_pretrained_weights()
        for param in self.parameters():
            param.requires_grad = False

    def _load_pretrained_weights(self, name: str = "vgg_lpips") -> None:
        ckpt = get_ckpt_path(name)
        try:
            state = torch.load(ckpt, map_location=torch.device("cpu"))
        except Exception:
            ckpt = get_ckpt_path(name, check=True)
            state = torch.load(ckpt, map_location=torch.device("cpu"))
        self.load_state_dict(state, strict=False)
        if dist_rank() == 0:
            print(f"[LPIPS] Loaded pretrained weights from {ckpt}")

    def forward(self, input: torch.Tensor, target: torch.Tensor,
                reduction: str = "mean") -> torch.Tensor:
        in_scaled = self.scaling_layer(input)
        tgt_scaled = self.scaling_layer(target)
        feats_in = self.net(in_scaled)
        feats_tgt = self.net(tgt_scaled)

        lin_layers = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        value = torch.zeros(1, device=input.device)
        for feat_in, feat_tgt, lin in zip(feats_in, feats_tgt, lin_layers):
            diff = (_normalize(feat_in) - _normalize(feat_tgt)) ** 2
            value = value + _spatial_average(lin.model(diff), keepdim=True)

        if reduction == "none":
            return value
        if reduction == "sum":
            return torch.sum(value)
        if reduction == "mean":
            return torch.mean(value)
        raise ValueError(f"Unsupported reduction '{reduction}'")
