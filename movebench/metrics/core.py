"""The six metrics, and the one function that scores one video pair.

Design rule for this whole repo: **every launcher calls the same
`score_one_video()`.** The bash-fork baseline, the Ray actor pool, and the
single-process reference differ *only* in orchestration. That is what makes the
parity check meaningful -- if two launchers disagree on a number, it is the
orchestration that broke, because the arithmetic is shared code.

The six metrics, and where each one runs
---------------------------------------
| metric | model                        | device it actually uses        |
|--------|------------------------------|--------------------------------|
| CLIP-I | CLIP ViT-L/14                | **GPU**                        |
| EPE    | RAFT-Large optical flow      | **GPU**                        |
| LPIPS  | VGG16 perceptual             | **CPU** (see note)             |
| SSIM   | 3D gaussian conv            | CPU or GPU (follows tensor)    |
| PSNR   | closed form                  | CPU or GPU (follows tensor)    |
| FVD    | I3D (video-level, over set)  | **GPU**                        |

The LPIPS note is the crux of this repo's CPU/GPU finding. In the original
production suite (`ray_learning/movebench/utils/lpips.py`) the LPIPS module is
constructed and never moved to the GPU, and the frames it is fed are CPU
tensors from `transforms.ToTensor()`. So the single most expensive per-frame
metric -- a VGG16 forward pass, per frame, per video -- runs entirely on the
host. That is why a pipeline everyone calls "the GPU eval" is in fact
CPU-bound, and why giving it more GPUs makes it slower once the fixed CPU budget
gets divided (README "The finding").

We preserve that placement by default (`device_policy="original"`) so the
baseline we compare against is the real production behaviour, not a strawman.
`device_policy="gpu"` moves LPIPS/SSIM/PSNR onto the GPU and is offered as the
*fix*, measured separately.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Lazily-built, per-process model cache.
#
# Models load once per process and are reused for every video. In the bash-fork
# launcher that means once per forked process; in the Ray pool, once per actor.
# Getting this wrong (loading per video) is what made one of the ViPE variants
# we benchmarked look 1.7x slower than it was -- see
# ../vipe_slow/README.md "the per-video make_pipeline() trap".
# ---------------------------------------------------------------------------
_CACHE: dict = {}

METRIC_NAMES = ("clip", "epe", "lpips", "ssim", "psnr")
VIDEO_LEVEL_METRICS = ("fvd",)
ALL_METRICS = METRIC_NAMES + VIDEO_LEVEL_METRICS


def _torch():
    import torch

    return torch


@dataclass
class ScoreConfig:
    """Everything that changes what gets computed (as opposed to how it is scheduled)."""

    metrics: tuple[str, ...] = METRIC_NAMES
    # "original" keeps production's placement (LPIPS/SSIM/PSNR on CPU);
    # "gpu" is the fix: everything on the accelerator.
    device_policy: str = "original"
    # Cap frames per video; None = all frames. Used to shrink smoke tests.
    max_frames: int | None = None
    # Batch the host->device copy instead of per-frame (EXP-009's lever).
    pinned_memory: bool = False

    def key(self) -> tuple:
        return (self.device_policy, tuple(sorted(self.metrics)), self.pinned_memory)


def load_models(cfg: ScoreConfig, device: str = "cuda") -> dict:
    """Build (or fetch) this process's models. Idempotent and thread-safe enough
    for our use: one call per worker before the work loop starts."""
    ck = ("models", cfg.key(), device)
    if ck in _CACHE:
        return _CACHE[ck]

    torch = _torch()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    on_gpu = cfg.device_policy == "gpu" and dev.type == "cuda"
    models: dict = {"device": dev, "cpu_metrics_on_gpu": on_gpu}

    if "clip" in cfg.metrics:
        from transformers import CLIPModel, CLIPProcessor

        name = "openai/clip-vit-large-patch14"
        models["clip_model"] = CLIPModel.from_pretrained(name).to(dev).eval()
        models["clip_proc"] = CLIPProcessor.from_pretrained(name)

    if "epe" in cfg.metrics:
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

        w = Raft_Large_Weights.DEFAULT
        models["raft"] = raft_large(weights=w, progress=False).to(dev).eval()
        models["raft_tf"] = w.transforms()

    if "lpips" in cfg.metrics:
        import lpips as lpips_lib

        net = lpips_lib.LPIPS(net="vgg", verbose=False)
        # The placement decision that drives this whole repo's finding.
        net = net.to(dev) if on_gpu else net.cpu()
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        models["lpips"] = net

    if "fvd" in cfg.metrics:
        models["i3d"] = _load_i3d(dev)

    _CACHE[ck] = models
    return models


def _load_i3d(dev):
    """I3D for FVD. Weights ship with the original suite; skip FVD if absent."""
    torch = _torch()
    from .i3d import InceptionI3d

    wpath = Path(__file__).parent / "weights" / "i3d_pretrained_400.pt"
    if not wpath.exists():
        return None
    net = InceptionI3d(400, in_channels=3).to(dev)
    net.load_state_dict(torch.load(str(wpath), map_location=dev))
    return net.eval()


# ---------------------------------------------------------------------------
# per-frame metrics
# ---------------------------------------------------------------------------
def _psnr(a, b):
    torch = _torch()
    return (-10 * torch.log10(((a - b) ** 2).mean())).item()


def _ssim(a, b, window_cache: dict):
    """3D-gaussian SSIM, matching the production implementation's shape handling."""
    torch = _torch()
    import torch.nn.functional as F
    from math import exp

    ws = 11
    ck = ("w3d", a.device, a.dtype)
    if ck not in window_cache:
        g = torch.tensor(
            [exp(-((x - ws // 2) ** 2) / (2 * 1.5**2)) for x in range(ws)],
            dtype=a.dtype,
        )
        g = (g / g.sum()).unsqueeze(1)
        w2 = g.mm(g.t())
        w3 = (w2.unsqueeze(2) @ g.t()).expand(1, 1, ws, ws, ws).contiguous()
        window_cache[ck] = w3.to(a.device)
    window = window_cache[ck]

    L = 1.0  # inputs are in [0,1]
    pad = (5,) * 6
    x, y = a.unsqueeze(1), b.unsqueeze(1)
    mu1 = F.conv3d(F.pad(x, pad, mode="replicate"), window, groups=1)
    mu2 = F.conv3d(F.pad(y, pad, mode="replicate"), window, groups=1)
    mu1s, mu2s, mu12 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    s1 = F.conv3d(F.pad(x * x, pad, mode="replicate"), window, groups=1) - mu1s
    s2 = F.conv3d(F.pad(y * y, pad, mode="replicate"), window, groups=1) - mu2s
    s12 = F.conv3d(F.pad(x * y, pad, mode="replicate"), window, groups=1) - mu12
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    v1, v2 = 2.0 * s12 + c2, s1 + s2 + c2
    return (((2 * mu12 + c1) * v1) / ((mu1s + mu2s + c1) * v2)).mean().item()


def _clip_i(models, img_a, img_b):
    torch = _torch()
    proc, model, dev = models["clip_proc"], models["clip_model"], models["device"]
    with torch.no_grad():
        ia = proc(images=img_a, return_tensors="pt").to(dev)
        ib = proc(images=img_b, return_tensors="pt").to(dev)
        fa = model.get_image_features(**ia)
        fb = model.get_image_features(**ib)
        fa = fa / fa.norm(dim=-1, keepdim=True)
        fb = fb / fb.norm(dim=-1, keepdim=True)
        return torch.matmul(fa, fb.T).item()


def _lpips(models, a, b):
    """LPIPS on [-1,1] inputs. Runs where the module lives (see module docstring)."""
    torch = _torch()
    net = models["lpips"]
    target = next(net.parameters()).device
    with torch.no_grad():
        return net(a.to(target).sub(0.5).div(0.5), b.to(target).sub(0.5).div(0.5)).item()


def _epe(models, raw_video, gen_video):
    """Endpoint error between the two videos' optical-flow fields.

    Measures whether the generated video *moves* like the reference, which is
    the metric that actually matters for camera/motion control -- and is why
    this suite exists rather than just PSNR.
    """
    torch = _torch()
    import torchvision.transforms.functional as TF

    model, tf, dev = models["raft"], models["raft_tf"], models["device"]

    def prep(v):
        src = TF.resize(v[:-1], size=[480, 832], antialias=False)
        dst = TF.resize(v[1:], size=[480, 832], antialias=False)
        return tf(src, dst)

    with torch.no_grad():
        a_s, a_d = prep(raw_video)
        b_s, b_d = prep(gen_video)
        fa = model(a_s.to(dev).contiguous(), a_d.to(dev).contiguous())[-1]
        fb = model(b_s.to(dev).contiguous(), b_d.to(dev).contiguous())[-1]
        return torch.norm(fa - fb, p=2, dim=1).mean().item()


def _fvd_logits(models, video_uint8):
    """I3D embedding for one video, (T,H,W,C) uint8 -> (1,400)."""
    torch = _torch()
    import torch.nn.functional as F

    i3d, dev = models.get("i3d"), models["device"]
    if i3d is None:
        return None
    with torch.no_grad():
        v = torch.from_numpy(video_uint8).float()  # T,H,W,C
        v = v.permute(0, 3, 1, 2).contiguous()  # T,C,H,W
        v = F.interpolate(v, size=(224, 224), mode="bilinear", align_corners=False)
        v = v.permute(1, 0, 2, 3).unsqueeze(0)  # 1,C,T,H,W
        v = 2.0 * v / 255.0 - 1.0
        return i3d(v.to(dev)).cpu().numpy()


# ---------------------------------------------------------------------------
# the unit of work
# ---------------------------------------------------------------------------
def read_video_frames(path: Path, max_frames: int | None = None) -> np.ndarray:
    """Decode to (T,H,W,C) uint8 RGB."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {path}")
    out = []
    while max_frames is None or len(out) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if not out:
        raise ValueError(f"decoded 0 frames from {path}")
    return np.stack(out)


def score_one_video(
    name: str,
    raw_path: Path,
    gen_path: Path,
    models: dict,
    cfg: ScoreConfig,
) -> dict:
    """Score ONE video pair. The single unit of work every launcher schedules.

    Returns per-metric means over frames, plus the I3D embeddings FVD needs
    (FVD is a set-level statistic, so it cannot be finished per video -- the
    embeddings are reduced across the whole dataset at merge time).
    """
    torch = _torch()
    from PIL import Image

    raw = read_video_frames(raw_path, cfg.max_frames)
    gen = read_video_frames(gen_path, cfg.max_frames)
    t = min(len(raw), len(gen))
    raw, gen = raw[:t], gen[:t]

    per_frame: dict[str, list[float]] = {m: [] for m in cfg.metrics if m in METRIC_NAMES}
    window_cache: dict = {}

    # One host->device staging decision for the whole video (EXP-009: pinning is
    # the real lever, 2.6x on the transfer, not batching, 1.15x).
    def to_tensor(arr: np.ndarray):
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div(255.0)
        return x.pin_memory() if cfg.pinned_memory else x

    raw_v = to_tensor(raw)
    gen_v = to_tensor(gen)

    if models.get("cpu_metrics_on_gpu"):
        dev = models["device"]
        raw_v = raw_v.to(dev, non_blocking=cfg.pinned_memory)
        gen_v = gen_v.to(dev, non_blocking=cfg.pinned_memory)

    with torch.no_grad():
        for i in range(t):
            a, b = raw_v[i : i + 1], gen_v[i : i + 1]
            if "psnr" in per_frame:
                per_frame["psnr"].append(_psnr(a, b))
            if "ssim" in per_frame:
                per_frame["ssim"].append(_ssim(a, b, window_cache))
            if "lpips" in per_frame:
                per_frame["lpips"].append(_lpips(models, a, b))
            if "clip" in per_frame:
                per_frame["clip"].append(
                    _clip_i(models, Image.fromarray(raw[i]), Image.fromarray(gen[i]))
                )

        if "epe" in cfg.metrics:
            per_frame["epe"] = [_epe(models, raw_v.cpu(), gen_v.cpu())]

    result = {
        "name": name,
        "frames": t,
        "scores": {k: float(np.mean(v)) for k, v in per_frame.items() if v},
    }

    if "fvd" in cfg.metrics:
        ra, ga = _fvd_logits(models, raw), _fvd_logits(models, gen)
        if ra is not None:
            result["fvd_logits"] = {"raw": ra.tolist(), "gen": ga.tolist()}

    return result


def list_pairs(dataset: Path) -> list[tuple[str, Path, Path]]:
    """Discover (name, raw, gen) triples in a dataset built by make_dataset.py."""
    raw_dir, gen_dir = dataset / "raw", dataset / "gen"
    if not raw_dir.is_dir() or not gen_dir.is_dir():
        raise SystemExit(
            f"{dataset} is not a dataset root (expected raw/ and gen/ subdirs).\n"
            f"Build one with: python common/make_dataset.py --source <video> --out {dataset}"
        )
    pairs = []
    for r in sorted(raw_dir.glob("*.mp4")):
        g = gen_dir / r.name
        if g.exists():
            pairs.append((r.stem, r, g))
    if not pairs:
        raise SystemExit(f"no matching raw/gen .mp4 pairs under {dataset}")
    return pairs
