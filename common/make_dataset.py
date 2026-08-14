"""Build a reproducible video-evaluation dataset from one raw video.

Why synthesize instead of shipping a dataset
--------------------------------------------
A video-eval benchmark needs *pairs*: a reference ("raw") video and a
"generated" video to score against it. Real generated videos are large, and the
originals used for this project's numbers live on a scratch mount that is not
guaranteed to exist in your checkout. So this script builds the dataset from a
single source clip, deterministically:

    raw/<stem>_<i>.mp4    N copies of the source, standardized to 81 frames
    gen/<stem>_<i>.mp4    the same clip put through a controlled degradation

The degradation is what makes the metrics non-trivial. Each variant `i` gets a
different, seeded degradation strength, so CLIP/LPIPS/SSIM/PSNR/EPE all return a
*spread* of values rather than 1.0/0.0 -- which means a parity check between two
launchers is actually testing something.

Why 81 frames
-------------
81 = 16*5 + 1, the frame count modern latent video diffusion models emit (a
temporal-VAE stride of 4 over 20 latent frames, plus the anchor frame). Fixing
it makes every sample cost the same, which matters for two separate reasons:

  1. ViPE SLAM cost scales with frame count, so equal-length videos make
     wall-clock comparisons across GPU counts clean.
  2. It removes duration variance -- so any speed difference we measure between
     a static shard and a dynamic work queue is *not* load balancing. (When we
     want to test load balancing, `--jitter-frames` deliberately puts the
     variance back. See README "Two datasets, on purpose".)

Usage
-----
    python make_dataset.py --source vipe_slow/test_video/recam1.mp4 \
        --out data/eval81 --num 32

    # a duration-variance version, to exercise dynamic scheduling
    python make_dataset.py --source ... --out data/eval81_jitter --num 32 \
        --jitter-frames 25 81
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

STANDARD_FRAMES = 81


# --------------------------------------------------------------------------
# degradations -- deterministic given (variant index, frame index)
# --------------------------------------------------------------------------
def degrade(frame: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    """Apply a blend of realistic generative-video artifacts.

    Chosen so that different metrics disagree about the damage, which is the
    point: PSNR/SSIM see the blur and noise, LPIPS sees the texture loss, CLIP
    sees almost nothing (semantics survive), EPE sees the temporal jitter.
    """
    out = frame.astype(np.float32)

    # 1. Gaussian blur -- loss of high-frequency detail (hits SSIM, LPIPS).
    if strength > 0:
        k = int(1 + 2 * round(strength * 4))  # odd kernel, 1..9
        if k >= 3:
            out = cv2.GaussianBlur(out, (k, k), sigmaX=strength * 2.0)

    # 2. Additive noise -- hits PSNR hardest.
    if strength > 0:
        out = out + rng.normal(0.0, strength * 12.0, size=out.shape)

    # 3. Slight colour/contrast drift -- what a mistuned VAE decoder does.
    out = out * (1.0 - 0.10 * strength) + 128.0 * (0.10 * strength)

    # 4. Sub-pixel spatial shift -- creates optical-flow error (EPE).
    if strength > 0:
        dx = float(rng.normal(0.0, strength * 1.5))
        dy = float(rng.normal(0.0, strength * 1.5))
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        out = cv2.warpAffine(
            out, m, (out.shape[1], out.shape[0]), borderMode=cv2.BORDER_REFLECT
        )

    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
def read_frames(path: Path, limit: int | None = None) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open source video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    frames = []
    while limit is None or len(frames) < limit:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise SystemExit(f"source video decoded 0 frames: {path}")
    return frames, fps


def standardize(frames: list[np.ndarray], n: int) -> list[np.ndarray]:
    """Make exactly `n` frames: loop (ping-pong) if short, truncate if long.

    Ping-pong rather than restart-from-0 so the motion stays continuous -- a hard
    cut would inject a huge fake optical-flow spike and make EPE meaningless.
    """
    if len(frames) >= n:
        return frames[:n]
    out = list(frames)
    forward = False
    while len(out) < n:
        seq = frames if forward else frames[::-1]
        out.extend(seq[1:])  # skip the duplicate boundary frame
        forward = not forward
    return out[:n]


def write_video(
    path: Path, frames: list[np.ndarray], fps: float, resize: tuple[int, int] | None
) -> None:
    h, w = frames[0].shape[:2]
    if resize:
        w, h = resize
    path.parent.mkdir(parents=True, exist_ok=True)
    # mp4v is available in every opencv wheel; avc1 often is not.
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise SystemExit(f"cannot open VideoWriter for {path}")
    for f in frames:
        if resize:
            f = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(f)
    writer.release()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build an 81-frame raw/generated video-eval dataset from one clip.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--source", required=True, type=Path, help="a single source .mp4")
    ap.add_argument("--out", required=True, type=Path, help="output dataset root")
    ap.add_argument("--num", type=int, default=32, help="number of video pairs")
    ap.add_argument(
        "--frames", type=int, default=STANDARD_FRAMES, help="frames per video"
    )
    ap.add_argument(
        "--resolution",
        default="832x480",
        help="WxH output resolution, or 'source' to keep the input size",
    )
    ap.add_argument(
        "--jitter-frames",
        nargs=2,
        type=int,
        metavar=("MIN", "MAX"),
        help="vary frame count uniformly in [MIN,MAX] instead of fixing it "
        "(use to create duration variance for load-balancing tests)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--min-strength", type=float, default=0.15, help="weakest degradation"
    )
    ap.add_argument(
        "--max-strength", type=float, default=0.85, help="strongest degradation"
    )
    ap.add_argument("--force", action="store_true", help="overwrite an existing dataset")
    args = ap.parse_args()

    resize = None
    if args.resolution != "source":
        try:
            w, h = (int(x) for x in args.resolution.lower().split("x"))
            resize = (w, h)
        except ValueError:
            raise SystemExit(f"bad --resolution: {args.resolution!r} (want WxH)")

    out = args.out
    if out.exists():
        if not args.force:
            raise SystemExit(f"{out} exists; pass --force to overwrite")
        shutil.rmtree(out)
    raw_dir, gen_dir = out / "raw", out / "gen"
    raw_dir.mkdir(parents=True)
    gen_dir.mkdir(parents=True)

    src_frames, fps = read_frames(args.source)
    print(f"source: {args.source}  decoded {len(src_frames)} frames @ {fps:g} fps")

    root_rng = np.random.default_rng(args.seed)
    stem = args.source.stem
    manifest = []

    for i in range(args.num):
        if args.jitter_frames:
            lo, hi = args.jitter_frames
            n_frames = int(root_rng.integers(lo, hi + 1))
        else:
            n_frames = args.frames

        # Per-variant seed => reproducible, and independent of iteration order.
        rng = np.random.default_rng([args.seed, i])
        strength = args.min_strength + (args.max_strength - args.min_strength) * (
            i / max(1, args.num - 1)
        )

        base = standardize(src_frames, n_frames)
        name = f"{stem}_{i:04d}.mp4"

        # Resize BEFORE degrading: the blur/warp cost scales with pixel count, and
        # degrading then downscaling would partly average the artifacts away. Doing
        # it in this order is both ~6x cheaper and a truer model of a generator that
        # emits at the target resolution.
        if resize:
            base = [
                cv2.resize(f, resize, interpolation=cv2.INTER_AREA) for f in base
            ]

        write_video(raw_dir / name, base, fps, None)
        write_video(gen_dir / name, [degrade(f, strength, rng) for f in base], fps, None)

        manifest.append(
            {"name": name, "frames": n_frames, "degrade_strength": round(strength, 4)}
        )
        print(
            f"  [{i + 1:>3}/{args.num}] {name}  frames={n_frames}  "
            f"strength={strength:.3f}"
        )

    meta = {
        "source": str(args.source.resolve()),
        "num_pairs": args.num,
        "frames": "jittered" if args.jitter_frames else args.frames,
        "jitter_range": args.jitter_frames,
        "resolution": args.resolution,
        "fps": fps,
        "seed": args.seed,
        "strength_range": [args.min_strength, args.max_strength],
        "videos": manifest,
    }
    (out / "manifest.json").write_text(json.dumps(meta, indent=2))

    total_frames = sum(v["frames"] for v in manifest)
    print(
        f"\nwrote {args.num} pairs ({total_frames} frames each side) to {out}\n"
        f"  raw/  {raw_dir}\n  gen/  {gen_dir}\n  manifest.json"
    )


if __name__ == "__main__":
    sys.exit(main())
