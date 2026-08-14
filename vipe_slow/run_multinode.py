#!/usr/bin/env python3
import os
import sys
import argparse
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import hydra
from vipe import get_config_path, make_pipeline
from vipe.streams.raw_mp4_stream import RawMp4Stream
from vipe.streams.base import ProcessedVideoStream
from vipe.streams.base import StreamList


def quiet_logs():
    logging.getLogger().setLevel(logging.ERROR)
    for name in ["vipe", "hydra", "omegaconf"]:
        logging.getLogger(name).setLevel(logging.ERROR)


def build_pipeline(pipeline_name: str, out_root: Path, skip_exists: bool, save_viz: bool):
    """
    Build ViPE pipeline ONCE, like ViPE CLI / run.py do (compose config -> make_pipeline). [1](https://github.com/nv-tlabs/vipe/issues/53)[2](https://github.com/nv-tlabs/vipe/blob/main/run.py)
    We set output.path to a root directory; ViPE will save per-stream results under it.
    """
    overrides = [
        f"pipeline={pipeline_name}",
        f"pipeline.output.path={str(out_root)}",
        f"pipeline.output.skip_exists={'true' if skip_exists else 'false'}",
        f"pipeline.output.save_viz={'true' if save_viz else 'false'}",
        "pipeline.output.save_artifacts=true",
    ]

    # Initialize Hydra once and compose config
    with hydra.initialize_config_dir(config_dir=str(get_config_path()), version_base=None):
        cfg = hydra.compose("default", overrides=overrides)

    return make_pipeline(cfg.pipeline)


def main():
    ap = argparse.ArgumentParser("ViPE 1-node sharded inference (CSV -> 1/8 per GPU)")
    ap.add_argument("--metadata", required=True, help="CSV file with 'sequence' column")
    ap.add_argument("--rgb_dir", required=True, help="Directory with <sequence>.mp4")
    ap.add_argument("--out_root", required=True, help="Root output directory")
    ap.add_argument("--pipeline", default="default", help="ViPE pipeline config name")
    ap.add_argument("--rank", type=int, required=True, help="Shard rank (0..world_size-1)")
    ap.add_argument("--world_size", type=int, default=8, help="Total shards (default 8)")
    ap.add_argument("--min_frames", type=int, default=0, help="Optional: filter num_frames >= this")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle CSV deterministically")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_exists", action="store_true", help="Skip if outputs already exist")
    ap.add_argument("--save_viz", action="store_true", help="Enable visualization video (default off)")
    ap.add_argument("--shard_num", type=int, default=1)
    ap.add_argument("--total_shard", type=int, default=1)
    args = ap.parse_args()

    quiet_logs()

    meta = Path(args.metadata)
    rgb_dir = Path(args.rgb_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if not meta.exists():
        raise FileNotFoundError(meta)
    if not rgb_dir.exists():
        raise FileNotFoundError(rgb_dir)

    df = pd.read_csv(meta)
    if "sequence" not in df.columns:
        raise ValueError(f"CSV missing 'sequence' column. columns={list(df.columns)}")

    chunk_size = len(df) // args.total_shard
    start_idx = (args.shard_num - 1) * chunk_size
    end_idx = start_idx + chunk_size if args.shard_num < args.total_shard else len(df)
    df = df.iloc[start_idx:end_idx]

    if args.min_frames > 0 and "num_frames" in df.columns:
        df["num_frames"] = pd.to_numeric(df["num_frames"], errors="coerce")
        df = df[df["num_frames"] >= args.min_frames].copy()

    if args.shuffle:
        df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    sequences = df["sequence"].astype(str).tolist()

    # Take 1/world_size of dataframe by index
    my_seqs = [s for i, s in enumerate(sequences) if (i % args.world_size) == args.rank]

    print(f"[rank {args.rank}] total={len(sequences)} my_shard={len(my_seqs)}", flush=True)

    pipeline = build_pipeline(args.pipeline, out_root, skip_exists=args.skip_exists, save_viz=args.save_viz)

    pbar = tqdm(my_seqs, desc=f"gpu-rank{args.rank}", dynamic_ncols=True)
    done = missing = failed = 0

    for seq in pbar:
        try:
            video_path = rgb_dir / f"{seq}"
            if not video_path.exists():
                missing += 1
                continue

            video_stream = ProcessedVideoStream(RawMp4Stream(video_path), []).cache(desc="Reading video stream")
            #video_stream = StreamList.make(video_path)[0]
            pipeline.run(video_stream)

            done += 1
        except Exception as e:
            failed += 1
            print(f"[rank {args.rank}] FAILED {seq}: {e}", file=sys.stderr, flush=True)

        pbar.set_postfix(done=done, missing=missing, failed=failed)

    print(f"[rank {args.rank}] FINISHED done={done} missing={missing} failed={failed}", flush=True)


if __name__ == "__main__":
    main()