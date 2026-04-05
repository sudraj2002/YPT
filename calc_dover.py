import os
import argparse
import yaml
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

from dover.datasets import ViewDecompositionDataset
from dover.models import DOVER


def fuse_results(results: list):
    # results[0]: aesthetic, results[1]: technical
    t = (results[1] - 0.1107) / 0.07355
    a = (results[0] + 0.08285) / 0.03774
    x = t * 0.6104 + a * 0.3896
    return {
        "aesthetic": 1 / (1 + np.exp(-a)),
        "technical": 1 / (1 + np.exp(-t)),
        "overall": 1 / (1 + np.exp(-x)),
    }


def has_any_mp4(folder: Path) -> bool:
    # fast check (non-recursive); switch to rglob if you need recursion
    return any(folder.glob("*.mp4"))


@torch.no_grad()
def run_folder(
    evaluator,
    opt,
    input_video_dir: str,
    device: str,
    num_workers: int,
):
    """
    Runs DOVER on all mp4s inside input_video_dir (dataset folder).
    Returns:
      per_video_rows: list of dicts
      summary: dict with mean scores + counts
    """
    dopt = opt["data"]["val-l1080p"]["args"].copy()
    dopt["anno_file"] = None
    dopt["data_prefix"] = input_video_dir  # folder containing mp4s

    dataset = ViewDecompositionDataset(dopt)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=True,
    )

    sample_types = ["aesthetic", "technical"]

    per_video = []
    failed = 0

    for _, data in enumerate(tqdm(dataloader, desc=f"  DOVER on {Path(input_video_dir).name}", leave=False)):
        # their dataset sometimes returns {"name": ...} only when decode fails
        if not isinstance(data, dict) or len(data.keys()) == 1:
            failed += 1
            continue

        video = {}
        for key in sample_types:
            if key in data:
                x = data[key].to(device)  # (B,C,T,H,W)
                b, c, t, h, w = x.shape
                nclips = int(data["num_clips"][key])
                x = (
                    x.reshape(b, c, nclips, t // nclips, h, w)
                     .permute(0, 2, 1, 3, 4, 5)
                     .reshape(b * nclips, c, t // nclips, h, w)
                )
                video[key] = x

        results = evaluator(video, reduce_scores=False)
        results = [float(np.mean(l.detach().cpu().numpy())) for l in results]
        rescaled = fuse_results(results)

        # name can be full path; keep both
        name = data["name"][0]
        per_video.append({
            "path": name,
            "aesthetic": rescaled["aesthetic"] * 100.0,
            "technical": rescaled["technical"] * 100.0,
            "overall": rescaled["overall"] * 100.0,
        })

    if len(per_video) == 0:
        summary = {
            "count": 0,
            "failed": failed,
            "mean_aesthetic": float("nan"),
            "mean_technical": float("nan"),
            "mean_overall": float("nan"),
        }
    else:
        A = np.array([r["aesthetic"] for r in per_video], dtype=np.float64)
        T = np.array([r["technical"] for r in per_video], dtype=np.float64)
        O = np.array([r["overall"] for r in per_video], dtype=np.float64)
        summary = {
            "count": int(len(per_video)),
            "failed": int(failed),
            "mean_aesthetic": float(A.mean()),
            "mean_technical": float(T.mean()),
            "mean_overall": float(O.mean()),
        }

    return per_video, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, default="./dover.yml", help="dover option yaml")
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Root directory that contains method subfolders")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=8)

    # outputs
    parser.add_argument("--out_per_video_csv", type=str, default="dover_per_video.csv")
    parser.add_argument("--out_summary_csv", type=str, default="dover_summary.csv")

    # optional filters
    parser.add_argument("--methods", type=str, default="",
                        help="Comma-separated list of method folder names to run. Empty = all")
    parser.add_argument("--datasets", type=str, default="",
                        help="Comma-separated list of dataset folder names to run. Empty = all")

    args = parser.parse_args()

    root = Path(args.root_dir)
    assert root.is_dir(), f"Not a directory: {root}"

    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)

    # Load DOVER once
    evaluator = DOVER(**opt["model"]["args"]).to(args.device)
    evaluator.load_state_dict(torch.load(opt["test_load_path"], map_location=args.device))
    evaluator.eval()

    method_filter = set([m.strip() for m in args.methods.split(",") if m.strip()]) if args.methods else None
    dataset_filter = set([d.strip() for d in args.datasets.split(",") if d.strip()]) if args.datasets else None

    # write headers
    with open(args.out_per_video_csv, "w") as f:
        f.write("method,dataset,path,aesthetic,technical,overall\n")
    with open(args.out_summary_csv, "w") as f:
        f.write("method,dataset,count,failed,mean_aesthetic,mean_technical,mean_overall\n")

    # enumerate methods
    method_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if method_filter is not None:
        method_dirs = [p for p in method_dirs if p.name in method_filter]

    for mdir in tqdm(method_dirs, desc="Methods"):
        # enumerate datasets inside method
        dataset_dirs = sorted([p for p in mdir.iterdir() if p.is_dir()])
        if dataset_filter is not None:
            dataset_dirs = [p for p in dataset_dirs if p.name in dataset_filter]

        for ddir in tqdm(dataset_dirs, desc=f"Datasets in {mdir.name}", leave=False):
            if not has_any_mp4(ddir):
                continue

            per_video, summary = run_folder(
                evaluator=evaluator,
                opt=opt,
                input_video_dir=str(ddir),
                device=args.device,
                num_workers=args.num_workers,
            )

            # append per-video rows
            with open(args.out_per_video_csv, "a") as f:
                for r in per_video:
                    f.write(
                        f"{mdir.name},{ddir.name},{r['path']},"
                        f"{r['aesthetic']:.6f},{r['technical']:.6f},{r['overall']:.6f}\n"
                    )

            # append summary row
            with open(args.out_summary_csv, "a") as f:
                f.write(
                    f"{mdir.name},{ddir.name},{summary['count']},{summary['failed']},"
                    f"{summary['mean_aesthetic']:.6f},{summary['mean_technical']:.6f},{summary['mean_overall']:.6f}\n"
                )

    print("Done.")
    print(f"Per-video CSV : {args.out_per_video_csv}")
    print(f"Summary CSV   : {args.out_summary_csv}")


if __name__ == "__main__":
    main()
