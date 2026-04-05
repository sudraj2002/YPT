#!/usr/bin/env python3
import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List

IMG_EXTS = {".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"}
WAN_PRED_DIRNAME = "pred"

# Extract frame index for sorting
RE_FRAME = re.compile(r"(?:^|[_-])frame[_-]?(?P<idx>\d+)", re.IGNORECASE)   # frame_000012
RE_LASTNUM = re.compile(r"(?P<idx>\d+)(?:\D*$)")                            # last digits near end
RE_ANYNUM = re.compile(r"(\d+)")

# 4KRD special split: input300/VID_013_00225...
RE_4KRD_VID = re.compile(r"^VID_(?P<vid>\d{3})_(?P<frame>\d+)", re.IGNORECASE)

def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in {e.lower() for e in IMG_EXTS}

def extract_frame_idx(name: str) -> int:
    stem = Path(name).stem
    m = RE_FRAME.search(stem)
    if m:
        return int(m.group("idx"))
    m = RE_LASTNUM.search(stem)
    if m:
        return int(m.group("idx"))
    nums = RE_ANYNUM.findall(stem)
    return int(nums[-1]) if nums else 0

def images_in_dir(d: Path) -> List[Path]:
    frames = [p for p in d.iterdir() if is_image(p)]
    frames.sort(key=lambda p: extract_frame_idx(p.name))
    return frames

def ffmpeg_make_mp4(frames: List[Path], out_mp4: Path, fps: int, lossless: bool, size: int) -> None:
    if not frames:
        return
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        list_file = Path(td) / "list.txt"
        with open(list_file, "w") as f:
            for p in frames:
                pp = str(p).replace("'", "'\\''")
                f.write(f"file '{pp}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-r", str(fps),
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-an",
        ]

        if size and size > 0:
            cmd += ["-vf", f"scale={size}:{size}:flags=lanczos,setsar=1"]

        if lossless:
            cmd += ["-c:v", "libx264", "-crf", "0", "-preset", "veryslow", "-pix_fmt", "yuv444p"]
        else:
            cmd += ["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p"]

        cmd += [str(out_mp4)]
        subprocess.run(cmd, check=True)

def split_4krd_by_vid(frames: List[Path]) -> Dict[str, List[Path]]:
    """
    Split frames like:
      .../4KRD/input300/pred/VID_013_00225.jpg
    into groups keyed by:
      input300_VID_013
    """
    groups: Dict[str, List[Path]] = {}

    for p in frames:
        m = RE_4KRD_VID.match(p.name)
        if m:
            # frames_dir is .../input300/pred, so parent of parent is input300
            prefix = p.parent.parent.name
            key = f"{prefix}_VID_{m.group('vid')}"
        else:
            key = "__single__"

        groups.setdefault(key, []).append(p)

    for k, v in groups.items():
        v.sort(key=lambda x: extract_frame_idx(x.name))

    if set(groups.keys()) == {"__single__"}:
        return {"__single__": groups["__single__"]}

    return {k: v for k, v in groups.items() if k != "__single__"}

def process_method(method_dir: Path, out_root: Path, fps: int, lossless: bool, min_frames: int, size: int) -> None:
    for dataset_dir in sorted([d for d in method_dir.iterdir() if d.is_dir()]):
        # each subdir is a sequence id (e.g., 368) OR sometimes input300
        seq_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
        if not seq_dirs:
            continue

        # detect WAN by presence of pred/
        is_wan = any((sd / WAN_PRED_DIRNAME).is_dir() for sd in seq_dirs[:50])

        for seq_dir in sorted(seq_dirs):
            # skip logs/csv folders if any weirdness
            if not seq_dir.is_dir():
                continue

            if is_wan:
                frames_dir = seq_dir / WAN_PRED_DIRNAME
                if not frames_dir.is_dir():
                    continue
            else:
                frames_dir = seq_dir

            frames = images_in_dir(frames_dir)
            if not frames:
                continue

            # 4KRD special case: folder like input300 contains many VID_### sequences
            if dataset_dir.name.lower().startswith("4krd") or dataset_dir.name.lower().startswith("4krd"):
                groups = split_4krd_by_vid(frames)
            else:
                groups = {"__single__": frames}

            for gname, gframes in sorted(groups.items()):
                if len(gframes) < min_frames:
                    continue

                if gname == "__single__":
                    out_name = f"{seq_dir.name}.mp4"
                else:
                    out_name = f"{gname}.mp4"  # e.g. input300_VID_013.mp4

                out_mp4 = out_root / method_dir.name / dataset_dir.name / out_name
                ffmpeg_make_mp4(gframes, out_mp4, fps=fps, lossless=lossless, size=size)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=str)
    ap.add_argument("--out_root", required=True, type=str)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--lossless", action="store_true")
    ap.add_argument("--min_frames", type=int, default=33)
    ap.add_argument("--size", type=int, default=0)
    ap.add_argument("--methods", type=str, default="", help="comma-separated subset (for WAN/AverNet/ViWS-Net)")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")

    root = Path(args.root).resolve()
    out_root = Path(args.out_root).resolve()

    allowed = None
    if args.methods.strip():
        allowed = {m.strip() for m in args.methods.split(",") if m.strip()}

    for method_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        if allowed and method_dir.name not in allowed:
            continue
        print(f"[+] {method_dir.name}")
        process_method(method_dir, out_root, fps=args.fps, lossless=args.lossless,
                       min_frames=args.min_frames, size=args.size)

    print(f"\nDone. MP4s at: {out_root}")

if __name__ == "__main__":
    main()