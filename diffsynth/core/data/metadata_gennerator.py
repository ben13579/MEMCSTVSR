#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".wmv", ".mkv", ".flv", ".webm"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def is_video(p: Path) -> bool:
    return p.suffix.lower() in VIDEO_EXTS


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def find_videos(base_dir: Path) -> List[Path]:
    return sorted([p for p in base_dir.rglob("*") if p.is_file() and is_video(p)])


def find_frame_sequences(base_dir: Path, min_images: int = 2) -> List[Tuple[Path, List[Path]]]:
    """
    找出疑似「影片 frame 資料夾」：任意資料夾底下若含 >= min_images 的影像檔，就當作一個 sequence。
    回傳: [(seq_dir, [img_paths_sorted]), ...]
    """
    seqs = []
    # 用 rglob 掃 image，然後按 parent 分組，避免掃每個資料夾兩次
    buckets = {}
    for img in base_dir.rglob("*"):
        if img.is_file() and is_image(img):
            buckets.setdefault(img.parent, []).append(img)

    for d, imgs in buckets.items():
        if len(imgs) >= min_images:
            def numeric_stem_key(p):
                # p.stem = "0" / "1" / "11"
                try:
                    return int(p.stem)
                except ValueError:
                    return p.stem  # fallback

            imgs_sorted = sorted(imgs, key=numeric_stem_key)
            seqs.append((d, imgs_sorted))

    # 固定排序：依資料夾路徑
    seqs.sort(key=lambda x: str(x[0]))
    return seqs


def safe_relpath(path: Path, base_dir: Path) -> str:
    # 轉成 POSIX，避免 Windows 反斜線造成 downstream parser 麻煩
    return path.relative_to(base_dir).as_posix()


def write_jsonl(path: Path, rows: List[dict], dry_run: bool = False):
    if dry_run:
        print("[jsonl] dry-run: not writing file")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[jsonl] wrote: {path}")



def get_video_frame_count_imageio(video_path: Path) -> Optional[int]:
    """
    優先用 imageio 讀取 count_frames。不同 codec/容器可能失敗，失敗就回傳 None。
    """
    try:
        import imageio
        reader = imageio.get_reader(str(video_path))
        try:
            n = int(reader.count_frames())
            return n
        finally:
            reader.close()
    except Exception:
        return None


def generate_video_jsonl(
    base_dir: Path,
    out_jsonl: Path,
    k: int,
    stride: int = 1,
    start_offset: int = 0,
    max_videos: int = 0,
    dry_run: bool = False,
) -> None:
    videos = find_videos(base_dir)
    if max_videos and max_videos > 0:
        videos = videos[:max_videos]

    rows = []
    skipped = 0

    for vp in videos:
        n = get_video_frame_count_imageio(vp)
        if n is None:
            skipped += 1
            continue

        # 可從 start_offset 起算（例如跳過前面幾幀）
        # window start: start_offset, start_offset+stride, ...
        last_start = n - k
        if last_start < start_offset:
            skipped += 1
            continue

        rel_v = safe_relpath(vp, base_dir)
        for s in range(start_offset, last_start + 1, stride):
            rows.append({
                "video": rel_v,
                "prompt": "",  # 一律空字串
                "start": s,
                "k": k,
            })

    print(f"[video] base_dir={base_dir}")
    print(f"[video] found_videos={len(videos)}, skipped_videos={skipped}, total_rows={len(rows)}")
    write_jsonl(out_jsonl, rows, dry_run)


def generate_frames_jsonl(
    base_dir: Path,
    out_jsonl: Path,
    k: int,
    stride: int = 1,
    min_images_per_seq: int = 2,
    max_seqs: int = 0,
    dry_run: bool = False,
) -> None:
    seqs = find_frame_sequences(base_dir, min_images=min_images_per_seq)
    if max_seqs and max_seqs > 0:
        seqs = seqs[:max_seqs]

    rows = []
    skipped = 0

    for seq_dir, imgs in seqs:
        n = len(imgs)
        if n < k:
            skipped += 1
            continue

        # 產 sliding windows
        for s in range(0, n - k + 1, stride):
            clip = imgs[s:s + k]
            rels = [safe_relpath(p, base_dir) for p in clip]
            rows.append({
                "frames": rels,   # JSONL 可直接放 list
                "prompt": "",
            })

    print(f"[frames] base_dir={base_dir}")
    print(f"[frames] found_seqs={len(seqs)}, skipped_seqs={skipped}, total_rows={len(rows)}")
    write_jsonl(out_jsonl, rows, dry_run)


def main():
    ap = argparse.ArgumentParser("STVSR metadata generator (relative paths based on base_dir)")
    ap.add_argument("--base_dir", type=str, required=True,
                    help="metadata 生成路徑（也會以此為 root 遞迴搜尋資料）；JSONL 內路徑會相對於此。")
    ap.add_argument("--out_jsonl", type=str, required=True,
                    help="輸出 JSONL 路徑（建議放在 base_dir 裡；若放外面也沒問題，路徑仍以 base_dir 計）。")
    ap.add_argument("--mode", type=str, choices=["video", "frames"], required=True,
                    help="video: 輸出 video,prompt,start,k；frames: 輸出 frames,prompt（frames 為 JSON list 字串）")
    ap.add_argument("--k", type=int, required=True, help="clip 長度 k")
    ap.add_argument("--stride", type=int, default=1, help="sliding window stride（預設 1）")

    # video-only
    ap.add_argument("--start_offset", type=int, default=0, help="(video) window 起始 frame offset（預設 0）")
    ap.add_argument("--max_videos", type=int, default=0, help="(video) 只處理前 N 個 video（0=不限）")

    # frames-only
    ap.add_argument("--min_images_per_seq", type=int, default=2,
                    help="(frames) 資料夾至少幾張圖才視為 sequence（預設 2）")
    ap.add_argument("--max_seqs", type=int, default=0, help="(frames) 只處理前 N 個 sequences（0=不限）")

    ap.add_argument("--dry_run", action="store_true", help="不寫檔，只印統計")
    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    out_jsonl = Path(args.out_jsonl).resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"base_dir not found: {base_dir}")

    if args.k <= 0 or args.stride <= 0:
        raise ValueError("k and stride must be positive integers.")

    if args.mode == "video":
        generate_video_jsonl(
            base_dir=base_dir,
            out_jsonl=out_jsonl,
            k=args.k,
            stride=args.stride,
            start_offset=args.start_offset,
            max_videos=args.max_videos,
            dry_run=args.dry_run,
        )
    else:
        generate_frames_jsonl(
            base_dir=base_dir,
            out_jsonl=out_jsonl,
            k=args.k,
            stride=args.stride,
            min_images_per_seq=args.min_images_per_seq,
            max_seqs=args.max_seqs,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
