"""Overlay mask boundaries (red contour) onto generated samples.

For each mask `<stem>.png` in --mask_dir, find every generated image
`<stem>_sample*.png` in --gen_dir, draw the mask boundary as a red line on top,
and save to --out_dir with the same filename.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def find_samples(gen_dir: Path, stem: str) -> list[Path]:
    return sorted(gen_dir.glob(f"{stem}_sample*.png"))


def boundary_overlay(gen_bgr: np.ndarray, mask: np.ndarray,
                     color=(0, 0, 255), thickness=2) -> np.ndarray:
    # Resize mask to match generated image (NEAREST keeps binary edges crisp).
    h, w = gen_bgr.shape[:2]
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Binarize defensively in case mask has values other than {0, 255}.
    _, binmask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = gen_bgr.copy()
    cv2.drawContours(out, contours, -1, color, thickness, lineType=cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", default="./for_ControlNet_all/gen_4ch_avalon_fullweight_defect_image_epoc82")
    ap.add_argument("--mask_dir", default="./for_ControlNet_all/test_4ch_masks")
    ap.add_argument("--out_dir", default="./for_ControlNet_all/gen_4ch_avalon_fullweight_defect_image_epoc82_overlay")
    ap.add_argument("--thickness", type=int, default=2)
    args = ap.parse_args()

    gen_dir = Path(args.gen_dir)
    mask_dir = Path(args.mask_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    masks = sorted(mask_dir.glob("*.png"))
    if not masks:
        print(f"No masks found in {mask_dir}")
        return

    n_done = 0
    n_missing = 0
    for mpath in masks:
        mask = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"  skip unreadable mask: {mpath.name}")
            continue

        samples = find_samples(gen_dir, mpath.stem)
        if not samples:
            print(f"  no samples for {mpath.stem}")
            n_missing += 1
            continue

        for spath in samples:
            gen = cv2.imread(str(spath), cv2.IMREAD_COLOR)
            if gen is None:
                print(f"  skip unreadable sample: {spath.name}")
                continue
            out = boundary_overlay(gen, mask, thickness=args.thickness)
            cv2.imwrite(str(out_dir / spath.name), out)
            n_done += 1

    print(f"Wrote {n_done} overlays to {out_dir}")
    if n_missing:
        print(f"({n_missing} masks had no matching samples)")


if __name__ == "__main__":
    main()
