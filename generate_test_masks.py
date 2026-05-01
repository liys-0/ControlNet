"""Generate random binary defect masks (white convex polygons on black) for each
image in --image_dir, saving to --mask_dir under the same filename.

Each polygon is convex by construction (vertices sampled on an ellipse), with
bounding-box width and height both >= --min_size pixels.
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def sample_convex_polygon(img_w: int, img_h: int, min_size: int, max_size: int,
                          n_vertices: int, rng: random.Random) -> np.ndarray:
    """Convex polygon whose bbox is W x H with min_size <= W,H <= max_size, fitting in image."""
    if min_size > img_w or min_size > img_h:
        raise ValueError(f"min_size {min_size} too large for image {img_w}x{img_h}")
    if max_size < min_size:
        raise ValueError(f"max_size {max_size} < min_size {min_size}")

    # Random bbox dimensions and position fully inside the image.
    W = rng.randint(min_size, min(max_size, img_w))
    H = rng.randint(min_size, min(max_size, img_h))
    x0 = rng.randint(0, img_w - W)
    y0 = rng.randint(0, img_h - H)

    cx = x0 + W / 2.0
    cy = y0 + H / 2.0
    rx = W / 2.0
    ry = H / 2.0

    # Sample sorted angles with jitter so vertices are well-distributed and the
    # polygon stays convex (sorted angles on an ellipse => convex).
    base = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
    jitter = np.array([rng.uniform(-0.5, 0.5) for _ in range(n_vertices)])
    angles = np.sort(base + jitter * (2 * np.pi / n_vertices))

    pts = np.stack(
        [cx + rx * np.cos(angles), cy + ry * np.sin(angles)], axis=1
    )

    # Force the bbox to actually span the full W x H so the size guarantee holds
    # even after rounding (the unit-ellipse extremes are always hit at angles
    # 0, pi/2, pi, 3pi/2 — but we jitter, so re-snap to bbox corners on each axis).
    pts[:, 0] = np.clip(pts[:, 0], x0, x0 + W)
    pts[:, 1] = np.clip(pts[:, 1], y0, y0 + H)
    # Ensure at least one vertex touches each side of the bbox.
    pts[np.argmin(pts[:, 0]), 0] = x0
    pts[np.argmax(pts[:, 0]), 0] = x0 + W
    pts[np.argmin(pts[:, 1]), 1] = y0
    pts[np.argmax(pts[:, 1]), 1] = y0 + H

    # Take convex hull to guarantee convexity after the snap-to-bbox step.
    hull = cv2.convexHull(pts.astype(np.float32))
    return hull.astype(np.int32)


def make_mask(img_w: int, img_h: int, num_polys: int, min_size: int, max_size: int,
              n_vertices_range: tuple[int, int], rng: random.Random) -> np.ndarray:
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for _ in range(num_polys):
        n_v = rng.randint(*n_vertices_range)
        poly = sample_convex_polygon(img_w, img_h, min_size, max_size, n_v, rng)
        cv2.fillPoly(mask, [poly], 255)
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", default="./for_ControlNet_all/test_4ch_images")
    ap.add_argument("--mask_dir", default="./for_ControlNet_all/test_4ch_masks")
    ap.add_argument("--min_size", type=int, default=50,
                    help="Minimum bbox width/height of each polygon, in pixels.")
    ap.add_argument("--max_size", type=int, default=100,
                    help="Maximum bbox width/height of each polygon, in pixels.")
    ap.add_argument("--num_polys", type=int, default=1,
                    help="Number of polygons per mask.")
    ap.add_argument("--min_vertices", type=int, default=5)
    ap.add_argument("--max_vertices", type=int, default=10)
    ap.add_argument("--seed", type=int, default=None,
                    help="Random seed for reproducibility (default: nondeterministic).")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        print(f"No images found in {image_dir}")
        return

    for ipath in images:
        with Image.open(ipath) as im:
            w, h = im.size
        mask = make_mask(
            w, h, args.num_polys, args.min_size, args.max_size,
            (args.min_vertices, args.max_vertices), rng,
        )
        out_path = mask_dir / ipath.name
        cv2.imwrite(str(out_path), mask)

    print(f"Wrote {len(images)} masks to {mask_dir}")


if __name__ == "__main__":
    main()
