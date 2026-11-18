import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import laplace, binary_dilation, binary_opening
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.transform import resize

from despeckle import srad, lee, frost, nlm, auto_roi

EPS = 1e-8
SIZE = 256


def enl(img, roi):
    r0, r1, c0, c1 = roi
    p = img[r0:r1, c0:c1]
    return p.mean() ** 2 / (p.var() + EPS)


def epi(orig, filt):
    a, b = laplace(orig), laplace(filt)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + EPS))


def cnr(img, mask):
    lesion = mask > 0
    ring = binary_dilation(lesion, iterations=15) & ~binary_dilation(lesion, iterations=3)
    if lesion.sum() == 0 or ring.sum() == 0:
        return np.nan
    l, b = img[lesion], img[ring]
    return float(abs(l.mean() - b.mean()) / np.sqrt(l.var() + b.var() + EPS))


def dice_simple_seg(img, mask):
    g = gaussian(img, 2)
    seg = g < threshold_otsu(g)
    seg = binary_opening(seg, iterations=2)
    lab = label(seg)
    H, W = lab.shape
    best, best_size = None, 0
    for k in range(1, lab.max() + 1):
        comp = lab == k
        rows, cols = np.where(comp)
        touches = rows.min() == 0 or cols.min() == 0 or rows.max() == H - 1 or cols.max() == W - 1
        if touches or comp.sum() < 200:
            continue
        if comp.sum() > best_size:
            best, best_size = comp, comp.sum()
    if best is None:
        return 0.0
    gt = mask > 0
    return float(2 * (best & gt).sum() / (best.sum() + gt.sum() + EPS))


def get_filters():
    return {
        "Original": lambda x, roi: x,
        "Lee": lambda x, roi: lee(x, roi=roi),
        "Frost": lambda x, roi: frost(x),
        "NLM": lambda x, roi: nlm(x),
        "SRAD": lambda x, roi: srad(x, n_iter=100, dt=0.2, roi=roi),
    }


def synthetic_test(looks=4, seed=0):
    n = SIZE
    clean = np.full((n, n), 0.5)
    yy, xx = np.mgrid[:n, :n]
    clean[(yy - 90) ** 2 + (xx - 90) ** 2 < 40 ** 2] = 0.2
    clean[(yy - 170) ** 2 + (xx - 170) ** 2 < 35 ** 2] = 0.8
    rng = np.random.default_rng(seed)
    noisy = np.clip(clean * rng.gamma(looks, 1.0 / looks, clean.shape), 0, 1)
    roi = (200, 230, 20, 50)
    rows = []
    for name, f in get_filters().items():
        out = f(noisy, roi)
        rows.append({"method": name,
                     "PSNR": psnr(clean, out, data_range=1.0),
                     "SSIM": ssim(clean, out, data_range=1.0),
                     "ENL": enl(out, roi),
                     "EPI": epi(clean, out)})
    df = pd.DataFrame(rows).set_index("method")
    print("\n=== Synthetic phantom (ground truth available) ===")
    print(df.round(3))
    df.to_csv("results_synthetic.csv")


def load_busi(data_dir, n_images):
    pairs = []
    for img_path in sorted(glob.glob(os.path.join(data_dir, "*", "*).png"))):
        mask_path = img_path.replace(").png", ")_mask.png")
        if os.path.exists(mask_path) and "normal" not in img_path:
            pairs.append((img_path, mask_path))
    idx = np.linspace(0, len(pairs) - 1, min(n_images, len(pairs))).astype(int)
    return [pairs[i] for i in idx]


def read_gray(path, is_mask=False):
    x = io.imread(path)
    if x.ndim == 3:
        x = rgb2gray(x[..., :3])
    x = x.astype(np.float64)
    if x.max() > 1:
        x = x / 255.0
    return resize(x, (SIZE, SIZE), order=0 if is_mask else 1,
                  anti_aliasing=not is_mask)


def busi_experiment(data_dir, n_images):
    pairs = load_busi(data_dir, n_images)
    print(f"Running on {len(pairs)} BUSI images...")
    filters = get_filters()
    rows, example = [], None
    for i, (ip, mp) in enumerate(pairs):
        img, mask = read_gray(ip), read_gray(mp, is_mask=True) > 0.5
        roi = auto_roi(img)
        outs = {k: f(img, roi) for k, f in filters.items()}
        if example is None:
            example = (img, outs)
        for name, out in outs.items():
            rows.append({"image": os.path.basename(ip), "method": name,
                         "ENL": enl(out, roi),
                         "EPI": epi(img, out),
                         "SSIM_vs_original": ssim(img, out, data_range=1.0),
                         "CNR": cnr(out, mask),
                         "Dice_otsu": dice_simple_seg(out, mask)})
        print(f"  {i + 1}/{len(pairs)}", end="\r")
    df = pd.DataFrame(rows)
    df.to_csv("results_busi_per_image.csv", index=False)
    summary = df.groupby("method").mean(numeric_only=True).loc[list(filters)]
    print("\n\n=== BUSI mean results ===")
    print(summary.round(3))
    summary.to_csv("results_busi_summary.csv")
    img, outs = example
    fig, ax = plt.subplots(1, len(outs), figsize=(4 * len(outs), 4))
    for a, (k, v) in zip(ax, outs.items()):
        a.imshow(v, cmap="gray", vmin=0, vmax=1)
        a.set_title(k)
        a.axis("off")
    plt.tight_layout()
    plt.savefig("example_comparison.png", dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--data_dir", type=str, default=None)
    ap.add_argument("--n_images", type=int, default=50)
    args = ap.parse_args()
    if args.synthetic or args.data_dir is None:
        synthetic_test()
    if args.data_dir:
        busi_experiment(args.data_dir, args.n_images)
