"""Verification of an SRAD implementation against the claims of Yu & Acton,
"Speckle Reducing Anisotropic Diffusion", IEEE TIP 11(11), 2002.

Paper's claim (abstract): on simulated speckled (carotid-artery) images, SRAD excels over
traditional speckle filters and over conventional (Perona-Malik) anisotropic diffusion in
(1) mean preservation, (2) variance reduction, (3) edge localization.

We test these claims on a simulated carotid-like phantom with known ground truth, using
20 independent speckle realizations. Acceptance criteria R1-R4 are derived from the
abstract's three claims; PASS/FAIL is reported as-is.
Note: the edge metric was revised once after a first run (Canny sigma=2 smoothed away the
differences between filters, so every method scored the same); the criteria were not changed.

Run:  python reproduce_paper.py
"""
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter, distance_transform_edt
from skimage.feature import canny
from skimage.metrics import peak_signal_noise_ratio as psnr

from despeckle import srad, lee, frost, auto_roi

EPS = 1e-8
N = 256
N_RUNS = 20
ROI = (10, 60, 10, 60)  # homogeneous tissue patch, top-left


# ------------------------------------------------------------- extra filters
def kuan(img, win=7, roi=None):
    if roi is None:
        roi = auto_roi(img)
    r0, r1, c0, c1 = roi
    p = img[r0:r1, c0:c1]
    cu2 = p.var() / (p.mean() ** 2 + EPS)
    mean = uniform_filter(img, win)
    var = np.maximum(uniform_filter(img ** 2, win) - mean ** 2, 0)
    ci2 = var / (mean ** 2 + EPS)
    w = np.clip((1.0 - cu2 / (ci2 + EPS)) / (1.0 + cu2), 0.0, 1.0)
    return mean + w * (img - mean)


def perona_malik(img, n_iter=100, dt=0.15, kappa=0.1):
    """Conventional anisotropic diffusion (Perona & Malik 1990), exponential c(.)."""
    I = img.astype(np.float64).copy()
    for _ in range(n_iter):
        P = np.pad(I, 1, mode="edge")
        dN = P[:-2, 1:-1] - I
        dS = P[2:, 1:-1] - I
        dW = P[1:-1, :-2] - I
        dE = P[1:-1, 2:] - I
        I = I + dt * (np.exp(-(dN / kappa) ** 2) * dN + np.exp(-(dS / kappa) ** 2) * dS
                      + np.exp(-(dW / kappa) ** 2) * dW + np.exp(-(dE / kappa) ** 2) * dE)
    return I


# ------------------------------------------------------------- phantom + noise
def carotid_phantom():
    """Longitudinal carotid-like image: dark lumen, bright vessel walls, tissue, plaque."""
    img = np.full((N, N), 0.3)
    img[96:104, :] = 0.5
    img[152:160, :] = 0.5
    img[104:152, :] = 0.1
    yy, xx = np.mgrid[:N, :N]
    img[(yy - 148) ** 2 + (xx - 150) ** 2 < 14 ** 2] = 0.4
    return img


def ideal_edge_map(clean):
    """One-pixel-wide ground-truth edges: wherever the clean image changes value."""
    e = np.zeros(clean.shape, bool)
    e[1:, :] |= np.abs(np.diff(clean, axis=0)) > 1e-6
    e[:, 1:] |= np.abs(np.diff(clean, axis=1)) > 1e-6
    return e


def add_speckle(clean, rng):
    """Fully developed speckle: multiplicative Rayleigh amplitude noise with mean 1."""
    return clean * rng.rayleigh(scale=1 / np.sqrt(np.pi / 2), size=clean.shape)


# ------------------------------------------------------------- metrics
def pratt_fom(ideal, detected, alpha=1.0 / 9):
    """Pratt's figure of merit for edge localization (1.0 = perfect)."""
    if detected.sum() == 0:
        return 0.0
    d = distance_transform_edt(~ideal)[detected]
    return float((1.0 / (1.0 + alpha * d ** 2)).sum() / max(ideal.sum(), detected.sum()))


def detect_edges(img):
    # sigma=1 and quantile thresholds so the detector does not hide the filter's quality
    return canny(img, sigma=1.0, low_threshold=0.90, high_threshold=0.95, use_quantiles=True)


def evaluate(clean, noisy, out, ideal):
    r0, r1, c0, c1 = ROI
    p, pn = out[r0:r1, c0:c1], noisy[r0:r1, c0:c1]
    return {
        "MeanRatio": out.mean() / noisy.mean(),          # mean preservation (ideal 1.0)
        "ENL": p.mean() ** 2 / (p.var() + EPS),          # variance reduction (higher = better)
        "VarRatio": p.var() / (pn.var() + EPS),          # lower = more variance removed
        "FOM": pratt_fom(ideal, detect_edges(out)),      # edge localization (higher = better)
        "PSNR": psnr(clean, np.clip(out, 0, 1), data_range=1.0),
    }


def get_methods():
    return {
        "Noisy": lambda x: x,
        "Lee": lambda x: lee(x, roi=ROI),
        "Frost": lambda x: frost(x),
        "Kuan": lambda x: kuan(x, roi=ROI),
        "PeronaMalik": lambda x: perona_malik(x),
        "SRAD": lambda x: srad(x, n_iter=100, dt=0.2, roi=ROI),
    }


def main():
    clean = carotid_phantom()
    ideal = ideal_edge_map(clean)
    methods = get_methods()
    rows = []
    for seed in range(N_RUNS):
        noisy = add_speckle(clean, np.random.default_rng(seed))
        for name, f in methods.items():
            m = evaluate(clean, noisy, f(noisy), ideal)
            m.update(method=name, seed=seed)
            rows.append(m)
    df = pd.DataFrame(rows)
    df.to_csv("paper_repro_per_run.csv", index=False)
    order = list(methods)
    t = df.groupby("method").mean(numeric_only=True).drop(columns="seed").loc[order]
    s = df.groupby("method").std(numeric_only=True).drop(columns="seed").loc[order]
    t.to_csv("paper_repro_summary.csv")
    print(f"\n=== Simulated carotid phantom: mean over {N_RUNS} speckle realizations ===")
    print(t.round(3))
    print("\n=== Std over realizations ===")
    print(s.round(3))

    trad = ["Lee", "Frost", "Kuan"]
    allref = trad + ["PeronaMalik"]
    claims = [
        ("R1 Mean preservation: SRAD output mean within 1% of input mean",
         abs(1 - t.loc["SRAD", "MeanRatio"]) < 0.01),
        ("R2 Variance reduction: SRAD ENL higher than Lee/Frost/Kuan/PM",
         all(t.loc["SRAD", "ENL"] > t.loc[m, "ENL"] for m in allref)),
        ("R3 Edge localization: SRAD FOM higher than Perona-Malik",
         t.loc["SRAD", "FOM"] > t.loc["PeronaMalik", "FOM"]),
        ("R4 Edge localization: SRAD FOM higher than Lee/Frost/Kuan",
         all(t.loc["SRAD", "FOM"] > t.loc[m, "FOM"] for m in trad)),
    ]
    print("\n=== Verification of the paper's claims (criteria derived from the paper's abstract) ===")
    for text, ok in claims:
        print(("PASS  " if ok else "FAIL  ") + text)

    # Matched-smoothing check: ENL can be inflated by more iterations, so also compare
    # edge localization of SRAD at several iteration counts against the traditional filters.
    print("\n=== SRAD iteration sweep (dt=0.2) vs traditional filters ===")
    sweep = []
    for n_iter in [25, 50, 75, 100, 150, 200]:
        r = []
        for seed in range(10):
            noisy = add_speckle(clean, np.random.default_rng(seed))
            r.append(evaluate(clean, noisy, srad(noisy, n_iter=n_iter, dt=0.2, roi=ROI), ideal))
        m = pd.DataFrame(r).mean()
        sweep.append({"n_iter": n_iter, "ENL": m["ENL"], "FOM": m["FOM"], "PSNR": m["PSNR"]})
    sw = pd.DataFrame(sweep)
    sw.to_csv("paper_repro_srad_sweep.csv", index=False)
    print(sw.round(3).to_string(index=False))
    print("\nReference (traditional filters, from the table above):")
    print(t.loc[trad, ["ENL", "FOM", "PSNR"]].round(3))


if __name__ == "__main__":
    main()
