import numpy as np
from scipy.ndimage import uniform_filter
from skimage.restoration import denoise_nl_means, estimate_sigma

EPS = 1e-8


def auto_roi(img, size=24, stride=12, min_mean=0.1):
    H, W = img.shape
    best, best_cv2 = None, np.inf
    for r in range(0, H - size, stride):
        for c in range(0, W - size, stride):
            p = img[r:r + size, c:c + size]
            m = p.mean()
            if m < min_mean:
                continue
            cv2 = p.var() / (m ** 2 + EPS)
            if cv2 < best_cv2:
                best_cv2, best = cv2, (r, r + size, c, c + size)
    return best if best is not None else (0, H, 0, W)


def srad(img, n_iter=100, dt=0.2, roi=None):
    I = img.astype(np.float64) + 1e-3
    if roi is None:
        roi = auto_roi(img)
    r0, r1, c0, c1 = roi
    for _ in range(n_iter):
        patch = I[r0:r1, c0:c1]
        q0_sq = patch.var() / (patch.mean() ** 2 + EPS)
        P = np.pad(I, 1, mode="edge")
        dN = P[:-2, 1:-1] - I
        dS = P[2:, 1:-1] - I
        dW = P[1:-1, :-2] - I
        dE = P[1:-1, 2:] - I
        G2 = (dN ** 2 + dS ** 2 + dW ** 2 + dE ** 2) / (I ** 2)
        L = (dN + dS + dW + dE) / I
        q_sq = (0.5 * G2 - (L ** 2) / 16.0) / ((1.0 + 0.25 * L) ** 2 + EPS)
        c = 1.0 / (1.0 + (q_sq - q0_sq) / (q0_sq * (1.0 + q0_sq) + EPS))
        c = np.clip(c, 0.0, 1.0)
        Cp = np.pad(c, 1, mode="edge")
        cS = Cp[2:, 1:-1]
        cE = Cp[1:-1, 2:]
        div = cS * dS + c * dN + cE * dE + c * dW
        I = I + 0.25 * dt * div
    return np.clip(I - 1e-3, 0.0, 1.0)


def lee(img, win=7, roi=None):
    if roi is None:
        roi = auto_roi(img)
    r0, r1, c0, c1 = roi
    p = img[r0:r1, c0:c1]
    cu2 = p.var() / (p.mean() ** 2 + EPS)
    mean = uniform_filter(img, win)
    var = np.maximum(uniform_filter(img ** 2, win) - mean ** 2, 0)
    ci2 = var / (mean ** 2 + EPS)
    w = np.clip(1.0 - cu2 / (ci2 + EPS), 0.0, 1.0)
    return np.clip(mean + w * (img - mean), 0.0, 1.0)


def frost(img, win=7, damping=2.0):
    r = win // 2
    H, W = img.shape
    mean = uniform_filter(img, win)
    var = np.maximum(uniform_filter(img ** 2, win) - mean ** 2, 0)
    cv2 = var / (mean ** 2 + EPS)
    P = np.pad(img, r, mode="reflect")
    num = np.zeros_like(img)
    den = np.zeros_like(img)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            w = np.exp(-damping * cv2 * np.hypot(dy, dx))
            num += w * P[r + dy:r + dy + H, r + dx:r + dx + W]
            den += w
    return np.clip(num / (den + EPS), 0.0, 1.0)


def nlm(img, patch_size=5, patch_distance=6, h_factor=3.0):
    sigma = float(np.mean(estimate_sigma(img)))
    out = denoise_nl_means(img, h=h_factor * sigma, sigma=sigma,
                           patch_size=patch_size, patch_distance=patch_distance,
                           fast_mode=True)
    return np.clip(out, 0.0, 1.0)
