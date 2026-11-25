import numpy as np

SCALE = 1.0 / np.sqrt(np.pi / 2)  # Rayleigh scale giving mean 1
FLOOR = 1e-3


def random_phantom(rng, size=64):
    img = np.full((size, size), rng.uniform(0.1, 0.6))
    yy, xx = np.mgrid[:size, :size]
    for _ in range(int(rng.integers(3, 7))):
        level = rng.uniform(0.05, 0.9)
        kind = int(rng.integers(0, 3))
        if kind == 0:    # rectangle
            y0, x0 = rng.integers(0, size - 8, 2)
            h, w = rng.integers(6, size // 2, 2)
            img[y0:y0 + h, x0:x0 + w] = level
        elif kind == 1:  # circle
            cy, cx = rng.integers(0, size, 2)
            r = int(rng.integers(5, size // 3))
            img[(yy - cy) ** 2 + (xx - cx) ** 2 < r * r] = level
        else:            # band (vessel-like)
            y0 = int(rng.integers(0, size - 6))
            h = int(rng.integers(4, size // 3))
            img[y0:y0 + h, :] = level
    if rng.random() < 0.5:
        img = img.T
    return img


def make_batch(rng, batch=32, size=64):
    X, Y = [], []
    for _ in range(batch):
        clean = random_phantom(rng, size)
        noisy = np.maximum(clean * rng.rayleigh(SCALE, clean.shape), FLOOR)
        X.append(np.log(noisy))
        Y.append(np.log(noisy) - np.log(clean))   # = log(speckle)
    return (np.stack(X)[:, None].astype(np.float32),
            np.stack(Y)[:, None].astype(np.float32))
