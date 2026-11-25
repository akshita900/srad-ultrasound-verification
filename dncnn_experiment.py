import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dncnn_data import make_batch, FLOOR
from reproduce_paper import (carotid_phantom, ideal_edge_map, add_speckle, evaluate,
                             get_methods, N_RUNS)

WEIGHTS = "dncnn_speckle.pt"


def build_model(depth=8, ch=64):
    import torch.nn as nn

    class DnCNN(nn.Module):
        def __init__(self):
            super().__init__()
            layers = [nn.Conv2d(1, ch, 3, padding=1), nn.ReLU(inplace=True)]
            for _ in range(depth - 2):
                layers += [nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                           nn.BatchNorm2d(ch), nn.ReLU(inplace=True)]
            layers += [nn.Conv2d(ch, 1, 3, padding=1)]
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)   # predicts the log-speckle residual

    return DnCNN()


def train(steps=3000, batch=32, lr=1e-3, seed=123):
    import torch
    import torch.nn as nn
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device} for {steps} steps")
    torch.manual_seed(seed)
    model = build_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(steps // 2, 1), gamma=0.1)
    loss_fn = nn.MSELoss()
    rng = np.random.default_rng(seed)
    for step in range(1, steps + 1):
        X, Y = make_batch(rng, batch)
        X = torch.from_numpy(X).to(device)
        Y = torch.from_numpy(Y).to(device)
        loss = loss_fn(model(X), Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 250 == 0 or step == 1:
            print(f"  step {step}/{steps}  loss {loss.item():.4f}")
    torch.save(model.state_dict(), WEIGHTS)
    return model


def make_denoiser(model):
    import torch
    device = next(model.parameters()).device
    model.eval()

    def denoise(img):
        x = np.log(np.maximum(img, FLOOR)).astype(np.float32)
        with torch.no_grad():
            t = torch.from_numpy(x)[None, None].to(device)
            pred = model(t)[0, 0].cpu().numpy()
        return np.exp(x - pred)

    return denoise


def evaluate_phantom(denoise, label="DnCNN"):
    clean = carotid_phantom()
    ideal = ideal_edge_map(clean)
    methods = get_methods()
    methods[label] = denoise
    rows = []
    for seed in range(N_RUNS):
        noisy = add_speckle(clean, np.random.default_rng(seed))
        for name, f in methods.items():
            m = evaluate(clean, noisy, f(noisy), ideal)
            m.update(method=name, seed=seed)
            rows.append(m)
    df = pd.DataFrame(rows)
    order = list(methods)
    t = df.groupby("method").mean(numeric_only=True).drop(columns="seed").loc[order]
    t.to_csv("dncnn_phantom_summary.csv")
    print(f"\n=== Carotid phantom incl. {label}: mean over {N_RUNS} realizations ===")
    print(t.round(3))

    noisy = add_speckle(clean, np.random.default_rng(0))
    panels = {"Clean": clean, "Noisy": noisy, "Frost": methods["Frost"](noisy),
              "SRAD": methods["SRAD"](noisy), label: denoise(noisy)}
    fig, ax = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for a, (k, v) in zip(ax, panels.items()):
        a.imshow(np.clip(v, 0, 1), cmap="gray", vmin=0, vmax=1)
        a.set_title(k)
        a.axis("off")
    plt.tight_layout()
    plt.savefig("dncnn_phantom_comparison.png", dpi=150)
    return t


def evaluate_busi(denoise, data_dir, n_images=50, label="DnCNN"):
    from run_experiment import load_busi, read_gray, enl, epi, cnr
    from despeckle import auto_roi, srad, lee
    pairs = load_busi(data_dir, n_images)
    print(f"\nBUSI transfer test on {len(pairs)} images...")
    rows = []
    for ip, mp in pairs:
        img = read_gray(ip)
        mask = read_gray(mp, is_mask=True) > 0.5
        roi = auto_roi(img)
        outs = {"Original": img, "Lee": lee(img, roi=roi),
                "SRAD": srad(img, n_iter=100, dt=0.2, roi=roi), label: denoise(img)}
        for name, out in outs.items():
            rows.append({"method": name, "ENL": enl(out, roi),
                         "EPI": epi(img, out), "CNR": cnr(out, mask)})
    t = pd.DataFrame(rows).groupby("method").mean().loc[["Original", "Lee", "SRAD", label]]
    t.to_csv("dncnn_busi_summary.csv")
    print(f"=== BUSI mean results incl. {label} ===")
    print(t.round(3))
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--load", action="store_true")
    ap.add_argument("--busi_dir", type=str, default=None)
    ap.add_argument("--n_images", type=int, default=50)
    args = ap.parse_args()

    if args.load and os.path.exists(WEIGHTS):
        import torch
        model = build_model()
        model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
        if torch.cuda.is_available():
            model = model.cuda()
    else:
        model = train(steps=args.steps)
    denoise = make_denoiser(model)
    evaluate_phantom(denoise)
    if args.busi_dir:
        evaluate_busi(denoise, args.busi_dir, args.n_images)
