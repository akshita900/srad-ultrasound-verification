# SRAD Verification: Speckle Reducing Anisotropic Diffusion for Ultrasound

Python implementation of **Speckle Reducing Anisotropic Diffusion (SRAD)**, written from the
published equations (Yu & Acton, *IEEE Trans. Image Processing*, 11(11), 2002), and a
verification study of the paper's claims against Lee, Frost, Kuan, Perona-Malik and
non-local means (NLM) filters.

**Author:** Akshita Choudhary, B.Tech Bioengineering, MIT ADT University

## Why this project

Speckle noise lowers contrast and edge clarity in ultrasound images, so speckle reduction is a
core image-quality step in ultrasound systems. The goal here was to implement the method from
the paper's equations, then check whether its published claims hold in my own experiments, and
record what passed and what did not.

## Method

SRAD replaces the gradient-based diffusion coefficient of Perona-Malik with the instantaneous
coefficient of variation `q`, so smoothing adapts to speckle statistics:

- `q^2 = (0.5 |grad I|^2 / I^2 - (1/16)(lap I / I)^2) / (1 + 0.25 lap I / I)^2`
- `c(q) = 1 / (1 + (q^2 - q0^2) / (q0^2 (1 + q0^2)))`
- update: `I <- I + (dt/4) div(c grad I)`, with `q0` estimated from a homogeneous region

Implementation: `despeckle.py` (SRAD, Lee, Frost, NLM) and `reproduce_paper.py` (Kuan,
Perona-Malik, phantom, metrics, verification).

## Study 1: verification of the paper's claims (simulated carotid-like phantom)

Setup: longitudinal carotid-like phantom (lumen, vessel walls, plaque) with known ground truth,
fully developed speckle (multiplicative Rayleigh noise), 20 independent noise realizations.
Acceptance criteria were derived from the abstract's three claims (mean preservation, variance
reduction, edge localization).

| ID | Criterion | Result |
|----|-----------|--------|
| R1 | SRAD output mean within 1% of input mean | PASS |
| R2 | SRAD ENL higher than Lee, Frost, Kuan, Perona-Malik | PASS |
| R3 | SRAD edge localization (Pratt FOM) higher than Perona-Malik | PASS |
| R4 | SRAD edge localization (Pratt FOM) higher than Lee, Frost, Kuan | **FAIL** |

Mean over 20 realizations:

| Method | Mean ratio | ENL | FOM | PSNR (dB) |
|--------|-----------|-----|-----|-----------|
| Noisy | 1.000 | 3.6 | 0.458 | 16.4 |
| Lee | 0.998 | 113.4 | 0.875 | 28.3 |
| Frost | 0.997 | 133.4 | 0.895 | 29.7 |
| Kuan | 0.998 | 134.9 | 0.891 | 28.9 |
| Perona-Malik | 1.000 | 10.6 | 0.579 | 20.2 |
| SRAD (100 iter) | 1.000 | 191.5 | 0.748 | 30.2 |

SRAD iteration sweep (dt = 0.2, 10 realizations): FOM rises from 0.605 at 25 iterations to
about 0.77 at 150 to 200 iterations. At matched smoothing (about 75 iterations, ENL 131) SRAD
scores FOM 0.73 against 0.89 for Frost at ENL 133, so the failed criterion is not only a
tuning effect.

**Reading the result.** Mean preservation, variance reduction and the advantage over
Perona-Malik are reproduced. The edge-localization advantage over Lee, Frost and Kuan is **not**
reproduced in this setup. This does not show the paper is wrong: the paper's test images,
edge metric and parameters differ from mine.

## Study 2: breast ultrasound (BUSI), 50 images

Dataset: Breast Ultrasound Images Dataset (Al-Dhabyani et al., *Data in Brief*, 2020), 50 images
sampled evenly across benign and malignant cases, resized to 256 x 256. BUSI has no clean
reference images, so only no-reference metrics are used.

| Method | ENL | EPI vs original | CNR (lesion vs ring) |
|--------|-----|-----------------|----------------------|
| Original | 159.0 | 1.000 | 1.027 |
| Lee | 433.0 | 0.898 | 1.071 |
| Frost | 493.0 | 0.194 | 1.146 |
| NLM | 293.2 | 0.896 | 1.042 |
| SRAD | 723.8 | 0.557 | 1.100 |

SRAD gave the strongest speckle suppression (ENL about 4.5x the original) and a modest lesion
contrast gain (CNR +7%), while keeping more edge structure than Frost. Differences in lesion
contrast between methods are small.

## Limitations

- The paper's exact simulated images and metric definitions were not available; the phantom and
  Pratt FOM here are my own setup, so numbers are not directly comparable to the paper.
- The edge metric was revised once after a first run (Canny sigma = 2 hid differences between
  filters, so all methods scored the same). The acceptance criteria were not changed.
- EPI on BUSI is measured against the noisy original, so part of its drop is noise removal.
- The simple Otsu-based lesion segmentation (Dice) did not separate the methods and is not
  reported. A trained segmentation model would be a better downstream test.
- No statistical significance tests were run on the BUSI results; 50 images is a small sample.
- Research code, not a validated medical device or clinical claim.

## Reproduce

```
pip install -r requirements.txt
python reproduce_paper.py                       # Study 1
python run_experiment.py --synthetic            # simple phantom check
python run_experiment.py --data_dir Dataset_BUSI_with_GT --n_images 50   # Study 2
```

BUSI is not included; download it from Kaggle ("Breast Ultrasound Images Dataset").

## Reference

Y. Yu and S. T. Acton, "Speckle reducing anisotropic diffusion," *IEEE Transactions on Image
Processing*, vol. 11, no. 11, pp. 1260-1270, 2002.
