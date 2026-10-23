# CatDiffusion — GitHub Copilot Implementation Specification

> **Role:** Act as a senior PyTorch engineer and technical educator. Implement a complete, runnable, beginner-friendly DDPM project from scratch. Create and edit the files, do not merely describe what to write. Work incrementally, run tests, fix failures, and report any unverified steps. Do not claim training or GPU validation without actually performing it.

## 1. Project objective

Build **CatDiffusion**, an **unconditional pixel-space Denoising Diffusion Probabilistic Model (DDPM)** trained exclusively on cat images. Users must be able to:

1. Clone the repository and train locally on CPU or CUDA GPU.
2. Open a Google Colab notebook, download the dataset, train, visualize progress, and generate cat images.
3. Load a compatible trained checkpoint to generate images without retraining (only if an actual checkpoint is supplied; never fabricate weights).
4. See each key equation explained next to the corresponding PyTorch code.
5. Reproduce figures suitable for an educational Medium article.

**Scope:** Implement the DDPM algorithm ourselves, not a wrapper around a pretrained Stable Diffusion or Hugging Face diffusion pipeline. Do not implement text conditioning, CLIP, classifier-free guidance, or latent diffusion in v1. Clearly distinguish this educational DDPM from Stable Diffusion in documentation.

## 2. Required repository structure

Create this structure (additional small files are allowed when justified):

```text
CatDiffusion/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── pyproject.toml
├── notebooks/
│   └── CatDiffusion_Tutorial.ipynb
├── catdiffusion/
│   ├── __init__.py
│   ├── model.py
│   ├── diffusion.py
│   ├── dataset.py
│   ├── training.py
│   ├── sampling.py
│   └── utils.py
├── scripts/
│   ├── train.py
│   ├── generate.py
│   └── visualize_forward.py
├── configs/
│   ├── cpu.yaml
│   └── colab.yaml
├── tests/
│   ├── test_diffusion.py
│   ├── test_model.py
│   ├── test_dataset.py
│   └── test_smoke.py
├── assets/
│   └── .gitkeep
├── checkpoints/
│   └── .gitkeep
└── outputs/
    └── .gitkeep
```

All Python packages should be importable when the project is installed with `pip install -e .`. Use a minimal `pyproject.toml`. Pin or constrain dependencies sensibly and document tested versions; do not pretend an untested dependency set is verified. Use PyTorch, torchvision, Pillow, matplotlib, tqdm, PyYAML, numpy, pytest, and notebook tooling as appropriate. Do not depend on `diffusers` for the DDPM implementation.

## 3. Dataset: Oxford-IIIT Pet, cats only

Use `torchvision.datasets.OxfordIIITPet` with automatic download, and **explicitly filter to the 12 cat breeds**, not all 37 breeds. Do not assume category indices are cats without verifying the torchvision class mapping. Prefer the official breed/category metadata; document the filtering strategy and test that dog breeds are excluded. Use `trainval` for training; keep `test` separate for optional inspection and never silently mix splits. Handle download errors with helpful messages.

Preprocessing:
- RGB images; resize and center-crop to 32×32 (CPU preset) or 64×64 (Colab preset).
- Convert to tensors and normalize from `[0,1]` to `[-1,1]`.
- Optional random horizontal flip for training.
- Use configurable `num_workers`; default to 0 on Windows/CPU for portability.
- Provide a reproducible random seed and a `--max-images` option for quick demos.
- Provide a utility to display a grid of real cat training images.

Dataset licensing/attribution: link to the Oxford-IIIT Pet dataset website, state its applicable terms, and do not commit the dataset into Git. Never claim the dataset is wholly original.

## 4. DDPM mathematical requirements

Use zero-based Python indexing `i = 0, ..., T-1` and document that index `i` corresponds to mathematical timestep `t = i+1`. Define:

\[
\beta_t \in (0,1),\qquad \alpha_t = 1-\beta_t,\qquad
\bar\alpha_t = \prod_{s=1}^{t}\alpha_s.
\]

**Forward process (direct sampling):**

\[
q(x_t\mid x_0) = \mathcal N\big(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I\big),
\qquad
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,
\quad \epsilon\sim\mathcal N(0,I).
\]

**Noise predictor and loss:**

\[
\hat\epsilon = \epsilon_\theta(x_t,t),\qquad
\mathcal L = \mathbb E_{x_0,t,\epsilon}\left[\|\epsilon-\epsilon_\theta(x_t,t)\|_2^2\right].
\]

In code, use `F.mse_loss` (mean over tensor elements); explain this normalization convention.

**Reverse DDPM sampler:**

\[
\mu_\theta(x_t,t) = \frac{1}{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right),
\]

\[
\tilde\beta_t = \beta_t\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t},
\qquad x_{t-1}=\mu_\theta(x_t,t)+\sqrt{\tilde\beta_t}\,z,
\quad z\sim\mathcal N(0,I).
\]

At the final reverse step, **do not add noise**. Use `\bar\alpha_0=1` in the mathematical derivation; implement the first-step boundary correctly with zero-based arrays. Use posterior variance `\tilde\beta_t` for the educational sampler and document the choice. Clamp only the final displayed image to `[-1,1]` before converting to `[0,1]`; do not silently clamp every intermediate state. Use `torch.no_grad()` or `torch.inference_mode()` for sampling.

**Noise schedule caveat:** A linear beta schedule from `1e-4` to `0.02` over only 100 steps does **not** necessarily produce a near-Gaussian terminal state. Implement a configurable cosine cumulative-alpha schedule (with numerical clipping and valid beta values) as the **default** for CPU/short-step demos, and optionally a linear schedule for comparison. Include a test that the configured terminal `\bar\alpha_T` is sufficiently small for the intended sampling assumption, or explicitly warn when it is not. Never claim that simply reducing T speeds up each training batch proportionally; it mainly reduces reverse-sampling iterations.

Implementation requirements for `catdiffusion/diffusion.py`:
- A documented `DDPM` or `DiffusionSchedule` class that registers/prepares tensors on the correct device.
- `q_sample(x0, timesteps, noise=None)` returning `(xt, noise)`.
- A safe coefficient extraction helper that broadcasts `(B,)` coefficients to `(B,C,H,W)`.
- `p_sample(model, xt, t)` and `sample(model, shape, device, seed=None, capture_steps=None)`.
- Clear handling of tensor dtype, device, timestep bounds, and shapes.
- Optional capture of intermediate reverse states for a visualization grid, without retaining all steps in GPU memory unnecessarily.

## 5. Beginner-friendly timestep-conditioned U-Net

Implement in `catdiffusion/model.py`:
- Small U-Net operating on RGB tensors `(B,3,H,W)`.
- Configurable `base_channels` (e.g. 16 for CPU, 32 or 64 for GPU).
- Sinusoidal timestep embeddings plus a small MLP.
- Residual convolutional blocks that inject timestep embeddings.
- Downsampling, bottleneck, upsampling, and skip connections.
- Correct spatial dimensions for both 32×32 and 64×64 inputs.
- Output shape identical to input shape and 3 output channels for RGB noise prediction.
- Avoid pretrained weights or hidden model downloads.
- Explain every module with short docstrings, shape comments, and a README diagram.
- Keep architecture simple enough to teach; no unnecessary attention modules in v1.

## 6. Training implementation

Implement `catdiffusion/training.py` and `scripts/train.py` with:
- Device selection: CUDA if available, otherwise CPU; explicit `--device cpu|cuda|auto` override.
- CPU preset: image size 32, batch size 4 or 8, base channels 16, small subset and few epochs for a functional demo.
- Colab preset: image size 64, batch size around 32 (adjustable), base channels 32 or 64, longer training.
- Adam or AdamW optimizer with configurable learning rate (e.g. 2e-4).
- Random timestep independently sampled for every image in each batch.
- `torch.randn_like(x0)` noise; direct `q_sample`; model predicts noise; MSE; backward; optimizer step.
- `tqdm` progress; epoch and step metrics; CSV log of losses.
- Seed Python, NumPy, and PyTorch; note remaining nondeterminism.
- Save checkpoints at configurable intervals and at the end, including model weights, optimizer state, epoch/global step, model config, diffusion config, and normalization convention.
- Resume training from a checkpoint, validating configuration compatibility.
- Periodically save generated image grids (not too often; sampling is expensive).
- Keep checkpoint loading safe: use `torch.load(..., map_location=device, weights_only=True)` when supported, and store only primitive dictionaries/tensors. Document trusted-checkpoint requirements.
- Ensure training remains useful on CPU, but do not promise recognizable cats from a tiny quick demo.
- Include `--dry-run` or `--max-steps` for fast smoke tests.

## 7. Image generation CLI

Implement `catdiffusion/sampling.py` and `scripts/generate.py`:
- `python scripts/generate.py --checkpoint checkpoints/model.pt --count 16 --device auto --seed 42`
- Reconstruct model and schedule **from checkpoint metadata**, not from incompatible hardcoded defaults.
- Generate a batch of new samples from `x_T ~ N(0,I)`.
- Save PNG grid and optionally individual PNGs in `outputs/`.
- Optional reverse diffusion progression figure showing intermediate denoising states.
- Print the checkpoint, device, image shape, and sampling configuration used.
- Fail with a helpful error if the checkpoint is missing; never silently substitute random untrained weights.

## 8. Visualization utilities

Implement reproducible figures, each saved as a PNG under `outputs/`:
1. `real_cats.png` — grid of training images.
2. `forward_diffusion.png` — same cat at 5 or more noise levels, with timestep labels.
3. `noise_schedule.png` — plot of beta and cumulative alpha (separate labeled curves or panels).
4. `training_loss.png` — loss versus training step.
5. `reverse_diffusion.png` — intermediate reverse states for the same sampled trajectory.
6. `generated_cats.png` — final sample grid.

Figures must come from actual code execution; do not insert fabricated training results or AI-generated images as experimental outputs. In captions, distinguish illustrative visualizations from measured results.

## 9. Colab notebook: primary reader experience

Create a **valid, runnable** `notebooks/CatDiffusion_Tutorial.ipynb` with Markdown explanations, rendered LaTeX equations, and executable cells. Use imports from the package rather than duplicating model/sampler code in the notebook.

Notebook sequence:
1. Title, learning objectives, DDPM vs Stable Diffusion disclaimer.
2. Check Python, PyTorch, CUDA and available memory; select CPU/GPU preset automatically with user override.
3. Install project from GitHub using a placeholder `<GITHUB_USERNAME>/<REPOSITORY>` or clone current repository; clearly mark the one URL readers must update after publishing. Support local Jupyter too.
4. Download/filter Oxford-IIIT Pet to cat images; display sample grid.
5. Explain forward diffusion and show actual noisy cat examples.
6. Explain and plot noise schedule.
7. Explain and instantiate timestep-conditioned U-Net; print parameter count and input/output shapes.
8. Explain the noise prediction loss and train with a clearly labeled **Quick Demo** or **Full Training** switch.
9. Plot real training loss.
10. Save checkpoints to local Colab storage, optionally copy to Google Drive with user consent and explicit mount instructions.
11. Generate new cats using current checkpoint; show image grid and reverse diffusion progression.
12. Optional load-pretrained-checkpoint cell: use a documented user-supplied/release URL only if an actual checkpoint exists; otherwise show a clear placeholder and skip cleanly.
13. Summarize the three core equations and provide suggested experiments.

Colab constraints:
- Do not assume a GPU is always available.
- Do not hardcode `/content` paths without a fallback for local Jupyter.
- Avoid unnecessary repeated dataset downloads.
- Be honest about free-tier limits and possible disconnections.
- Add a working “Open in Colab” badge to README after the repository URL is known; leave a clearly labeled placeholder until then.
- Ensure all notebook cells execute top-to-bottom in a fresh session when configured with a valid repository URL.

## 10. Configuration files

Use readable YAML configurations with comments. Example defaults (adjust only with clear reasoning):

**`configs/cpu.yaml`**: 32×32, batch size 4, base channels 16, 100 diffusion steps, cosine schedule, 2 demo epochs, subset of 256–512 cat images, 0 workers. This is a correctness/demo preset, **not** a promise of recognizable images.

**`configs/colab.yaml`**: 64×64, batch size 32, base channels 32, 300 diffusion steps, cosine schedule, configurable epochs (e.g. 50), entire cat training split, 2 workers. This is a starting point, not a guaranteed training duration or quality level.

Make it possible to override important values from the CLI (`--epochs`, `--batch-size`, `--max-images`, `--image-size`, `--max-steps`, `--device`, `--seed`, `--output-dir`). Avoid conflicting sources of truth; log the resolved config.

## 11. Testing and acceptance criteria

Use `pytest` and include tests that run quickly on CPU without downloading the full dataset:

- Noise schedule: betas in `(0,1)`, monotonic cumulative alpha, correct shapes, finite values, near-noise terminal state for the chosen demo schedule.
- `q_sample`: at a fixed timestep and supplied noise, matches the mathematical formula numerically.
- U-Net: output shape equals input shape for 32×32 and 64×64, finite outputs, successful backward pass.
- Reverse sampler: returns correct shape, finite values, deterministic results under fixed seed and same device/config; final step adds no fresh noise.
- Dataset: cat-only filter verified with synthetic breed metadata or a small fixture, no dog category leaks; transforms yield `(3,H,W)` tensors in `[-1,1]`.
- Smoke test: one optimizer step on synthetic RGB images and one short sampling run with a tiny model.
- Checkpoint round-trip: save/load metadata and weights, reconstruct compatible model, sample successfully.
- CLI `--help` works for training, generation, and forward visualization.
- Notebook file is valid JSON and passes `nbformat` validation.

**Do not** make CI depend on full training, dataset downloads, a CUDA GPU, or a paid service. Run tests locally and fix failures. If GPU/Colab cannot be tested in this environment, state that explicitly.

## 12. README quality bar

Write an accessible README with:
- One-paragraph project motivation and clear disclaimer that this is **not full Stable Diffusion**.
- Architecture/training diagram (Mermaid is fine).
- Install instructions (Python version, `pip install -e .`).
- Quick CPU smoke-test commands.
- Full training commands for Colab/local CUDA.
- Generate-from-checkpoint command.
- Colab notebook link/badge placeholder until repository is published.
- Dataset source and attribution.
- Equation-to-code mapping table.
- Expected limitations, realistic training expectations, and troubleshooting (no GPU, OOM, missing checkpoint, Colab timeout).
- Reproducibility details (seed, model configuration, checkpoint metadata).
- Explicit note that `assets/` images should be produced by real experiments.
- Citations: Ho et al. 2020 DDPM; Rombach et al. 2022 Latent Diffusion; optional Song et al. 2021 SDE.
- License file: use MIT unless the repository owner requests otherwise; distinguish code license from dataset terms.

## 13. Copilot implementation workflow

Implement in this order, completing files rather than leaving pseudocode or TODOs:

1. Project scaffolding, package metadata, dependencies, `.gitignore`, configurations.
2. Dataset loader and cat-only filter; unit tests.
3. Diffusion schedule, forward process, reverse sampler; mathematical tests.
4. U-Net and shape/backprop tests.
5. Training and checkpoint logic; synthetic smoke test.
6. Generation and visualization CLIs.
7. Colab notebook using the package modules.
8. README, LICENSE, notebook validation, final tests.

After each phase, run relevant tests and fix issues. At the end, report:
- Files created/modified.
- Exact commands tested and their results.
- Which checks were not run (especially full training and Colab GPU).
- How to run a CPU smoke test, train in Colab, and generate images.
- Any placeholders the repository owner must replace (GitHub URL, pretrained checkpoint URL).

**Definition of done:** The repository is usable end-to-end with no fake checkpoints, no invented result figures, no unimplemented critical functions, and no dependency on pretrained diffusion pipelines. A fresh user can follow the README or Colab notebook to train a tiny model and sample images; high-quality cat generation may require longer GPU training.
