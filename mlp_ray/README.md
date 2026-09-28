# FashionMNIST MLP — Ray Tune hyperparameter search

An MLP (multi-layer perceptron) trained to classify FashionMNIST images (10 clothing
categories), with [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) searching over
`hidden_size`, `lr`, and `batch_size` instead of picking them by hand, using
[ASHA](https://docs.ray.io/en/latest/tune/api/schedulers.html) to kill the worst-performing
runs early instead of letting every one finish all its epochs.

This directory is self-contained: copy just `mlp_ray/` to a new machine and it can install its
own dependencies, train, and classify — no need for anything else from this repo.

## Files

| File | Role |
|---|---|
| `model.py` | The `MLP` architecture, FashionMNIST data loading, and one-epoch train/validate functions. Shared by the search, the final retrain, and `classify.py`, so all three use the exact same model. |
| `train_ray_tune.py` | Runs the Ray Tune search, prints the best config found, then re-trains that config for longer and saves it. |
| `classify.py` | Pure-inference script: loads a saved checkpoint + class names and classifies one image, no training code involved. |
| `pyproject.toml` / `requirements.txt` | This directory's own dependency set (`torch`, `torchvision`, `ray[tune]`) — independent of anything else on the machine. |
| `setup.sh` | One-command install into a local `.venv` right here (see below). |
| `sample_images/` | One demo PNG per FashionMNIST class (pulled from the validation set) for trying `classify.py` immediately. |

## Setup

```bash
./setup.sh
```

That creates `mlp_ray/.venv` and runs `uv sync` inside it (or falls back to
`python3 -m venv` + `pip install -r requirements.txt` if `uv` isn't installed).

`pyproject.toml` and `requirements.txt` both pin a CUDA PyTorch wheel index. On a machine with
a different CUDA version (or none): edit the index URL to match
([pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)), or drop the
`[tool.uv.sources]`/`[[tool.uv.index]]` blocks (and the `--extra-index-url` line in
`requirements.txt`) entirely to fall back to the default CPU-only build.

## Data

FashionMNIST (~30MB) downloads automatically via `torchvision` on first run, into a local
`data/` folder inside `mlp_ray/` — independent of any dataset that may already exist elsewhere
on the machine. No manual dataset/image setup needed.

## Train

```bash
source .venv/bin/activate     # if not already active
python train_ray_tune.py
```

This runs 8 trials (random `hidden_size` in `{128, 256, 512, 1024}`, `lr` in `[1e-4, 1e-2]`,
`batch_size` in `{64, 128, 256}`), up to 4 epochs each with early stopping, on the machine's
GPU — then re-trains the best config for 10 more epochs and saves it as
`best_mlp_fashion_ray_tune.pt` + `mlp_fashion_classes.json`.

Useful flags:

```bash
python train_ray_tune.py --num-samples 4 --max-epochs-per-trial 2   # quick smoke test (~1-2 min)
python train_ray_tune.py --gpus-per-trial 0.5                       # run 2 trials at once on 1 GPU
python train_ray_tune.py --skip-retrain                             # search only, don't save a model
python train_ray_tune.py --help                                     # full list of flags
```

## Classify an image

Once training has produced `best_mlp_fashion_ray_tune.pt` + `mlp_fashion_classes.json`:

```bash
python classify.py --image sample_images/7_Sneaker.png
python classify.py --image path/to/your_image.png
python classify.py --help                             # weights/classes path overrides
```

`classify.py` prints the model architecture (3-layer MLP, `hidden_size` read straight from the
checkpoint's weight shapes — whatever Ray Tune picked, not hardcoded), the predicted class, and
the full probability distribution over all 10 classes. Any image works: it's converted to
grayscale and resized to 28x28 before classification.

## Output

- Console: a live-updating table of trials during search, then the winning config and its
  validation accuracy.
- `ray_results/` — full per-trial logs and metrics (gitignored). View them with:
  ```bash
  tensorboard --logdir ray_results
  ```
- `best_mlp_fashion_ray_tune.pt` + `mlp_fashion_classes.json` — the retrained winning model:
  weights only (`state_dict`) plus the class-name list, ready for `classify.py`.
