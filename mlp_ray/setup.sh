#!/usr/bin/env bash
# Self-contained install: creates a venv inside this directory and installs
# everything train_ray_tune.py needs (torch, torchvision, ray[tune]) on cuda enabled machine.
# If machine doesnt have cuda
# 	If using uv , edit pyproject.toml to remove cuda references
# 	if using pip, edit requirements.txt to remove cuda references
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if command -v uv >/dev/null 2>&1; then
    uv sync
else
    echo "uv not found - falling back to python3 -m venv + pip." >&2
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
fi

echo ""
echo "Done. Activate the venv with:"
echo "  source $(pwd)/.venv/bin/activate"
echo ""
echo "Train (runs the Ray Tune search, then retrains + saves the best config):"
echo "  python train_ray_tune.py"
echo "  python train_ray_tune.py --num-samples 4 --max-epochs-per-trial 2   # quick smoke test (~1-2 min)"
echo "  python train_ray_tune.py --gpus-per-trial 0.5                       # run 2 trials at once on 1 GPU"
echo "  python train_ray_tune.py --skip-retrain                             # search only, don't save a model"
echo "  python train_ray_tune.py --help                                     # full list of flags"
echo ""
echo "Model: a 3-layer fully-connected MLP (784 -> hidden_size -> hidden_size -> 10),"
echo "trained to classify FashionMNIST images into 10 clothing categories. hidden_size"
echo "(128/256/512/1024) is whatever Ray Tune picked as the best config for this checkpoint."
echo ""
echo "Classify an image with the saved model (best_mlp_fashion_ray_tune.pt + mlp_fashion_classes.json):"
echo "  python classify.py --image sample_images/7_Sneaker.png"
echo "  python classify.py --image path/to/your_image.png"
echo "  python classify.py --help                             # weights/classes path overrides"
