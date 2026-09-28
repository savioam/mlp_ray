"""Hyperparameter search for the FashionMNIST MLP, using Ray Tune.

Same model and data pipeline as notebooks/01_first_neural_network.ipynb
(see model.py) - this script just replaces "pick hidden_size/lr/batch_size
by hand" with a search across many configs, early-stopping the bad ones.

Usage:
    python train_ray_tune.py                          # default search
    python train_ray_tune.py --num-samples 4 --max-epochs-per-trial 2   # quick smoke test
    python train_ray_tune.py --gpus-per-trial 0.5      # run 2 trials at once on 1 GPU

See README.md for setup and a full walkthrough of the output.
"""

import argparse
import json
import os

import torch
import torch.nn as nn
from ray import tune
from ray.tune import RunConfig
from ray.tune.schedulers import ASHAScheduler

from model import MLP, load_data, train_epoch, validate_epoch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def trainable(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader, _ = load_data(batch_size=config["batch_size"])

    model = MLP(hidden_size=config["hidden_size"]).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    for _ in range(config["max_epochs_per_trial"]):
        train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer, device)
        valid_loss, valid_acc = validate_epoch(model, valid_loader, loss_fn, device)
        # Reported once per epoch: ASHA uses this stream to kill underperforming
        # trials early rather than waiting for every trial to finish all epochs.
        tune.report({
            "valid_acc": valid_acc,
            "valid_loss": valid_loss,
            "train_acc": train_acc,
            "train_loss": train_loss,
        })


def run_search(args):
    search_space = {
        "hidden_size": tune.choice([128, 256, 512, 1024]),
        "lr": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([64, 128, 256]),
        "max_epochs_per_trial": args.max_epochs_per_trial,
    }

    scheduler = ASHAScheduler(
        metric="valid_acc",
        mode="max",
        max_t=args.max_epochs_per_trial,
        grace_period=1,  # let every trial run at least 1 epoch before it can be stopped early
    )

    tuner = tune.Tuner(
        tune.with_resources(
            trainable,
            resources={"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial},
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=args.num_samples,
            scheduler=scheduler,
        ),
        run_config=RunConfig(
            name="fashion_mnist_mlp",
            storage_path=os.path.abspath(args.storage_path),
            verbose=1,
        ),
    )

    results = tuner.fit()
    best_result = results.get_best_result(metric="valid_acc", mode="max")
    return best_result


def retrain_best(best_config, epochs, output_dir):
    """Re-train one model with the winning config for longer, and save it the
    same way the notebook does: weights + class names, ready for inference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader, classes = load_data(batch_size=best_config["batch_size"])

    model = MLP(hidden_size=best_config["hidden_size"]).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=best_config["lr"])

    print(f"\nRe-training best config for {epochs} epochs...")
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer, device)
        valid_loss, valid_acc = validate_epoch(model, valid_loader, loss_fn, device)
        print(
            f"Epoch {epoch + 1}/{epochs}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
            f"valid_loss={valid_loss:.4f} valid_acc={valid_acc:.4f}"
        )

    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, "best_mlp_fashion_ray_tune.pt")
    classes_path = os.path.join(output_dir, "mlp_fashion_classes.json")
    torch.save(model.state_dict(), weights_path)
    with open(classes_path, "w") as f:
        json.dump(classes, f)
    print(f"Saved weights to {weights_path} and class names to {classes_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-samples", type=int, default=8, help="number of hyperparameter configs to try")
    p.add_argument("--max-epochs-per-trial", type=int, default=4, help="epoch budget per trial during search")
    p.add_argument("--cpus-per-trial", type=float, default=2.0)
    p.add_argument("--gpus-per-trial", type=float, default=1.0, help="fraction of a GPU per trial (e.g. 0.5 to run 2 trials at once on 1 GPU)")
    p.add_argument("--storage-path", default=os.path.join(THIS_DIR, "ray_results"))
    p.add_argument("--final-epochs", type=int, default=10, help="epochs used to re-train the winning config")
    p.add_argument("--output-dir", default=THIS_DIR)
    p.add_argument("--skip-retrain", action="store_true", help="only run the search, don't retrain/save a final model")
    return p.parse_args()


def main():
    args = parse_args()

    best_result = run_search(args)

    print("\n" + "=" * 60)
    print("Best config found:")
    for key, value in best_result.config.items():
        print(f"  {key}: {value}")
    print(f"  -> valid_acc: {best_result.metrics['valid_acc']:.4f}")
    print("=" * 60)

    if not args.skip_retrain:
        retrain_best(best_result.config, args.final_epochs, args.output_dir)


if __name__ == "__main__":
    main()
