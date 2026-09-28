"""Classify a single image with a saved FashionMNIST MLP checkpoint.

Pure-inference script: loads only the saved weights + class names from disk,
no dataset or training code involved. Same shape as the notebook's "Reload
and classify a new image" cell (see notebooks/01_first_neural_network.ipynb).

Usage:
    python classify.py --image sample_images/7_Sneaker.png
    python classify.py --image path/to/photo.png --weights best_mlp_fashion_ray_tune.pt
"""

import argparse
import json
import os

import torch
from PIL import Image
from torchvision import transforms

from model import MLP

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def load_model(weights_path, classes_path):
    with open(classes_path) as f:
        classes = json.load(f)

    state = torch.load(weights_path, map_location="cpu")
    hidden_size = state["fc1.weight"].shape[0]  # whatever Ray Tune picked, not hardcoded

    model = MLP(hidden_size=hidden_size)
    model.load_state_dict(state)
    model.eval()
    return model, classes, hidden_size


def classify(model, classes, image_path):
    img = Image.open(image_path).convert("L").resize((28, 28))
    x = transforms.ToTensor()(img).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze()
        predicted_idx = logits.argmax(1).item()

    return predicted_idx, probs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True, help="path to the image to classify")
    p.add_argument("--weights", default=os.path.join(THIS_DIR, "best_mlp_fashion_ray_tune.pt"))
    p.add_argument("--classes", default=os.path.join(THIS_DIR, "mlp_fashion_classes.json"))
    return p.parse_args()


def main():
    args = parse_args()
    model, classes, hidden_size = load_model(args.weights, args.classes)
    num_params = sum(p.numel() for p in model.parameters())

    print(f"Model: MLP (3 fully-connected layers, hidden_size={hidden_size}, {num_params:,} params)")
    print(f"Weights: {args.weights}")
    print(f"Image: {args.image}")

    predicted_idx, probs = classify(model, classes, args.image)

    print(f"\nPredicted class: {classes[predicted_idx]}  (index {predicted_idx})")
    print(f"Confidence: {probs[predicted_idx].item():.4f}")
    print("\nFull class probabilities:")
    for i, c in enumerate(classes):
        print(f"  {c:15s} {probs[i].item():.4f}")


if __name__ == "__main__":
    main()
