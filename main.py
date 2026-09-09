"""Entry point: train the label placement model, evaluate it, and run it on single images.

Examples:
    python main.py                        # train, then plot input/prediction/target
    python main.py --epochs 5 --device cpu
    python main.py --checkpoint checkpoints/labelnet_last.pt   # skip training
    python main.py --checkpoint checkpoints/labelnet_last.pt \
                   --image road.jpg --out placement.png        # inference on one image
"""

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from labelnet.config import Config
from labelnet.data import load_dataset, load_image, train_test_split
from labelnet.evaluate import predict, visualize_results
from labelnet.train import get_device, load_model, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate the road label placement U-Net.")
    parser.add_argument("--data-dir", type=Path, default=None, help="folder containing roadnetwork/ and labels/ (default: ./data)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--checkpoint", type=Path, default=None, help="evaluate an existing checkpoint instead of training")
    parser.add_argument("--image", type=Path, default=None, help="run inference on a single road network image (requires --checkpoint)")
    parser.add_argument("--out", type=Path, default=Path("prediction.png"), help="output mask for --image (default: prediction.png)")
    parser.add_argument("--no-show", action="store_true", help="save evaluation.png instead of opening a plot window")
    return parser.parse_args()


def run_single_image(args: argparse.Namespace) -> None:
    """Run the model on one road network image and save a binarized placement mask."""
    config = Config()
    device = get_device(args.device)
    model = load_model(args.checkpoint, device)
    image = load_image(args.image, config.image_size)
    predictions = predict(model, image[np.newaxis, ...], device)
    mask = (predictions[0] >= 0.5).astype(np.uint8) * 255
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(args.out)
    print(f"Saved placement mask to {args.out}")


def main() -> None:
    args = parse_args()
    if args.image is not None:
        if args.checkpoint is None:
            raise SystemExit("--image requires --checkpoint")
        run_single_image(args)
        return
    config = Config()
    overrides = {
        "data_dir": args.data_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    }
    config = replace(config, **{k: v for k, v in overrides.items() if v is not None})

    inputs, targets = load_dataset(config.input_dir, config.target_dir, config.image_size)
    x_train, y_train, x_test, y_test = train_test_split(inputs, targets, config.test_fraction, config.seed)
    print(f"Loaded {len(inputs)} image pairs: {len(x_train)} train / {len(x_test)} test")

    device = get_device(args.device)
    if args.checkpoint:
        model = load_model(args.checkpoint, device)
    else:
        model = train_model(config, x_train, y_train, x_test, y_test, device=device)

    predictions = predict(model, x_test, device)
    fig = visualize_results(x_test, predictions, y_test)
    if args.no_show:
        fig.savefig("evaluation.png", dpi=100)
        print("Saved evaluation.png")
    else:
        plt.show()


if __name__ == "__main__":
    main()
