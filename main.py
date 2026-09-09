"""Entry point: train the label placement model and visualize results.

Examples:
    python main.py                        # train, then plot input/prediction/target
    python main.py --epochs 5 --device cpu
    python main.py --checkpoint checkpoints/labelnet_last.pt   # skip training
"""

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt

from labelnet.config import Config
from labelnet.data import load_dataset, train_test_split
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
    parser.add_argument("--no-show", action="store_true", help="save evaluation.png instead of opening a plot window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
