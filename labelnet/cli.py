"""Command line interface for labelnet.

Subcommands:
    train    train a model and save its final predictions
    predict  run a checkpoint on a single road network image
    tune     grid-search learning rate and batch size
    compare  compare predictions of all intermediate checkpoints
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import Config


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="stackedhourglass", help="model name or prefix (default: stackedhourglass)")
    parser.add_argument("--data-name", default="combined", help="dataset sub-folder (default: combined)")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="dataset root (default: ./data)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | xpu")


def _config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        model_name=args.model,
        data_name=args.data_name,
        data_root=args.data_root,
        epochs=args.epochs,
        learning_rate=getattr(args, "lr", None),
        batch_size=getattr(args, "batch_size", None),
        number_of_data_pairs=getattr(args, "pairs", None),
        data_split_proportion=getattr(args, "split", 0.2),
        device=args.device,
    )


def _cmd_train(args: argparse.Namespace) -> None:
    from .evaluate.compare import load_test_set, predict_test_set, run_comparison, save_predictions
    from .train import train_model

    config = _config_from_args(args)
    result = train_model(config, intermediate_saves=not args.no_intermediate_saves)

    if args.compare:
        run_comparison(config)
    else:
        # Old main.py behaviour: save the final model's test-set predictions.
        final_name = result.checkpoints[-1].stem
        x_test, _ = load_test_set(config)
        prediction = predict_test_set(config, final_name, x_test)
        save_predictions(config, final_name, x_test, prediction)

    print(f"Last checkpoint: {result.checkpoints[-1]}")
    print(f"Best epoch: {result.best_epoch}")


def _cmd_predict(args: argparse.Namespace) -> None:
    import numpy as np
    from PIL import Image

    from .inference import infer_model_name, load_model, predict_image
    from .vectorize import bbox_transform, mask_to_polygons, write_vectors

    model_name = args.model or infer_model_name(args.checkpoint)
    model = load_model(model_name, args.checkpoint, device=args.device)

    image = Image.open(args.image)
    mask = predict_image(model, image, Config())

    if args.out:
        Image.fromarray((mask * 255).astype(np.uint8)).save(args.out)
        print(f"Saved prediction mask to {args.out}")

    if args.geojson:
        if not args.bbox:
            raise SystemExit("--geojson requires --bbox minx,miny,maxx,maxy")
        parts = [float(v) for v in args.bbox.split(",")]
        if len(parts) != 4:
            raise SystemExit("--bbox must be minx,miny,maxx,maxy")
        transform = bbox_transform(*parts, mask.shape[1], mask.shape[0])
        polygons = mask_to_polygons((mask > args.threshold).astype(np.uint8), transform)
        write_vectors(args.geojson, polygons, None, value=[1] * len(polygons))
        print(f"Wrote {len(polygons)} polygons to {args.geojson}")


def _cmd_tune(args: argparse.Namespace) -> None:
    from .train import tune_model

    config = _config_from_args(args)
    tune_model(config, args.lrs, args.batch_sizes)


def _cmd_compare(args: argparse.Namespace) -> None:
    from .evaluate.compare import run_comparison

    run_comparison(_config_from_args(args))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="labelnet", description="Train and run road label placement models")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train a model and save its final predictions")
    _add_model_args(p)
    p.add_argument("--lr", type=float, default=None, help="learning rate (default: per-model tuned value)")
    p.add_argument("--batch-size", type=int, default=None, help="batch size (default: per-model tuned value)")
    p.add_argument("--pairs", type=int, default=None, help="number of image pairs (default: all on disk)")
    p.add_argument("--split", type=float, default=0.2, help="test split fraction (default: 0.2)")
    p.add_argument("--no-intermediate-saves", action="store_true", help="do not save per-tenth-epoch checkpoints")
    p.add_argument("--compare", action="store_true", help="after training, compare all epoch checkpoints")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("predict", help="run a checkpoint on a single road network image")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", default=None, help="model name (default: inferred from the checkpoint name)")
    p.add_argument("--image", type=Path, required=True, help="input road network image")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", type=Path, default=None, help="save the prediction mask as a PNG")
    p.add_argument("--bbox", default=None, help="minx,miny,maxx,maxy for georeferencing")
    p.add_argument("--threshold", type=float, default=0.5, help="mask threshold for the polygon output")
    p.add_argument("--geojson", type=Path, default=None, help="write the predicted polygons as GeoJSON")
    p.set_defaults(func=_cmd_predict)

    p = sub.add_parser("tune", help="grid-search learning rate and batch size")
    _add_model_args(p)
    p.add_argument("--lr", dest="lrs", nargs="+", type=float, required=True)
    p.add_argument("--batch-size", dest="batch_sizes", nargs="+", type=int, required=True)
    p.add_argument("--pairs", type=int, default=None)
    p.add_argument("--split", type=float, default=0.2)
    p.set_defaults(func=_cmd_tune)

    p = sub.add_parser("compare", help="compare predictions of all intermediate checkpoints")
    _add_model_args(p)
    p.set_defaults(func=_cmd_compare)

    args = parser.parse_args(argv)
    args.func(args)
