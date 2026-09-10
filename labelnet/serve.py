"""Run a trained model as a prediction service (FastAPI).

Start the service with:

    labelnet serve --checkpoint checkpoints/stackedhourglass/stackedhourglass_100.pth

and predict on a road network image with:

    curl -F image=@road.jpg http://localhost:8000/predict -o prediction.png

Add ``-F bbox=minx,miny,maxx,maxy -F geojson=true`` to get the predicted
label polygons back as GeoJSON instead of a mask PNG.
"""

from __future__ import annotations

import argparse
import io
import json
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from shapely.geometry import mapping

from . import inference, vectorize
from . import __version__
from .config import Config


def create_app(checkpoint: Path, model_name: str | None = None, device: str = "auto") -> FastAPI:
    """Build the FastAPI app around the checkpoint at ``checkpoint``."""
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resolved_name = model_name or inference.infer_model_name(checkpoint)
        state["model"] = inference.load_model(resolved_name, checkpoint, device)
        state["model_name"] = resolved_name
        state["config"] = Config()
        yield
        state.clear()

    app = FastAPI(title="labelnet", version=__version__, lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": state.get("model_name")}

    @app.post("/predict")
    async def predict(
        image: UploadFile = File(...),
        bbox: str | None = Form(None),
        threshold: float = Form(0.5),
        geojson: bool = Form(False),
    ) -> Response:
        model = state.get("model")
        if model is None:
            raise HTTPException(status_code=503, detail="Model is still loading")
        config = state["config"]

        pil_image = Image.open(io.BytesIO(await image.read()))
        mask = inference.predict_image(model, pil_image, config)

        if not geojson:
            buffer = io.BytesIO()
            Image.fromarray((mask * 255).astype(np.uint8)).save(buffer, format="PNG")
            return Response(buffer.getvalue(), media_type="image/png")

        if not bbox:
            raise HTTPException(status_code=400, detail="bbox=minx,miny,maxx,maxy is required for the GeoJSON output")
        parts = [float(v) for v in bbox.split(",")]
        if len(parts) != 4:
            raise HTTPException(status_code=400, detail="bbox must be minx,miny,maxx,maxy")
        transform = vectorize.bbox_transform(*parts, mask.shape[1], mask.shape[0])
        polygons = vectorize.mask_to_polygons((mask > threshold).astype(np.uint8), transform)
        features = [
            {"type": "Feature", "geometry": mapping(poly), "properties": {"value": 1}}
            for poly in polygons
        ]
        return Response(
            json.dumps({"type": "FeatureCollection", "features": features}),
            media_type="application/geo+json",
        )

    return app


def serve(checkpoint: Path, model_name: str | None = None, device: str = "auto", host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the prediction service (blocking)."""
    import uvicorn

    app = create_app(checkpoint, model_name, device)
    uvicorn.run(app, host=host, port=port)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a labelnet checkpoint as a prediction service")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", default=None, help="model name (default: inferred from the checkpoint name)")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | xpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    serve(args.checkpoint, model_name=args.model, device=args.device, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
