"""labelnet: train and run neural networks that place road labels on maps.

The package is organised around the four concerns of the research pipeline:

- ``collect``   : harvest new road network / label image pairs (WMS + QGIS)
- ``data``      : load the image pairs and split them into train/test sets
- ``train``     : train a model and save checkpoints
- ``inference`` : run a trained model (single images, or as a service)
"""

__version__ = "0.1.0"
