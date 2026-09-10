"""Tools to collect new road network / label image pairs.

- ``qgis_pan.py``        : pan the QGIS canvas to record WMS requests (run in the QGIS console)
- ``download_images.py`` : download the images of a WMS request log to disk
- ``extract_labels.py``  : render rvimage annotation JSON as label masks
- ``run_wms.sh``         : start the local WMS instances (docker)
- ``map_config_files/``  : mapserver map files and fonts for the WMS
"""
