# Data collection

The tools in this package harvest new road network / label image pairs
from local WMS instances that serve the Brtachtergrondkaart mapserver map
files.

## 1. Start the WMS instances

Two instances are needed: the road network (port 80) and the label
placement (port 8080), each in its own terminal:

```
./run_wms.sh brt-achtergrondkaart-standaard-weg-met-rand.map
./run_wms.sh brt-achtergrondkaart-standaard-only-marked-labels.map 8080
```

Both serve the map files in `map_config_files/` via docker (`pdok/mapserver`).

## 2. Record the WMS requests in QGIS

Add the WMS layer to QGIS, then run `qgis_pan.py` in the QGIS Python
console (Plugins > Python Console) with the developer tools recording
(View > Tools > Developer > Requests). When the pan has finished, stop
recording and download the requests log to `data/json_files/<name>.json`.
Note that the log has a limit on the number of stored requests, so for
larger areas the pan has to be repeated and the logs combined.

## 3. Download the images

With the road network WMS running:

```
labelnet collect download --json data/json_files/<name>.json --out data/roadnetwork/<name>
```

and, with the label WMS running, the same for the labels into
`data/labels/<name>`.

## 4. Extract the label masks (rvimage)

When the labels were traced in rvimage instead of recorded from the WMS,
render the annotation JSON into label mask images:

```
labelnet collect extract --json data/json_files/<name>.json --out data/labels/<name>
```

Per-city label folders can then be combined into one training folder,
renaming the images to `image_0`, `image_1`, ...:

```
labelnet collect combine --source data/labels --out data/labels/combined \
    zutphen_1000 deventer_2500_land deventer_2500_stad \
    apeldoorn_5000_land apeldoorn_5000_stad \
    --start deventer_2500_land=300 --start deventer_2500_stad=525 \
    --start apeldoorn_5000_land=750 --start apeldoorn_5000_stad=975
```
