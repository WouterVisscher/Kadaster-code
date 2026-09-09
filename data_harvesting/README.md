# Data harvesting

This folder contains the toolchain used to collect (road network, label placement)
image pairs from the PDOK/BRT map. The collected dataset itself lives in the
separate [`Kadaster-data`](https://github.com/IlanWS/Kadaster-data) repository.

To collect new image pairs, serve the two map styles as a local WMS, record the
image requests QGIS makes while panning the canvas, and download the images.

## 1. Start the WMS server

The map styles in `map_config_files/` are served by the `pdok/mapserver` docker
image via `run_wms.sh`:

```bash
# Input images: full standard background map (road network)
./run_wms.sh map_config_files/brt-achtergrondkaart-standaard-alles-wit.map

# Target images: background map with only the marked labels
./run_wms.sh map_config_files/brt-achtergrondkaart-standaard-only-marked-labels.map
```

Note: `run_wms.sh` mounts `map_config_files/` into the container at `/srv/data`,
which is where the `.map` files expect `fonts.list` and the fonts to be. If the
image does not provide a default `example.conf`, add one to `map_config_files/`.

## 2. Record the image requests in QGIS

1. In QGIS, add a WMS layer pointing at `http://localhost/mapserver` with the
   style served in step 1.
2. Open View > Debugger/Developer > Requests and press **record**.
3. Run `qgis_pan_canvas.py` in the QGIS Python console (it pans a 25x25 grid in
   100 m steps, issuing one image request per position).
4. When it finishes, stop recording and download the request log as JSON.
   Note that QGIS limits the number of recorded requests (a few hundred), so
   repeat the pan for larger datasets.

The resulting file looks like `examples/zutphen-met-labels.json` (truncated to
two entries). Each entry's `URL` points at a rendered 512x512 image.

## 3. Download the images

With the WMS still running, run the downloader once per map style. Both runs use
the same request log, so the produced filenames match 1:1:

```bash
# With the road network WMS running:
python data_harvesting/download_images.py --json data/zutphen-met-labels.json --output data/roadnetwork

# With the labels WMS running:
python data_harvesting/download_images.py --json data/zutphen-met-labels.json --output data/labels
```

The images are named `image_<index>.jpg` by their position in the request log.

## Layout

```
data_harvesting/
├── README.md                  # this file
├── run_wms.sh                 # starts the local WMS (docker)
├── qgis_pan_canvas.py         # QGIS console script: pans the canvas to trigger requests
├── download_images.py         # downloads the recorded request URLs to disk
├── map_config_files/          # mapserver styles + fonts
└── examples/                  # example (truncated) QGIS request log
```
