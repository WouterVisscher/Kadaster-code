#!/usr/bin/env bash
# Run the PDOK mapserver container as a local WMS for image harvesting.
#
# Usage:
#   ./run_wms.sh map_config_files/brt-achtergrondkaart-standaard-alles-wit.map   # input (road network)
#   ./run_wms.sh map_config_files/brt-achtergrondkaart-standaard-only-marked-labels.map   # target (labels)
set -euo pipefail

MAPFILE="${1:?Usage: $0 <map-file under map_config_files/>}"

if [[ "$MAPFILE" != map_config_files/*.map ]]; then
    echo "Expected a path under map_config_files/, got: $MAPFILE" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker run \
    -e MAPSERVER_CONFIG_FILE=/srv/data/example.conf \
    -e MS_MAPFILE="/srv/data/$(basename "$MAPFILE")" \
    -e SERVICE_TYPE=WMS \
    --rm -p 80:80 --name mapserver-example \
    -v "$SCRIPT_DIR/map_config_files":/srv/data \
    pdok/mapserver
