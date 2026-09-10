#!/usr/bin/env bash
# Start a local WMS instance serving one of the mapserver map files.
# Usage: ./run_wms.sh <map-file> [port]
#
# Road network:   ./run_wms.sh brt-achtergrondkaart-standaard-weg-met-rand.map
# Label placement: ./run_wms.sh brt-achtergrondkaart-standaard-only-marked-labels.map 8080
set -euo pipefail

MAPFILE="${1:?usage: run_wms.sh <map-file> [port]}"
PORT="${2:-80}"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker run \
  -e MAPSERVER_CONFIG_FILE=/srv/data/example.conf \
  -e MS_MAPFILE="/srv/data/${MAPFILE}" \
  -e SERVICE_TYPE=WMS \
  --rm \
  -p "${PORT}:80" \
  --name "mapserver-$(basename "${MAPFILE}" .map)" \
  -v "${DIR}/map_config_files:/srv/data" \
  pdok/mapserver
