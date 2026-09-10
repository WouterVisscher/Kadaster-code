#!/usr/bin/env bash
# Fetch the Kadaster-data dataset into ./data.
#
# Expects a checkout of the Kadaster-data repository next to this one by
# default; override with KADASTER_DATA_SOURCE=/path/to/Kadaster-data.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="${KADASTER_DATA_SOURCE:-$ROOT/../Kadaster-data}"

if [[ ! -d "$SOURCE" ]]; then
    echo "error: Kadaster-data not found at: $SOURCE" >&2
    echo "hint:  git clone https://github.com/WouterVisscher/Kadaster-data.git" >&2
    echo "       or set KADASTER_DATA_SOURCE to an existing checkout" >&2
    exit 1
fi

mkdir -p "$ROOT/data"
for dir in roadnetwork labels test ground_truth json_files outputs; do
    rm -rf "$ROOT/data/$dir"
    cp -r "$SOURCE/$dir" "$ROOT/data/$dir"
done

echo "dataset ready in $ROOT/data"
