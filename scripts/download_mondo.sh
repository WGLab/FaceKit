#!/usr/bin/env bash
# Download mondo.obo from the OBO Foundry and pin it by data-version.
#
# Saves to:   ~/data/ontologies/mondo-<YYYY-MM-DD>.obo
# Symlinks:   ~/data/ontologies/mondo.obo -> the dated file
#
# The <YYYY-MM-DD> token is read from the `data-version:` line of the
# downloaded OBO, so the filename pins the actual MONDO release rather
# than the local download date.
#
# Idempotent: refuses to overwrite an existing dated file unless --force.
set -euo pipefail

URL="https://purl.obolibrary.org/obo/mondo.obo"
DEST_DIR="${HOME}/data/ontologies"
FORCE=0

for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        -h|--help)
            grep -E '^# ' "$0" | sed 's/^# //'
            exit 0
            ;;
        *)
            echo "unknown arg: $arg" >&2
            exit 2
            ;;
    esac
done

mkdir -p "$DEST_DIR"

TMP_FILE="$(mktemp -t mondo.XXXXXX.obo)"
trap 'rm -f "$TMP_FILE"' EXIT

echo "[download_mondo] fetching $URL ..."
curl -fsSL "$URL" -o "$TMP_FILE"

# Read data-version from the OBO header. Format example:
#   data-version: mondo/releases/2024-09-03/mondo.obo
VERSION_LINE="$(grep -m1 '^data-version:' "$TMP_FILE" || true)"
if [[ -z "$VERSION_LINE" ]]; then
    echo "[download_mondo] ERROR: no data-version line in downloaded OBO" >&2
    exit 1
fi
# Extract first YYYY-MM-DD on the line.
VERSION_DATE="$(echo "$VERSION_LINE" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}' | head -n1 || true)"
if [[ -z "$VERSION_DATE" ]]; then
    echo "[download_mondo] ERROR: could not parse YYYY-MM-DD from: $VERSION_LINE" >&2
    exit 1
fi

DATED_FILE="${DEST_DIR}/mondo-${VERSION_DATE}.obo"
SYMLINK="${DEST_DIR}/mondo.obo"

if [[ -e "$DATED_FILE" && "$FORCE" -ne 1 ]]; then
    echo "[download_mondo] already have $DATED_FILE (use --force to overwrite)"
else
    mv "$TMP_FILE" "$DATED_FILE"
    trap - EXIT
    echo "[download_mondo] wrote $DATED_FILE"
fi

ln -sfn "$DATED_FILE" "$SYMLINK"
echo "[download_mondo] symlink: $SYMLINK -> $DATED_FILE"
echo "[download_mondo] MONDO data-version: $VERSION_DATE"
