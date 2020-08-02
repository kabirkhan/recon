#!/bin/sh -e
set -x

# Sort imports one per line, so autoflake can remove unused imports
isort recon tests docs/src
sh ./scripts/format.sh
