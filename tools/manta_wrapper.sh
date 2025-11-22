#!/bin/bash
# Wrapper to run Manta with Python 2.7
# Directly invokes Python 2.7 to bypass shebang #!/usr/bin/env python2 issues

PYTHON2="$HOME/miniconda3/envs/manta_py27/bin/python2"
MANTA_CONFIG="$HOME/BioPipelines/tools/manta-1.6.0.centos6_x86_64/bin/configManta.py"

# Verify python2 exists
if [[ ! -f "$PYTHON2" ]]; then
    echo "Error: python2 not found at $PYTHON2" >&2
    exit 1
fi

# Run Manta with explicit Python interpreter (bypasses shebang)
"$PYTHON2" "$MANTA_CONFIG" "$@"
