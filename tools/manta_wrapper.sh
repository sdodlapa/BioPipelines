#!/bin/bash
# Wrapper to run Manta with Python 2.7
# Manta's configManta.py requires '#!/usr/bin/env python2' to find python2 in PATH

# Set PATH to include manta_py27 environment's bin directory FIRST
export PATH="$HOME/miniconda3/envs/manta_py27/bin:$PATH"

# Verify python2 is accessible (log to stderr for debugging)
which python2 >&2 || { echo "Error: python2 not found in PATH" >&2; exit 1; }

# Run Manta configManta.py
~/BioPipelines/tools/manta-1.6.0.centos6_x86_64/bin/configManta.py "$@"
