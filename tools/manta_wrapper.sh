#!/bin/bash
# Wrapper to run Manta with Python 2.7 from isolated conda env
# Usage: manta_wrapper.sh <manta_args>

# Activate Python 2.7 environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate manta_py27

# Export PATH to ensure /usr/bin/env finds python2
export PATH="$CONDA_PREFIX/bin:$PATH"

# Run Manta with all provided arguments
~/BioPipelines/tools/manta-1.6.0.centos6_x86_64/bin/configManta.py "$@"

# Deactivate
conda deactivate
