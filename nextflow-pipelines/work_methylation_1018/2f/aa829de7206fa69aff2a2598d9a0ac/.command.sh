#!/bin/bash -ue
ln -sf /opt/conda/lib/libfreetype.so.6.20.4 libfreetype.so.6 || true
export LD_LIBRARY_PATH=$PWD:/opt/conda/lib:$LD_LIBRARY_PATH

trim_galore \
     \
    --cores 2 \
    --paired \
    --basename sample1 \
    sample1_R1.fastq.gz \
    sample1_R2.fastq.gz

cat <<-END_VERSIONS > versions.yml
"TRIM_GALORE":
    trimgalore: $(echo $(trim_galore --version 2>&1) | sed 's/^.*version //; s/ .*$//')
END_VERSIONS
