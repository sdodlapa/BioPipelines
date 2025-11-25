#!/bin/bash -ue
# Fix missing libfreetype.so.6 symlink for Java font rendering
# Create symlink in work directory (writable)
ln -sf /opt/conda/lib/libfreetype.so.6.20.4 libfreetype.so.6
export LD_LIBRARY_PATH=$PWD:/opt/conda/lib:${LD_LIBRARY_PATH:-}

fastqc \
     \
    --threads 4 \
    --memory 2048 \
    --quiet \
    h3k4me3_rep2.trimmed.fastq.gz

cat <<-END_VERSIONS > versions.yml
"FASTQC":
    fastqc: $( fastqc --version | sed '/FastQC v/!d; s/.*v//' )
END_VERSIONS
