#!/bin/bash -ue
# Fix libfreetype symlink (same as other modules)
ln -sf /opt/conda/lib/libfreetype.so.6.20.4 libfreetype.so.6 2>/dev/null || true
export LD_LIBRARY_PATH=$PWD:/opt/conda/lib:${LD_LIBRARY_PATH:-}

bowtie2 \
    -x bowtie2_index/hg38 \
    -1 new_sample2_R1.trimmed.fastq.gz -2 new_sample2_R2.trimmed.fastq.gz \
    --threads 8 \
     \
    2> new_sample2.bowtie2.log \
    | samtools sort -@ 8 -O bam -o new_sample2.bam -

samtools index new_sample2.bam

cat <<-END_VERSIONS > versions.yml
"BOWTIE2_ALIGN":
    bowtie2: $(echo $(bowtie2 --version 2>&1) | head -n1 | sed 's/^.*version //; s/ .*$//')
    samtools: $(echo $(samtools --version 2>&1) | head -n1 | sed 's/^.*samtools //; s/ .*$//')
END_VERSIONS
