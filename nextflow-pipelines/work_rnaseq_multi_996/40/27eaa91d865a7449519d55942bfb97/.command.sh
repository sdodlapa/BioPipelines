#!/bin/bash -ue
# Fix missing libfreetype.so.6 symlink (same as FastQC)
ln -sf /opt/conda/lib/libfreetype.so.6.20.4 libfreetype.so.6
export LD_LIBRARY_PATH=$PWD:/opt/conda/lib:${LD_LIBRARY_PATH:-}

STAR \
    --genomeDir star_index_hg38 \
    --readFilesIn wt_rep2_R1.trimmed.fastq.gz wt_rep2_R2.trimmed.fastq.gz \
    --readFilesCommand zcat \
    --runThreadN 8 \
    --outFileNamePrefix wt_rep2. \
    --outSAMtype BAM SortedByCoordinate \
    --outSAMattributes NH HI AS NM MD \
    --outSAMattrRGline ID:wt_rep2 SM:wt_rep2 PL:ILLUMINA \
    --quantMode GeneCounts \


cat <<-END_VERSIONS > versions.yml
"STAR_ALIGN":
    star: $(STAR --version | sed -e "s/STAR_//g")
END_VERSIONS
