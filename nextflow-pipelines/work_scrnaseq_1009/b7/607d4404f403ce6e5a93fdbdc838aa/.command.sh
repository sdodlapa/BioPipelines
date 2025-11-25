#!/bin/bash -ue
ln -sf /opt/conda/lib/libfreetype.so.6.20.4 libfreetype.so.6 || true
export LD_LIBRARY_PATH=$PWD:/opt/conda/lib:$LD_LIBRARY_PATH

STAR \
    --runMode alignReads \
    --runThreadN 8 \
    --genomeDir star_index_hg38 \
    --readFilesIn sample1_R2.fastq.gz sample1_R1.fastq.gz \
    --readFilesCommand zcat \
    --outSAMtype BAM SortedByCoordinate \
    --outFileNamePrefix sample1_ \
    --soloType CB_UMI_Simple \
    --soloCBwhitelist 10x_whitelist_v3.txt \
    --soloCBlen 16 \
    --soloUMIlen 12 \
    --soloFeatures Gene GeneFull \
    --soloOutFileNames Solo.out/ genes.tsv barcodes.tsv matrix.mtx \


cat <<-END_VERSIONS > versions.yml
"STARSOLO":
    star: $(STAR --version | sed -e "s/STAR_//g")
END_VERSIONS
