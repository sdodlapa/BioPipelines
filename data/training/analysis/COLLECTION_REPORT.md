# Training Data Collection Report

**Generated:** 2025-12-03T02:40:56.787436

## Summary

| Metric | Value |
|--------|-------|
| Total Conversations | 100 |
| Total Turns | 544 |
| Success Rate | 100.0% |
| Workflow Generation Rate | 100.0% |
| Avg Latency | 0.02ms |

## Analysis Type Coverage

| Analysis Type | Coverage |
|---------------|----------|
| genome_assembly | 7.0% |
| medip_seq | 0.0% |
| long_read_variant_calling | 0.0% |
| structural_variant_detection | 0.0% |
| long_read_isoseq | 0.0% |
| multi_omics_integration | 0.0% |
| single_cell_rna_seq | 0.0% |
| metagenomics_profiling | 0.0% |
| rna_seq_differential_expression | 0.0% |
| rna_seq_basic | 0.0% |
| metagenomics_assembly | 0.0% |
| chip_seq_peak_calling | 0.0% |
| small_rna_seq | 0.0% |
| atac_seq | 0.0% |
| multi_modal_scrna | 0.0% |

## Top Tools Used

| Tool | Usage Count |
|------|-------------|
| fastqc | 105 |
| multiqc | 98 |
| samtools | 86 |
| picard | 86 |
| bwa | 76 |
| macs2 | 32 |
| deeptools | 32 |
| star | 28 |
| featurecounts | 28 |
| salmon | 28 |
| gatk | 26 |
| bcftools | 26 |
| homer | 25 |
| kraken2 | 17 |
| metaphlan | 17 |

## Gaps Identified

### 🟡 Coverage

**Severity:** MEDIUM
**Frequency:** 3

No test coverage for analysis types: atac_seq, chip_seq, rna_seq

**Recommendation:** Add test scenarios for missing analysis types

### 🟡 Response Quality

**Severity:** MEDIUM
**Frequency:** 100

Low confidence in intent or response

**Recommendation:** Train on more examples or improve parsing rules

## Files Generated

- Conversations: `data/training/conversations/*.json` (100 files)
- Run Results: `data/training/run_results/results_*.json`
- Analysis: `data/training/analysis/analysis_*.json`

## Next Steps

1. Review generated conversations for quality
2. Export successful interactions as training data  
3. Address identified gaps before next collection run
4. Scale to 1000+ conversations with varied scenarios
