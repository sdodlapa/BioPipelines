# BioPipelines TODO List

## 🔥 Immediate Actions (This Week)

### Day 1-2: Setup & Legal
- [ ] Add MIT LICENSE file
- [ ] Update README.md (replace YOUR_USERNAME with actual GitHub username)
- [ ] Create CONTRIBUTING.md
- [ ] Create .github/ISSUE_TEMPLATE/

### Day 3-5: Validation Infrastructure
- [ ] Download test datasets:
  - [ ] DNA-seq: Small human WGS sample (chr22 only)
  - [ ] RNA-seq: Small RNA-seq sample (subset)
  - [ ] ChIP-seq: Small ChIP-seq sample
  - [ ] ATAC-seq: Small ATAC-seq sample
- [ ] Create `data/test/README.md` with dataset descriptions
- [ ] Write `scripts/download_test_data.sh` script

### Day 6-7: First Pipeline Test
- [ ] Test DNA-seq variant calling pipeline end-to-end
- [ ] Document any errors encountered
- [ ] Fix bugs found during testing
- [ ] Create example output in `data/results/example/`

---

## 📅 Week 2: Core Testing

### DNA-seq Pipeline
- [ ] Write unit tests for variant calling utilities
- [ ] Test with different reference genomes
- [ ] Benchmark resource usage (time/memory)
- [ ] Document in `docs/pipelines/dna_seq.md`

### RNA-seq Pipeline
- [ ] Test differential expression pipeline
- [ ] Validate DESeq2 results
- [ ] Test with different conditions
- [ ] Document in `docs/pipelines/rna_seq.md`

### Python Utilities
- [ ] Write tests for `src/biopipelines/alignment/`
- [ ] Write tests for `src/biopipelines/preprocessing/`
- [ ] Write tests for `src/biopipelines/variant_calling/`
- [ ] Set up pytest configuration

---

## 📅 Week 3: Documentation

### Pipeline Docs
- [ ] Create `docs/pipelines/dna_seq_variant_calling.md`
- [ ] Create `docs/pipelines/rna_seq_differential_expression.md`
- [ ] Create `docs/pipelines/chip_seq_peak_calling.md`
- [ ] Create `docs/pipelines/atac_seq_accessibility.md`

### Tutorials
- [ ] Create `notebooks/tutorials/01_dna_seq_variant_calling.ipynb`
- [ ] Create `notebooks/tutorials/02_rna_seq_analysis.ipynb`
- [ ] Create `notebooks/tutorials/03_chip_seq_analysis.ipynb`
- [ ] Create installation guide: `docs/INSTALLATION.md`

### API Docs
- [ ] Add docstrings to all Python functions
- [ ] Set up Sphinx documentation
- [ ] Generate HTML docs
- [ ] Create `docs/api/index.md`

---

## 📅 Week 4-5: Missing Pipelines

### Metagenomics Pipeline
- [ ] Create `pipelines/metagenomics/taxonomic_profiling/`
- [ ] Implement Kraken2/MetaPhlAn workflow
- [ ] Add assembly step (MEGAHIT)
- [ ] Add functional annotation (HUMAnN3)
- [ ] Create config and environment files
- [ ] Test with mock community data

### RNA-seq Isoform Analysis
- [ ] Create `pipelines/rna_seq/isoform_analysis/Snakefile`
- [ ] Implement StringTie/Salmon workflow
- [ ] Add differential isoform usage analysis
- [ ] Create environment file
- [ ] Test with known isoform switching

### Standalone QC Workflows
- [ ] Create `pipelines/dna_seq/quality_control/Snakefile`
- [ ] Create `pipelines/rna_seq/quality_control/Snakefile`
- [ ] Modular QC-only workflows

---

## 📅 Week 6: Polish & CI/CD

### GitHub Actions
- [ ] Create `.github/workflows/tests.yml`
- [ ] Set up automated testing
- [ ] Add linting (black, flake8)
- [ ] Add coverage reporting

### Benchmarking
- [ ] Run performance benchmarks on all pipelines
- [ ] Compare with published workflows
- [ ] Document in `benchmarks/results/`

### Final Polish
- [ ] Update README with badges
- [ ] Add citation information
- [ ] Create CHANGELOG.md
- [ ] Tag v1.0.0 release

---

## 🎯 Critical Path (Minimum Viable Product)

If time is limited, focus on these essentials:

1. ✅ **Add LICENSE** (5 min)
2. ✅ **Test DNA-seq pipeline** (1 day)
3. ✅ **Test RNA-seq pipeline** (1 day)
4. ✅ **Write README installation instructions** (2 hours)
5. ✅ **Create one tutorial notebook** (4 hours)
6. ✅ **Write basic unit tests** (1 day)

**Total MVP Time:** ~3 days

---

## 🐛 Known Issues

- [ ] DNA-seq Snakefile has uncommitted changes
- [ ] `docs/submit_dna_seq.sh` is untracked - review and commit/remove
- [ ] Empty `__init__.py` files need proper imports
- [ ] Need to verify all conda environments install correctly
- [ ] Path issues in Snakefiles may need adjustment for different systems

---

## 💡 Nice-to-Have Features

### Future Enhancements
- [ ] Docker containers for each pipeline
- [ ] Nextflow version of pipelines
- [ ] Cloud deployment scripts (AWS/GCP)
- [ ] Web dashboard for results
- [ ] Parameter optimization tool
- [ ] Automatic report generation
- [ ] Integration with public databases

### Advanced Analysis
- [ ] Single-cell RNA-seq pipeline
- [ ] Spatial transcriptomics
- [ ] Long-read sequencing (PacBio/Nanopore)
- [ ] Multi-omics integration
- [ ] Machine learning QC predictions

---

## 📞 Support Needed

### Questions to Resolve
- [ ] Which GitHub username to use?
- [ ] Which reference genomes to support? (hg38, mm10, others?)
- [ ] What compute environment? (HPC, cloud, local?)
- [ ] Citation preferences?
- [ ] Target audience? (researchers, core facilities, students?)

### External Resources Needed
- [ ] Access to test datasets
- [ ] Compute resources for testing
- [ ] Code review partners
- [ ] Beta testers

---

## 🎓 Learning Resources

If you need to learn more about components:

- **Snakemake:** https://snakemake.readthedocs.io/
- **GATK Best Practices:** https://gatk.broadinstitute.org/
- **RNA-seq Analysis:** https://www.bioconductor.org/packages/release/workflows/html/rnaseqGene.html
- **Conda/Bioconda:** https://bioconda.github.io/
- **pytest:** https://docs.pytest.org/

---

*Last Updated: November 20, 2025*

