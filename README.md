# Demo repository - scoring mutations impact on RBP binding

## Overview

## Set-up

### Install

```bash
conda env create --file envs/conda_pysster_jupyter.yaml
```

### Data

Example data are available for the demo notebooks.


- Download and prepare the FASTA file for the hg38 genome:

```bash
wget http://hgdownload.soe.ucsc.edu/goldenpath/hg38/bigZips/hg38.fa.gz -O data/hg38.fa.gz
gunzip data/hg38.fa.gz
# Index with pyFaidx

# We also need the chromosome sizes file.
wget http://hgdownload.soe.ucsc.edu/goldenpath/hg38/bigZips/hg38.chrom.sizes -O data/hg38.chrom.sizes

# Alternatively download the 2bit file format.
wget http://hgdownload.soe.ucsc.edu/goldenpath/hg38/bigZips/hg38.2bit -O data/hg38.2bit
```


- Manually download Pysster models: <https://drive.google.com/file/d/1CQ2m9hOTi0_ruhIfTEzQFpj-1H8J2buP/view>


Note: the FASTA files can be maintained in compressed format and indexed with samtools `faidx` when compressed with `bgzip` ; this is done automatically by `pyfaidx` when the FASTA file is loaded for the first time.
