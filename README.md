# LiquidBiopsyEngine

**Cell-Free DNA / ctDNA Analysis Pipeline**

A pure-Python computational engine for liquid biopsy analysis of circulating tumor DNA (ctDNA).

## Features
- cfDNA fragment length analysis (nucleosome positioning signal, short/long ratio)
- Tumor fraction estimation (ichorCNA-style copy number variance)
- Low-VAF somatic variant calling (binomial test, BH FDR, depth ≥1000x)
- Methylation-based tissue-of-origin deconvolution (NNLS)
- Longitudinal tumor burden tracking (5 response patterns)

## Results
- 40 patients, 5 time points, 500 genomic bins
- Fragment ratio vs TF: r=0.995
- Variant calling: F1=0.917, Precision=1.000
- Tissue-of-origin accuracy: 0.950 (38/40)

## Usage
```bash
pip install numpy scipy matplotlib
python liquid_biopsy_engine.py
```

## Tags
`liquid-biopsy` `ctdna` `cell-free-dna` `tumor-fraction` `cfDNA`
