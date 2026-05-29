# Machine Learning for Bioinformatics

A collection of machine learning projects and tools for biological data analysis — designed with **biologists in mind**.

---

## What is Machine Learning?

Machine learning (ML) is a branch of artificial intelligence where computers learn patterns from data instead of following explicit rules. For biologists, think of it like this:

> **Traditional approach:** You write a rule :"if sequence has motif X, then it's a promoter."
> **ML approach:** You show the computer thousands of known promoters and non-promoters, and it *learns* the distinguishing patterns on its own.

ML is especially powerful in bioinformatics because biological data is complex, high-dimensional, and often contains patterns too subtle for manual rule-writing. Common applications include:

- **Classification** : Is this sequence coding or non-coding? Is this mutation pathogenic?
- **Regression** : How strongly will this drug bind to a target protein?
- **Clustering** : Which viruses are genetically similar? Which patients group together?
- **Dimensionality reduction** — Visualizing high-dimensional gene expression data in 2D/3D.

---

## Repository Contents

### Python Scripts

| File | Description |
|------|-------------|
| `Covid_virus.py` | SARS-CoV-2 sequence analysis and visualization |
| `Covid_virus_advanced.py` | Advanced COVID-19 analysis with ML models |
| `Covid_virus_advanced_api.py` | API-based COVID analysis pipeline |
| `esm.py` | ESM (Evolutionary Scale Modeling) protein language model interface |
| `ESm_Fold.py` | Protein structure prediction using ESMFold |
| `biopthon.py` | Biopython utilities for sequence handling |
| `drug_binding.py` | Drug-target binding affinity prediction |
| `plot.py` | Bioinformatics visualization utilities |
| `structure prediction.py` | Protein structure analysis tools |
| `generate_docx_explanations.py` | Auto-generates documentation from analysis |

### ML Projects (`3_ML_Projects/`)

| Project | Description |
|---------|-------------|
| **Project 1** : DNA K-mer Classifier | Classifies DNA sequences using k-mer frequency features with Random Forest / SVM |
| **Project 2** : Protein CNN Classifier | 1D Convolutional Neural Network for protein family classification from sequence |
| **Project 3** : Mutation ESM-2 Analysis | Uses ESM-2 embeddings to predict mutation effects on protein function |
| **Project 4** : Viral Host Predictor | Predicts viral host species from genomic signatures using ML |

Each project includes a standalone script and detailed explanation documents.

### Documentation

- `COVID_ADVANCED_DOCS.md` :Comprehensive documentation for the COVID-19 ML pipeline
- `COVID_ADVANCED_API.md` :API reference for COVID analysis tools
- `3_ML_Projects/README.md` : Per-project breakdown and usage instructions

### Presentation

- `Machine Learning in Bioinformatics.pptx` — Introductory slides covering ML concepts applied to bioinformatics

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/ShivMC/ML-for-bioinformatics.git
cd ML-for-bioinformatics

# Install dependencies
pip install -r 3_ML_Projects/requirements.txt
```

### Quick Start — DNA K-mer Classifier

```python
from Bio.Seq import Seq
from sklearn.ensemble import RandomForestClassifier

# See 3_ML_Projects/project1_dna_kmer_classifier.py
```

### Quick Start — Mutation Analysis with ESM-2

```python
# See 3_ML_Projects/project3_mutation_esm2.py
# Requires: torch, transformers
```

---

## Who Is This For?

- **Biologists** who want to apply ML to their research
- **Bioinformatics students** learning computational methods
- **ML practitioners** entering the biology domain

No advanced ML background required — each script is documented step-by-step.

---

## Author

**Shivani Pawar**

--- 

## License

This project is for educational and research purposes.
