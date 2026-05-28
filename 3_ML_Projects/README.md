
# ML Projects: Genomics & Structural Bioinformatics

Four scaffolded projects from basic → advanced, using Python + Biopython.

Projects 1-3 based on the **ML_Genomics_Structural_Bioinformatics Workshop**.
Project 4 based on **Alam et al. (2025) *BMC Biology* 23:324** (Young et al. 2020).

---

## Project 1 — Basic (Day 1)
**DNA Sequence Classification with k-mer Features + Random Forest**

| Aspect | Detail |
|---|---|
| **Problem** | Classify DNA sequences by species (high-GC vs low-GC organism) |
| **Biology** | GC-content and k-mer compositional signatures distinguish genomes |
| **Features** | Tetranucleotide (4-mer) frequency vectors (4^4 = 256 features) |
| **Model** | Random Forest Classifier (200 trees, depth=20) |
| **Framework** | scikit-learn, Biopython, numpy, pandas |
| **File** | `project1_dna_kmer_classifier.py` |
| **Key Concept** | Converting biological sequences into numerical feature vectors via k-mer counting |
| **Metrics** | Accuracy, ROC-AUC, Confusion Matrix, Feature Importance |

**What you learn:**
- k-mer feature extraction from DNA
- Supervised classification with ensemble methods
- Biological interpretation of feature importance (which k-mers discriminate species)
- ROC curves and model evaluation

**Run:**
```bash
python project1_dna_kmer_classifier.py
```

---

## Project 2 — Intermediate (Day 2)
**Protein Family Classification with 1D Convolutional Neural Network**

| Aspect | Detail |
|---|---|
| **Problem** | Classify protein sequences into 6 families (Kinase, GPCR, Protease, Ion Channel, Helicase, Zinc Finger) |
| **Biology** | Each family has conserved sequence motifs that define its function |
| **Architecture** | 1D-CNN: Conv1D(64,k=5) → MaxPool → Conv1D(128,k=3) → GlobalMaxPool → Dense(64) → Dense(6) |
| **Input** | One-hot encoded sequences (20 amino acids × 250 residues) |
| **Framework** | PyTorch, Biopython |
| **File** | `project2_protein_cnn_classifier.py` |
| **Key Concept** | CNNs learn position-independent motif detectors — analogous to biological sequence motifs |
| **Metrics** | Per-class F1, Accuracy, Confusion Matrix |

**What you learn:**
- One-hot encoding for protein sequences
- 1D CNN architecture for sequence data
- Motif detection via convolutional filters
- Training loop design, loss curves, overfitting monitoring
- Multi-class classification in PyTorch

**Run:**
```bash
python project2_protein_cnn_classifier.py
```

---

## Project 3 — Advanced (Day 3)
**Mutation Impact Prediction using ESM-2 Protein Language Model**

| Aspect | Detail |
|---|---|
| **Problem** | Predict whether a single amino acid mutation destabilizes a protein |
| **Biology** | Mutations alter protein stability → disease, drug resistance, loss of function |
| **Model** | ESM-2 (Meta) — transformer trained on 65M protein sequences via masked language modeling |
| **Approach** | (1) Zero-shot: log-likelihood ratio of mutant vs wild-type at mutated position |
| | (2) Fine-tuned: regression head on frozen ESM-2 embeddings |
| **Framework** | HuggingFace Transformers, PyTorch, ESM |
| **File** | `project3_mutation_esm2.py` |
| **Key Concept** | Protein language models encode evolutionary constraints — mutations violating these constraints are likely destabilizing |
| **Metrics** | ROC-AUC, Pearson r, MSE |

**What you learn:**
- Loading and using pretrained protein language models
- Zero-shot mutation effect prediction (no training needed)
- Extracting embeddings from transformer models
- Fine-tuning regression heads on frozen representations
- The bridge between AI and structural biology

**Run:**
```bash
python project3_mutation_esm2.py
```

---

---

## Project 4 — Paper-Based
**Viral Host Taxonomy Prediction with SVM + XGBoost**

| Aspect | Detail |
|---|---|
| **Reference** | Alam et al. (2025) *BMC Biology* 23:324; Young et al. (2020) *PLoS Comput Biol* 16(5):e1007894 |
| **Problem** | Predict host taxonomic class (Mammal, Bird, Plant, Bacteria) from viral genome sequence |
| **Biology** | Host-imposed selective pressures shape viral nucleotide composition: CpG suppression in mammals, GC biases, dinucleotide preferences |
| **Features** | k=3 (64) + k=4 (256) k-mer frequencies, dinucleotide bias ratios (16), GC content/skew (3), Ti/Tv ratio (1) = 340 total |
| **Models** | SVM (linear kernel — as in Young et al.) + XGBoost (for comparison) |
| **Framework** | scikit-learn, XGBoost, Biopython |
| **File** | `project4_viral_host_predictor.py` |
| **Key Concept** | Viral genomes encode host-specific signatures in their nucleotide composition — SVM can decode these |
| **Metrics** | Accuracy, ROC-AUC (one-vs-rest), Confusion Matrix, Feature Importance |

**What you learn:**
- Reproducing a published bioinformatics ML study from the literature
- Multi-class SVM with linear kernel
- Multiple feature type integration (k-mer + dinucleotide + GC)
- Biological interpretation: CpG suppression, GC content as host predictors
- Model comparison (SVM vs XGBoost)

**Run:**
```bash
python project4_viral_host_predictor.py
```

---

## Skill Progression

```
Day 1 (Basic)         Day 2 (Intermediate)       Day 3 (Advanced)
────────────────────────────────────────────────────────────────────
k-mer features        One-hot encoding            Pretrained transformers
scikit-learn          PyTorch                     HuggingFace / ESM
Random Forest         CNN architecture            Zero-shot scoring
Feature importance    Loss curves / training      Embedding extraction
Accuracy / ROC-AUC    Multi-class confusion matrix Regression fine-tuning
```

## Installation

```bash
pip install -r requirements.txt
```

Each project generates a PNG figure summarizing results.

## Real Data Substitution

| Project | Real Data Source |
|---|---|---|
| 1 | NCBI GenBank via `Bio.Entrez.efetch(db="nucleotide")` |
| 2 | UniProt/Swiss-Prot via `Bio.ExPASy.get_sprot_raw()` |
| 3 | ProTherm (stability), ClinVar (pathogenicity), ProteinGym (DMS) |
| 4 | Virus-Host DB (genome.jp/virushostdb/), NCBI RefSeq viral genomes |

## Paper Reference

Alam, M.N.U. et al. (2025). Machine learning in biological research: key algorithms, applications, and future directions. *BMC Biology*, 23:324. https://doi.org/10.1186/s12915-025-02424-3

Young, F., Rogers, S., Robertson, D.L. (2020). Predicting host taxonomic information from viral genomes: a comparison of feature representations. *PLoS Computational Biology*, 16(5):e1007894. https://doi.org/10.1371/journal.pcbi.1007894
