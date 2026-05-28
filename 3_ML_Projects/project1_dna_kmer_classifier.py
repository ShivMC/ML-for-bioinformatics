"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PROJECT 1 (BASIC) — DAY 1                                                 ║
║  DNA Sequence Classification using k-mer Features + Random Forest          ║
╚══════════════════════════════════════════════════════════════════════════════╝

────────────────────────────────────────────────────────────────────────────
IDEA & BIOLOGICAL CONTEXT
────────────────────────────────────────────────────────────────────────────

In genomics, one fundamental task is classifying DNA sequences — determining
whether a sequence belongs to a particular species, is coding vs. non-coding,
or originates from a pathogenic vs. non-pathogenic organism.

The core idea: DNA sequences are composed of 4 nucleotides (A, T, G, C). By
breaking sequences into overlapping substrings of length k ("k-mers") and
counting their frequencies, we convert raw sequences into fixed-length
numerical feature vectors. This "k-mer profiling" captures compositional
signatures that differ between organisms or sequence types.

For example, E. coli has a different GC-content and different dinucleotide
(2-mer) biases compared to human DNA. A Random Forest classifier can learn
these patterns from k-mer frequency vectors to distinguish species.

Real-world application: Metagenomic binning — assigning DNA fragments from
an environmental sample to their source organism. This is critical in
microbiome research, pathogen detection, and forensic genomics.

────────────────────────────────────────────────────────────────────────────
ML FRAMEWORK & APPROACH
────────────────────────────────────────────────────────────────────────────

Algorithm:     Random Forest Classifier
               (ensemble of decision trees — robust, interpretable, works
               well on high-dimensional sparse features like k-mers)

Feature Type:  k-mer frequencies (k=3,4,5,6) — normalized occurrence counts
               of each possible nucleotide substring of length k

Type:          Supervised classification

Tools:         Python, Biopython, NumPy, pandas, scikit-learn, matplotlib

Evaluation:    Accuracy, Confusion Matrix, ROC-AUC, classification report

────────────────────────────────────────────────────────────────────────────
DATASET (Synthetic — fully reproducible)
────────────────────────────────────────────────────────────────────────────

We generate synthetic DNA sequences mimicking two species:
  - Species A: High-GC genome (e.g., Streptomyces — GC ~70%)
  - Species B: Low-GC genome (e.g., Plasmodium — GC ~20%)

Each sequence is 500 bp long. We generate 500 sequences per species.
The model learns to classify species from k-mer profiles alone.

You can swap this for real genomic data from NCBI GenBank using Bio.Entrez.

────────────────────────────────────────────────────────────────────────────
HOW TO RUN
────────────────────────────────────────────────────────────────────────────
  pip install biopython numpy pandas scikit-learn matplotlib
  python project1_dna_kmer_classifier.py
────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from collections import Counter
import itertools
import random

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Generate synthetic DNA sequences for two "species"
# ─────────────────────────────────────────────────────────────────────────────

def generate_dna_sequence(length, gc_content):
    """Generate a random DNA sequence with a target GC-content."""
    at_prob = (1 - gc_content) / 2
    gc_prob = gc_content / 2
    bases = np.random.choice(
        ['A', 'T', 'G', 'C'],
        size=length,
        p=[at_prob, at_prob, gc_prob, gc_prob]
    )
    return ''.join(bases)

def create_dataset(seq_length=500, n_per_class=500):
    sequences = []
    labels = []

    # Species A: high GC (e.g., Streptomyces bacteria)
    for _ in range(n_per_class):
        sequences.append(generate_dna_sequence(seq_length, gc_content=0.70))
        labels.append(0)

    # Species B: low GC (e.g., Plasmodium parasite)
    for _ in range(n_per_class):
        sequences.append(generate_dna_sequence(seq_length, gc_content=0.25))
        labels.append(1)

    return sequences, labels

print("=" * 65)
print("PROJECT 1: DNA Sequence Classification with k-mer + Random Forest")
print("=" * 65)

sequences, labels = create_dataset(seq_length=500, n_per_class=500)
print(f"\nGenerated {len(sequences)} sequences ({len(sequences)//2} per class)")
print(f"Sequence length: 500 bp")
print(f"Species A (label 0): High GC (~70%) — e.g., Streptomyces")
print(f"Species B (label 1): Low GC (~25%) — e.g., Plasmodium")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: k-mer feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_kmer_features(sequence, k=4):
    """
    Count occurrences of each possible k-mer in the sequence.
    Returns a vector of length 4^k.
    """
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    counter = Counter(kmers)
    all_kmers = [''.join(p) for p in itertools.product('ATGC', repeat=k)]
    feature_vector = np.array([counter.get(kmer, 0) for kmer in all_kmers], dtype=float)
    # Normalize to frequency (fraction of total k-mers)
    feature_vector /= feature_vector.sum()
    return feature_vector

k = 4  # tetranucleotide frequency
print(f"\nExtracting {k}-mer features (4^{k} = {4**k} features per sequence)...")

X = np.array([extract_kmer_features(seq, k=k) for seq in sequences])
y = np.array(labels)

print(f"Feature matrix shape: {X.shape}")
print(f"Label vector shape: {y.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Train / Test split
# ─────────────────────────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set:  {X_test.shape[0]} samples")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Train Random Forest Classifier
# ─────────────────────────────────────────────────────────────────────────────

print("\nTraining Random Forest classifier...")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Evaluate
# ─────────────────────────────────────────────────────────────────────────────

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"\n{'─' * 50}")
print(f"  Test Accuracy:  {accuracy:.4f}")
print(f"  ROC-AUC Score:  {roc_auc:.4f}")
print(f"{'─' * 50}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Species A (High GC)', 'Species B (Low GC)']))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Feature importance — which k-mers are most discriminative?
# ─────────────────────────────────────────────────────────────────────────────

all_kmers = [''.join(p) for p in itertools.product('ATGC', repeat=k)]
importances = rf.feature_importances_
top_n = 10
top_indices = np.argsort(importances)[::-1][:top_n]

print(f"\nTop {top_n} most discriminative {k}-mers:")
for i, idx in enumerate(top_indices):
    print(f"  {i+1}. {all_kmers[idx]} — importance: {importances[idx]:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Visualization
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Species A', 'Species B'],
            yticklabels=['Species A', 'Species B'])
axes[0].set_title(f'Confusion Matrix (Accuracy: {accuracy:.3f})')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, label=f'ROC-AUC = {roc_auc:.3f}', lw=2)
axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('project1_confusion_roc.png', dpi=150)
print("\n[Saved] project1_confusion_roc.png")
# plt.show()

print("\n" + "=" * 65)
print("PROJECT 1 COMPLETE")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# EXTENSIONS (for further exploration)
# ─────────────────────────────────────────────────────────────────────────────
#
# 1. Use real genomic data via Biopython Entrez:
#    from Bio import Entrez, SeqIO
#    Entrez.email = "your@email.com"
#    handle = Entrez.efetch(db="nucleotide", id="NC_000913", rettype="fasta")
#    record = SeqIO.read(handle, "fasta")
#
# 2. Try different k values (3, 5, 6) — trade-off: larger k = more features,
#    sparser data, need more samples
#
# 3. Try other classifiers: Logistic Regression, SVM, XGBoost
#
# 4. Apply to real problem: classify coding vs. non-coding DNA, or detect
#    horizontally transferred genes by anomalous k-mer signatures
#
# 5. Use reverse complement aware k-mers (collapse palindromic pairs)
# ─────────────────────────────────────────────────────────────────────────────
