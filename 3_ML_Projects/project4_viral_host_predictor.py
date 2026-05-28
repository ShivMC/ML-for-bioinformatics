"""
=============================================================================
  PROJECT 4 — VIRAL HOST TAXONOMY PREDICTION
  Based on: Young et al. (2020) PLoS Comput Biol 16(5):e1007894
  "Predicting host taxonomic information from viral genomes:
   a comparison of feature representations"

  Reference paper: Alam et al. (2025) BMC Biology 23:324
  "Machine learning in biological research: key algorithms,
   applications, and future directions"

  Key idea from the paper (SVM section):
    Viral genome sequences contain host-specific genomic signatures.
    Using k-mer frequencies and physicochemical properties as features,
    an SVM classifier can predict the taxonomic class of the host
    (mammal, bird, plant, bacteria) from the viral genome alone.
=============================================================================

IDEA & BIOLOGICAL CONTEXT
──────────────────────────
Viruses are obligate intracellular parasites — their genomes bear
evolutionary signatures of their host environment. Host-imposed
selective pressures shape viral nucleotide composition, codon usage,
and dinucleotide biases. For example:
  - Mammalian viruses avoid CpG dinucleotides (innate immune targeting
    via ZAP protein)
  - Plant viruses have different GC-content biases
  - Bacteriophages have distinct k-mer profiles

Young et al. (2020) showed that SVMs on k-mer features from viral
genomes can predict host taxonomy with >80% accuracy — a powerful
tool for characterizing newly discovered viruses where the host is
unknown (critical for pandemic preparedness).

APPLICATION
───────────
  - Characterizing viruses from metagenomic sequencing
  - Identifying potential zoonotic viruses
  - Understanding viral host range evolution
  - Pandemic surveillance of novel pathogens

ML FRAMEWORK
────────────
  Algorithm:    SVM (linear kernel) + XGBoost
  Features:     k-mer frequencies (k=3,4), GC-content, dinucleotide
                biases, transition/transversion ratios
  Type:         Multi-class supervised classification (4 host classes)
  Evaluation:   Accuracy, ROC-AUC (one-vs-rest), confusion matrix,
                feature importance
  Tools:        Python, Biopython, scikit-learn, XGBoost

HOW TO RUN
──────────
  pip install biopython numpy pandas scikit-learn xgboost matplotlib seaborn
  python project4_viral_host_predictor.py
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from itertools import product
import random
import warnings
import os
warnings.filterwarnings('ignore')

random.seed(42)
np.random.seed(42)

OUTPUT_DIR = 'project4_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────
# Synthetic viral genomes with host-specific compositional signatures.
# We model four host classes with distinct genomic biases:

HOST_CLASSES = ['Mammal', 'Bird', 'Plant', 'Bacteria']

# Host-specific genomic signatures (nucleotide biases + CpG suppression)
HOST_SIGNATURES = {
    'Mammal': {
        'gc_content': 0.42,         # Mammalian viruses ~42% GC
        'cpg_suppression': 0.35,     # Strong CpG suppression (ZAP protein)
        'dinuc_bias': {'AA': 1.1, 'CG': 0.35, 'TA': 0.8, 'GC': 1.0},
        'description': 'CpG suppression via ZAP; moderate GC'
    },
    'Bird': {
        'gc_content': 0.48,
        'cpg_suppression': 0.60,     # Weaker CpG suppression
        'dinuc_bias': {'AA': 1.0, 'CG': 0.60, 'TA': 0.9, 'GC': 1.1},
        'description': 'Moderate CpG suppression; higher GC'
    },
    'Plant': {
        'gc_content': 0.52,          # Plant viruses tend higher GC
        'cpg_suppression': 0.80,     # Minimal CpG suppression
        'dinuc_bias': {'AA': 0.9, 'CG': 0.80, 'TA': 1.1, 'GC': 1.2},
        'description': 'High GC; minimal CpG suppression'
    },
    'Bacteria': {
        'gc_content': 0.38,          # Phages often AT-rich
        'cpg_suppression': 0.95,     # No CpG suppression in bacteria
        'dinuc_bias': {'AA': 1.3, 'CG': 0.95, 'TA': 1.2, 'GC': 0.8},
        'description': 'AT-rich; no CpG suppression'
    },
}


def generate_viral_genome(length, host_class):
    """
    Generate a synthetic viral genome with host-specific compositional biases.

    The method:
      1. Set base composition from GC-content target
      2. Apply dinucleotide biases (especially CpG suppression)
      3. Add 2-4 host-specific "signature motifs" (short conserved sequences)
      4. Return the final genome string
    """
    sig = HOST_SIGNATURES[host_class]
    gc = sig['gc_content']
    at = 1.0 - gc
    p_gc = gc / 2  # per-base prob for G and C
    p_at = at / 2  # per-base prob for A and T

    # Initial sequence from base composition
    seq_list = np.random.choice(
        ['A', 'T', 'G', 'C'], size=length,
        p=[p_at, p_at, p_gc, p_gc]
    )

    # Apply dinucleotide bias correction via rejection sampling
    # We iterate and adjust based on target dinucleotide biases
    for i in range(len(seq_list) - 1):
        dinuc = seq_list[i] + seq_list[i+1]
        if dinuc in sig['dinuc_bias']:
            bias = sig['dinuc_bias'][dinuc]
            if np.random.random() > bias:
                # Replace with a more preferred dinucleotide
                preferred = max(sig['dinuc_bias'], key=lambda k: sig['dinuc_bias'][k])
                seq_list[i+1] = preferred[1]

    # Insert host-specific signature motifs
    motifs = {
        'Mammal':   ['GGAACC', 'CCAGGA', 'AACUGG', 'UCCCAG'],
        'Bird':     ['GGACGA', 'CCGAGA', 'AACGGA', 'CCGGAA'],
        'Plant':    ['GCCGCC', 'CGCGCC', 'GGCGGC', 'CCGCCG'],
        'Bacteria': ['GGATCC', 'AAGGTT', 'CCCTAG', 'GGGGAA'],
    }

    host_motifs = motifs[host_class]
    for motif in host_motifs:
        pos = random.randint(0, length - len(motif) - 1)
        for j, nt in enumerate(motif):
            if nt in 'ATGC':
                seq_list[pos + j] = nt

    return ''.join(seq_list)


print("=" * 65)
print("PROJECT 4: VIRAL HOST TAXONOMY PREDICTION")
print("Based on: Young et al. (2020) — SVM on viral genome features")
print("=" * 65)

print("\n[1/6] Generating synthetic viral genomes with host-specific signatures...")

N_PER_HOST = 200
GENOME_LENGTH = 5000

all_genomes = []
all_labels = []
metadata_rows = []

for host in HOST_CLASSES:
    for i in range(N_PER_HOST):
        seq = generate_viral_genome(GENOME_LENGTH, host)
        genome_id = f"{host[:4]}_{i:03d}"
        all_genomes.append((genome_id, seq))
        all_labels.append(host)
        metadata_rows.append({'id': genome_id, 'host': host})

meta_df = pd.DataFrame(metadata_rows)
print(f"  Genomes generated: {len(all_genomes)} ({N_PER_HOST} per host)")
print(f"  Genome length: {GENOME_LENGTH} nt")
print(f"  Host classes: {', '.join(HOST_CLASSES)}")

# Print some stats to verify host-specific patterns
print("\n  Verifying host-specific signatures:")
for host in HOST_CLASSES:
    host_seqs = [s for (_, s), h in zip(all_genomes, all_labels) if h == host]
    gc_vals = [(s.count('G') + s.count('C')) / len(s) for s in host_seqs[:10]]
    cpg_vals = [s.count('CG') / (len(s) - 1) * 100 for s in host_seqs[:10]]
    print(f"    {host:10s} | GC: {np.mean(gc_vals):.2f} | CpG/100nt: {np.mean(cpg_vals):.2f} | "
          f"{HOST_SIGNATURES[host]['description']}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
# Following Young et al. (2020), we extract multiple feature types:
#   1. k-mer frequencies (k=3, 4) — captures sequence composition
#   2. Dinucleotide bias ratios
#   3. GC content + GC skew
#   4. Transition/transversion ratio

print("\n[2/6] Extracting features from viral genomes...")

NUCLEOTIDES = ['A', 'T', 'G', 'C']


def extract_kmer_freqs(seq, k):
    """Return normalized k-mer frequency vector (length 4^k)."""
    kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
    counter = Counter(kmers)
    all_kmers = [''.join(p) for p in product(NUCLEOTIDES, repeat=k)]
    vec = np.array([counter.get(kmer, 0) for kmer in all_kmers], dtype=float)
    vec /= vec.sum() if vec.sum() > 0 else 1
    return vec


def extract_dinucleotide_bias(seq):
    """Calculate observed/expected dinucleotide ratios (16 values)."""
    mono_count = Counter(seq)
    total = len(seq)
    freq = {nt: mono_count[nt] / total for nt in NUCLEOTIDES}

    observed = {}
    for a, b in product(NUCLEOTIDES, repeat=2):
        observed[a+b] = seq.count(a+b)

    # Observed/expected ratio
    ratios = []
    for a, b in product(NUCLEOTIDES, repeat=2):
        expected = freq[a] * freq[b] * (total - 1)
        observed_val = observed[a+b]
        ratio = observed_val / expected if expected > 0 else 1.0
        ratios.append(ratio)
    return np.array(ratios)


def extract_gc_features(seq):
    """GC content, GC skew, AT skew."""
    g = seq.count('G')
    c = seq.count('C')
    a = seq.count('A')
    t = seq.count('T')
    total = len(seq)
    gc = (g + c) / total
    gc_skew = (g - c) / (g + c + 1)
    at_skew = (a - t) / (a + t + 1)
    return np.array([gc, gc_skew, at_skew])


def extract_titv_ratio(seq):
    """Transition / Transversion ratio in the genome (as neutral mutation proxy)."""
    transitions = 0
    transversions = 0
    for i in range(len(seq) - 1):
        pair = seq[i] + seq[i+1]
        if pair in [('A','G'), ('G','A'), ('C','T'), ('T','C')]:
            transitions += 1
        elif pair[0] != pair[1] and pair[0] in 'AG' and pair[1] in 'AG':
            transversions += 1
        elif pair[0] != pair[1] and pair[0] in 'CT' and pair[1] in 'CT':
            transversions += 1
        elif pair[0] != pair[1]:
            transversions += 1
    return np.array([transitions / (transversions + 1)])


# Build feature matrix
feature_names = []

# k=3 features (64)
all_kmers_3 = [''.join(p) for p in product(NUCLEOTIDES, repeat=3)]
feature_names += [f'k3_{k}' for k in all_kmers_3]

# k=4 features (256)
all_kmers_4 = [''.join(p) for p in product(NUCLEOTIDES, repeat=4)]
feature_names += [f'k4_{k}' for k in all_kmers_4]

# Dinucleotide bias (16)
feature_names += [f'di_{a}{b}' for a, b in product(NUCLEOTIDES, repeat=2)]

# GC features (3)
feature_names += ['gc_content', 'gc_skew', 'at_skew']

# Ti/Tv (1)
feature_names += ['titv_ratio']

print(f"  Total feature dimensions: {len(feature_names)}")
print(f"    k=3:     64 features")
print(f"    k=4:    256 features")
print(f"    Dinuc:   16 features")
print(f"    GC:       3 features")
print(f"    Ti/Tv:    1 feature")

X_list = []
for gid, seq in all_genomes:
    k3 = extract_kmer_freqs(seq, 3)
    k4 = extract_kmer_freqs(seq, 4)
    di = extract_dinucleotide_bias(seq)
    gc = extract_gc_features(seq)
    tv = extract_titv_ratio(seq)
    X_list.append(np.concatenate([k3, k4, di, gc, tv]))

X = np.array(X_list)
y = np.array(all_labels)

print(f"\n  Feature matrix shape: {X.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: TRAIN/TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"\n[3/6] Split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: SVM CLASSIFIER (as in Young et al. 2020)
# ─────────────────────────────────────────────────────────────────────────────

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from xgboost import XGBClassifier

print("\n[4/6] Training classifiers...")

# Standardize features (important for SVM)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ── 4A: SVM with linear kernel (as in Young et al. 2020) ──
print("\n  4A: SVM (linear kernel) — per Young et al. 2020...")

svm_model = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
svm_model.fit(X_train_s, y_train)
svm_pred = svm_model.predict(X_test_s)
svm_prob = svm_model.predict_proba(X_test_s)
svm_acc = accuracy_score(y_test, svm_pred)

print(f"    SVM Accuracy: {svm_acc:.4f}")
print(f"\n    SVM Classification Report:")
print(classification_report(y_test, svm_pred))

# ── 4B: XGBoost (for comparison) ──
print("\n  4B: XGBoost classifier...")

xgb_model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                           subsample=0.8, colsample_bytree=0.8,
                           random_state=42, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_prob = xgb_model.predict_proba(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)

print(f"    XGBoost Accuracy: {xgb_acc:.4f}")
print(f"\n    XGBoost Classification Report:")
print(classification_report(y_test, xgb_pred))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: FEATURE IMPORTANCE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

print("\n[5/6] Feature importance analysis...")

# SVM: use absolute coefficient magnitude as feature importance
svm_coef = np.abs(svm_model.coef_).mean(axis=0)
svm_importance = pd.DataFrame({
    'feature': feature_names,
    'importance_svm': svm_coef
}).sort_values('importance_svm', ascending=False)

# XGBoost: built-in feature importance
xgb_importance = pd.DataFrame({
    'feature': feature_names,
    'importance_xgb': xgb_model.feature_importances_
}).sort_values('importance_xgb', ascending=False)

# Top features from both
print("\n  Top 10 features (SVM):")
for _, r in svm_importance.head(10).iterrows():
    print(f"    {r['feature']:15s}  weight: {r['importance_svm']:.4f}")

print("\n  Top 10 features (XGBoost):")
for _, r in xgb_importance.head(10).iterrows():
    print(f"    {r['feature']:15s}  weight: {r['importance_xgb']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

print("\n[6/6] Generating figures...")

sns.set_style('whitegrid')
plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 150,
                      'font.size': 11, 'axes.titlesize': 13})

# ── Figure 1: Confusion matrices (SVM vs XGBoost) ──
fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))

for ax, model_name, y_pred in [(axes1[0], 'SVM (linear)', svm_pred),
                                 (axes1[1], 'XGBoost', xgb_pred)]:
    cm = confusion_matrix(y_test, y_pred, labels=HOST_CLASSES)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=HOST_CLASSES, yticklabels=HOST_CLASSES)
    ax.set_title(f'{model_name} (Acc: {accuracy_score(y_test, y_pred):.3f})')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
fig1.savefig(f'{OUTPUT_DIR}/01_confusion_matrices.png')
plt.close(fig1)

# ── Figure 2: ROC curves (one-vs-rest) ──
fig2, ax2 = plt.subplots(figsize=(9, 7))

# Use XGBoost probabilities for ROC
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)

for i, host in enumerate(HOST_CLASSES):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], xgb_prob[:, i])
    auc = roc_auc_score(y_test_bin[:, i], xgb_prob[:, i])
    ax2.plot(fpr, tpr, lw=2, label=f'{host} (AUC={auc:.3f})')

ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curves — XGBoost (One-vs-Rest)')
ax2.legend()
ax2.grid(alpha=0.3)
plt.tight_layout()
fig2.savefig(f'{OUTPUT_DIR}/02_roc_curves.png')
plt.close(fig2)

# ── Figure 3: Top feature comparison ──
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))

for ax, imp_df, title, col in [(axes3[0], svm_importance.head(12),
                                  'SVM — Top Features', '#3498db'),
                                 (axes3[1], xgb_importance.head(12),
                                  'XGBoost — Top Features', '#e74c3c')]:
    ax.barh(range(len(imp_df)), imp_df.iloc[:, 1].values, color=col, alpha=0.8)
    ax.set_yticks(range(len(imp_df)))
    ax.set_yticklabels(imp_df['feature'].values)
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
fig3.savefig(f'{OUTPUT_DIR}/03_feature_importance.png')
plt.close(fig3)

# ── Figure 4: Host-specific signature visualization ──
fig4, ax4 = plt.subplots(figsize=(10, 6))

host_stats = []
for host in HOST_CLASSES:
    host_seqs = [s for (_, s), h in zip(all_genomes, all_labels) if h == host]
    for s in host_seqs[:50]:
        gc = (s.count('G') + s.count('C')) / len(s)
        cpg = s.count('CG') / (len(s) - 1) * 100
        host_stats.append({'host': host, 'GC_content': gc, 'CpG_per_100nt': cpg})

stat_df = pd.DataFrame(host_stats)

colors = {'Mammal': '#e74c3c', 'Bird': '#3498db', 'Plant': '#2ecc71', 'Bacteria': '#f39c12'}
for host in HOST_CLASSES:
    subset = stat_df[stat_df['host'] == host]
    ax4.scatter(subset['GC_content'], subset['CpG_per_100nt'],
                c=colors[host], label=host, alpha=0.6, s=40)

ax4.set_xlabel('GC Content')
ax4.set_ylabel('CpG per 100 nt')
ax4.set_title('Viral Genomes Cluster by Host in GC-CpG Space')
ax4.legend()
ax4.grid(alpha=0.3)
plt.tight_layout()
fig4.savefig(f'{OUTPUT_DIR}/04_gc_cpg_clustering.png')
plt.close(fig4)

print(f"\n  Figures saved to '{OUTPUT_DIR}/':")
print(f"    01_confusion_matrices.png  — SVM vs XGBoost")
print(f"    02_roc_curves.png          — ROC curves (one-vs-rest)")
print(f"    03_feature_importance.png  — Top predictive features")
print(f"    04_gc_cpg_clustering.png   — Host clustering in GC-CpG space")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("PROJECT SUMMARY")
print("=" * 65)
print(f"""
  Reference:  Young et al. (2020) PLoS Comput Biol
              Alam et al. (2025) BMC Biology 23:324

  Problem:    Predict host taxonomy from viral genome sequences
  Classes:    {', '.join(HOST_CLASSES)}
  Genomes:    {len(all_genomes)} ({GENOME_LENGTH} nt each)
  Features:   {len(feature_names)} (k-mer, dinucleotide, GC, Ti/Tv)

  SVM Accuracy:      {svm_acc:.3f}
  XGBoost Accuracy:  {xgb_acc:.3f}

  Key biological finding:
    GC content and CpG dinucleotide frequency are the strongest
    predictors of viral host class. Mammalian viruses show
    strong CpG suppression, while bacteriophages are AT-rich.

  Match to paper:
    Young et al. used SVM with linear kernel on k-mer features
    to predict host taxonomy. This project reproduces and extends
    that approach with additional feature types and XGBoost.
""")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# EXTENSIONS
# ─────────────────────────────────────────────────────────────────────────────
#
# 1. Use REAL viral genome data:
#    - Virus-Host DB: https://www.genome.jp/virushostdb/
#    - NCBI RefSeq viral genomes via Bio.Entrez
#    - GISAID for influenza data
#
# 2. Add more host classes: Invertebrate, Fungal, Archaeal
#
# 3. Try other feature types from Young et al.:
#    - Physicochemical properties (amino acid composition)
#    - PFam domain presence/absence
#    - Codon usage bias (relative synonymous codon usage)
#
# 4. Phylogenetic correction:
#    - Use ANI (average nucleotide identity) filter to avoid
#      overestimating accuracy due to closely related viruses
#    - Implement holdout method as in Young et al.
#
# 5. Deep learning approach:
#    - Use a 1D CNN directly on one-hot encoded sequences
#    - Compare with SVM and XGBoost results
#
# 6. Regression task:
#    - Predict continuous host traits (body mass, lifespan)
#      rather than categorical taxonomy
# ─────────────────────────────────────────────────────────────────────────────
