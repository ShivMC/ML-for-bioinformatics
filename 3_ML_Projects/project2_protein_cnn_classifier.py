"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PROJECT 2 (INTERMEDIATE) — DAY 2                                          ║
║  Protein Family Classification with 1D Convolutional Neural Network        ║
╚══════════════════════════════════════════════════════════════════════════════╝

────────────────────────────────────────────────────────────────────────────
IDEA & BIOLOGICAL CONTEXT
────────────────────────────────────────────────────────────────────────────

Proteins are grouped into families based on sequence, structural, and
functional similarity. Families like kinases, GPCRs, proteases, and
transcription factors share conserved sequence motifs (short patterns of
amino acids) that are hallmarks of their function.

Classifying an unknown protein into its family is a fundamental task in
structural bioinformatics. Traditional methods use BLAST or HMMER (profile
HMMs). Here, we use a 1D Convolutional Neural Network (1D-CNN) that
*learns* these motif-like patterns directly from raw sequence data —
analogous to how CNNs learn edges and shapes in images.

A 1D-CNN slides learnable filters (kernels) along the protein sequence,
detecting conserved patterns regardless of their position. This makes it
ideal for motif discovery in biological sequences.

Real-world applications:
  - Annotating proteins from newly sequenced genomes
  - Identifying remote homologs (distant evolutionary relationships)
  - Characterizing proteins of unknown function in metagenomic datasets

────────────────────────────────────────────────────────────────────────────
ML FRAMEWORK & APPROACH
────────────────────────────────────────────────────────────────────────────

Architecture:  1D Convolutional Neural Network
               ┌──────────────────────────────────────────┐
               │ Input: one-hot encoded protein sequence  │
               │ → Conv1D(64, kernel=5, ReLU)             │
               │ → MaxPool1D(kernel=2)                    │
               │ → Conv1D(128, kernel=3, ReLU)            │
               │ → GlobalMaxPool1D                        │
               │ → Dropout(0.3)                           │
               │ → Dense(64, ReLU)                        │
               │ → Dense(n_classes, Softmax)              │
               └──────────────────────────────────────────┘

Framework:    PyTorch

Type:         Multi-class supervised classification

Amino acids:  20 standard + 1 unknown = 21 categories → one-hot vectors of
              length 21 per position. Sequence padded/cropped to fixed length.

Evaluation:   Accuracy, per-class F1, confusion matrix

────────────────────────────────────────────────────────────────────────────
DATASET (Synthetic — with conserved motifs)
────────────────────────────────────────────────────────────────────────────

We simulate 6 protein families, each defined by 2-3 conserved motifs embedded
in a random background:
  - Family 0: Kinase-like      — motifs: GXGXXG, VAVK, HRDL
  - Family 1: GPCR-like        — motifs: GNXXV, DRY, NPXXY
  - Family 2: Protease-like    — motifs: GDSGG, SCFS, VVAG
  - Family 3: Ion channel-like — motifs: TVGYG, PXP, AGF
  - Family 4: Helicase-like    — motifs: GKT, DEAH, SAT
  - Family 5: Zinc finger-like — motifs: CXXC, HXXH, CPXCG

Each protein is ~200-300 amino acids long with motifs inserted at random
positions. We generate 300 sequences per family.

You can swap this for real UniProt/Swiss-Prot data using Bio.Entrez.

────────────────────────────────────────────────────────────────────────────
HOW TO RUN
────────────────────────────────────────────────────────────────────────────
  pip install biopython torch numpy pandas scikit-learn matplotlib
  python project2_protein_cnn_classifier.py
────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from collections import Counter
import random
import warnings
import os
warnings.filterwarnings('ignore')

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Generate synthetic protein sequences with family-specific motifs
# ─────────────────────────────────────────────────────────────────────────────

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
NUM_AA = len(AMINO_ACIDS)  # 20

FAMILIES = {
    'Kinase':       {'motifs': ['GXGXXG', 'VAVK', 'HRDL'],         'weight': 1.0},
    'GPCR':         {'motifs': ['GNXXV', 'DRY',   'NPXXY'],        'weight': 1.0},
    'Protease':     {'motifs': ['GDSGG', 'SCFS',  'VVAG'],         'weight': 1.0},
    'Ion_Channel':  {'motifs': ['TVGYG', 'PXP',   'AGF'],          'weight': 1.0},
    'Helicase':     {'motifs': ['GKT',   'DEAH',  'SAT'],          'weight': 1.0},
    'Zinc_Finger':  {'motifs': ['CXXC',  'HXXH',  'CPXCG'],        'weight': 1.0},
}

MAX_LEN = 250
SEQ_PER_FAMILY = 300
FAMILY_NAMES = list(FAMILIES.keys())
N_CLASSES = len(FAMILY_NAMES)


def make_motif_pattern(motif_template):
    """Expand a motif template (X = any amino acid) into a concrete sequence."""
    result = []
    for c in motif_template:
        if c == 'X':
            result.append(random.choice(AMINO_ACIDS))
        else:
            result.append(c)
    return ''.join(result)


def generate_family_sequence(motifs, seq_length, weight=1.0):
    """Generate a protein sequence containing the given motifs at random positions."""
    # Start with random background
    seq = [random.choice(AMINO_ACIDS) for _ in range(seq_length)]

    # Insert concrete motifs at random positions
    concrete_motifs = [make_motif_pattern(m) for m in motifs]
    total_motif_len = sum(len(m) for m in concrete_motifs)

    if total_motif_len >= seq_length:
        # For very short sequences, just concatenate motifs
        return ''.join(concrete_motifs[:seq_length])

    # Place motifs without overlapping
    positions = sorted(random.sample(range(seq_length - max(len(m) for m in concrete_motifs)), len(concrete_motifs)))

    for pos, motif in zip(positions, concrete_motifs):
        for i, aa in enumerate(motif):
            if pos + i < seq_length:
                seq[pos + i] = aa

    return ''.join(seq)


def one_hot_encode(seq, max_len=MAX_LEN):
    """One-hot encode a protein sequence, padding or truncating to max_len."""
    seq = seq[:max_len]
    encoding = np.zeros((max_len, NUM_AA), dtype=np.float32)
    for i, aa in enumerate(seq):
        if aa in AA_TO_IDX:
            encoding[i, AA_TO_IDX[aa]] = 1.0
    return encoding


sequences = []
labels = []

for fam_idx, family_name in enumerate(FAMILY_NAMES):
    motifs = FAMILIES[family_name]['motifs']
    weight = FAMILIES[family_name]['weight']
    for _ in range(SEQ_PER_FAMILY):
        seq_len = random.randint(200, 300)
        seq = generate_family_sequence(motifs, seq_len, weight)
        sequences.append(seq)
        labels.append(fam_idx)

print("=" * 65)
print("PROJECT 2: Protein Family Classification with 1D CNN")
print("=" * 65)

print(f"\nGenerated {len(sequences)} protein sequences:")
for fam_idx, fam_name in enumerate(FAMILY_NAMES):
    count = sum(1 for l in labels if l == fam_idx)
    print(f"  {fam_name}: {count} sequences")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: One-hot encode and split
# ─────────────────────────────────────────────────────────────────────────────

print(f"\nOne-hot encoding sequences (max length = {MAX_LEN})...")
X = np.array([one_hot_encode(seq) for seq in sequences])  # (N, MAX_LEN, 20)
y = np.array(labels)

# Transpose for PyTorch Conv1D: (N, C, L) → (batch, channels, length)
X = X.transpose(0, 2, 1)  # (N, 20, MAX_LEN)

# Shuffle and split
indices = np.random.permutation(len(X))
split = int(0.8 * len(X))
train_idx, test_idx = indices[:split], indices[split:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"Train: {X_train.shape[0]} samples")
print(f"Test:  {X_test.shape[0]} samples")

# Convert to PyTorch tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Define 1D CNN Architecture
# ─────────────────────────────────────────────────────────────────────────────

class ProteinCNN(nn.Module):
    """
    1D CNN for protein sequence classification.

    Architecture:
      - Two Conv1D layers with ReLU and MaxPool for motif detection
      - Global Max Pooling to capture the most salient motif per filter
      - Fully connected layers for classification
    """
    def __init__(self, n_channels=20, n_classes=6):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=n_channels, out_channels=64, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.global_pool(x).squeeze(-1)  # (batch, 128)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


model = ProteinCNN(n_channels=NUM_AA, n_classes=N_CLASSES).to(device)
print(f"\nModel architecture:\n{model}")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Training loop
# ─────────────────────────────────────────────────────────────────────────────

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

n_epochs = 20
train_losses, test_losses = [], []
train_accs, test_accs = [], []

print(f"\n{'─' * 55}")
print(f"Training for {n_epochs} epochs...")
print(f"{'─' * 55}")

for epoch in range(n_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    train_loss = running_loss / total
    train_acc = correct / total

    # Evaluation
    model.eval()
    test_loss, test_correct, test_total = 0.0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            test_total += y_batch.size(0)
            test_correct += (predicted == y_batch).sum().item()

    test_loss /= test_total
    test_acc = test_correct / test_total

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:2d}/{n_epochs}  |  Train Loss: {train_loss:.4f}  |  "
              f"Train Acc: {train_acc:.4f}  |  Test Acc: {test_acc:.4f}")

print(f"{'─' * 55}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Final Evaluation
# ─────────────────────────────────────────────────────────────────────────────

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y_batch.numpy())

final_acc = accuracy_score(all_labels, all_preds)
print(f"\n{'=' * 50}")
print(f"  Final Test Accuracy: {final_acc:.4f}")
print(f"{'=' * 50}")

print(f"\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=FAMILY_NAMES))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Visualization
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Training curves
axes[0].plot(train_accs, label='Train Acc', lw=2)
axes[0].plot(test_accs, label='Test Acc', lw=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Training vs Test Accuracy')
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_ylim(0, 1.05)

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=[n[:10] for n in FAMILY_NAMES],
            yticklabels=[n[:10] for n in FAMILY_NAMES])
axes[1].set_title(f'Confusion Matrix (Accuracy: {final_acc:.3f})')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('project2_cnn_results.png', dpi=150)
print(f"\n[Saved] project2_cnn_results.png")
# plt.show()

print("\n" + "=" * 65)
print("PROJECT 2 COMPLETE")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# EXTENSIONS (for further exploration)
# ─────────────────────────────────────────────────────────────────────────────
#
# 1. Use real UniProt data:
#    from Bio import ExPASy, SwissProt
#    handle = ExPASy.get_sprot_raw('P04637')  # TP53 human
#    record = SwissProt.read(handle)
#
# 2. Replace synthetic motifs with real PROSITE patterns
#    (https://prosite.expasy.org/)
#
# 3. Add attention mechanism after conv layers to visualize which positions
#    the model focuses on (biological motif discovery)
#
# 4. Try residue embedding (nn.Embedding) instead of one-hot — reduces
#    dimensionality and can learn amino acid similarity
#
# 5. Predict enzymatic function (EC number) from sequence
#
# 6. Use protein language model features (from ProtBERT/ESM) as input to
#    the CNN instead of raw sequence
# ─────────────────────────────────────────────────────────────────────────────
