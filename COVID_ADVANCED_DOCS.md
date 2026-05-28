# Covid_virus_advanced.py — Documentation

## 1. Overview

`Covid_virus_advanced.py` is a complete, self-contained SARS-CoV-2 genomic analysis pipeline that extends the original `Covid_virus.py` with **protein-level analysis, evolutionary selection metrics, graph-based mutation networks, and gradient-boosted ML models**.

The pipeline generates synthetic SARS-CoV-2 genomes with variant-specific mutation profiles (Wuhan, Alpha, Delta, Omicron), calls mutations at nucleotide and amino acid levels, then runs 9 analysis modules culminating in 6 publication-quality figures.

---

## 2. Quick Start

```bash
# Install dependencies
pip install biopython numpy pandas scikit-learn xgboost networkx matplotlib seaborn

# Run
python Covid_virus_advanced.py
```

All output goes to `covid_advanced_output/`.

---

## 3. Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  1. DATA GENERATION                                              │
│  generate_synthetic_genomes()                                    │
│  → 200 genomes, 4 variants, years 2020-2026                     │
└──────────────────────────┬───────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│  2. MUTATION CALLING + PROTEIN TRANSLATION                       │
│  map_gene() | translate_codon() | classify_mutation()            │
│  mutation_stability_score() | get_spike_domain()                 │
│  → mut_df: per-mutation details (nucleotide + amino acid level)  │
└──────────────────────────┬───────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│  3. MUTATION TYPE PROFILING                                      │
│  Transition/Transversion ratio, Synonymous/Non-synonymous ratio  │
└──────────────────────────┬───────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│  4. dN/dS SELECTION ANALYSIS                                     │
│  Per gene, per variant: positive / neutral / purifying selection  │
└──────────────────────────┬───────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│  5. TEMPORAL FREQUENCY ANALYSIS                                  │
│  Mutation presence matrix → yearly frequency tracking            │
└──────────────────────────┬───────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│  6. ADVANCED ML (XGBoost)                                        │
│  6A: Variant of Concern Classification                           │
│  6B: Temporal Year Regression (trend forecasting)                │
│  6C: Emerging Mutation Detection (frequency shift)               │
└──────────────────────────┬───────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│  7. SPIKE PROTEIN DOMAIN ANALYSIS                                │
│  NTD | RBD | SD1 | SD2 | S1/S2 | FP | HR1 | HR2 | TM            │
│  Domain-level mutation density + stability scores                │
└──────────────────────────┬───────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│  8. MUTATION CO-OCCURRENCE NETWORK                                │
│  Graph of mutations that appear together → community detection   │
└──────────────────────────┬───────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│  9. VISUALIZATION (6 figures)                                    │
│  01_mutation_burden.png    02_spike_domain_heatmap.png           │
│  03_temporal_trajectories.png  04_cooccurrence_network.png       │
│  05_feature_importance.png  06_dnds_heatmap.png                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. Section-by-Section Breakdown

### Section 1 — Data Generation
**Function:** `generate_synthetic_genomes(n_per_variant=50)`

Generates synthetic SARS-CoV-2 genomes by taking a reference sequence (first 5000 nt of NC_045512.2) and applying:

- **Signature mutations** — variant-defining mutations for 4 variants:
  - `Wuhan_2020` — no signature mutations (ancestral)
  - `Alpha_2021` — 5 mutations including `A144T`, `C978T`
  - `Delta_2021` — 5 mutations including `A229G`, `T523T`
  - `Omicron_2022` — 10 mutations across the genome
- **Background mutations** — 5–20 random mutations per genome
- **Year simulation** — each sample gets year = variant year + random(0–5), capped at 2026

**Returns:** `(reference_seq, genomes_list, metadata_df)`

| Field | Type | Description |
|---|---|---|
| `reference` | str | Reference genome sequence (5000 nt) |
| `genomes_list` | list[dict] | Each dict: `id`, `seq`, `variant`, `year` |
| `meta_df` | DataFrame | Sample metadata: `id`, `variant`, `year` |

---

### Section 2 — Mutation Calling + Protein Translation

Key global data structures:

| Name | Type | Description |
|---|---|---|
| `AA_PROPERTIES` | dict | 20 amino acids × 4 properties (hydropathy, charge, volume, group) |
| `CODON_TABLE` | dict | 64 codons → amino acid translation |
| `TRANSITIONS` | set | 4 transition pairs: A↔G, C↔T |

**Functions:**

#### `map_gene(pos)`
- **Input:** `pos` (int) — 1-indexed genomic position
- **Output:** gene name (str) or `'Intergenic'`
- **Logic:** Checks against `GENE_BOUNDARIES` dict (11 SARS-CoV-2 genes)

#### `translate_codon(codon)`
- **Input:** 3-nt DNA string
- **Output:** single-letter amino acid code or `'X'` for unknown

#### `classify_mutation(ref_nt, alt_nt)`
- **Input:** reference and alternate nucleotides
- **Output:** `'transition'` or `'transversion'`

#### `get_spike_domain(aa_pos)`
- **Input:** amino acid position in Spike protein
- **Output:** domain name (NTD, RBD, SD1, SD2, S1_S2_cleavage, FP, HR1, HR2, TM, Other)

#### `mutation_stability_score(wt_aa, mut_aa)`
- **Input:** wild-type and mutant amino acid
- **Output:** float (0–5), higher = more destabilizing
- **Formula:**
  - `Δhydropathy × 0.3`
  - `+ Δvolume / 200`
  - `+ 1.0` if charge changes
  - `+ 0.5` if group changes (e.g. nonpolar → charged)

**Output DataFrame (`mut_df`):**

| Column | Type | Description |
|---|---|---|
| `id` | str | Sample identifier |
| `year` | int | Collection year |
| `variant` | str | Variant name |
| `mutation` | str | Format: `{ref}{pos}{alt}` (e.g. A23403G) |
| `position` | int | 1-indexed genomic position |
| `ref_nt` | str | Reference nucleotide |
| `alt_nt` | str | Alternate nucleotide |
| `mut_type` | str | transition / transversion |
| `gene` | str | SARS-CoV-2 gene name |
| `wt_aa` | str | Wild-type amino acid (if coding) |
| `mut_aa` | str | Mutant amino acid (if coding) |
| `aa_pos` | int | Amino acid position (if coding) |
| `synonymous` | bool | True if silent mutation |
| `stability_score` | float | 0–5 destabilization proxy |

---

### Section 3 — Mutation Type Profiling

Calculates:
- **Transition/Transversion ratio** — biological signal: Ti/Tv ~2–3 for recent SARS-CoV-2 evolution
- **Synonymous/Non-synonymous ratio** — used as input to dN/dS

---

### Section 4 — dN/dS Selection Analysis

**Method:** For each gene–variant pair with >5 mutations:
- `dNdS = (n_non + 0.5) / (n_syn + 0.5)` (pseudocount correction)
- Classification:
  - `dNdS > 2.0` = **positive selection** (adaptive evolution)
  - `dNdS < 0.5` = **purifying selection** (conserved)
  - Otherwise = **neutral**

**Output:** `dnds_df` DataFrame → saved as heatmap in Figure 6

---

### Section 5 — Temporal Frequency Analysis

Constructs a **binary mutation presence matrix** (`presence_df`):
- Rows = samples
- Columns = unique mutations
- Values = 0 (absent) / 1 (present)
- Merged with metadata (year, variant)

Computes per-year mutation frequencies and lists top-5 mutations per year.

---

### Section 6 — Advanced ML (XGBoost)

#### 6A: Variant of Concern Classification
- **Model:** `XGBClassifier(n_estimators=150, max_depth=6)`
- **Task:** Classify genome into Wuhan_2020 / Alpha_2021 / Delta_2021 / Omicron_2022
- **Evaluation:** Accuracy, per-class precision/recall/F1
- **Output:** Feature importance ranking — which mutations define each variant?

#### 6B: Temporal Year Regression
- **Model:** `XGBRegressor(n_estimators=150, max_depth=4)`
- **Task:** Predict collection year from mutation profile
- **Evaluation:** Mean Absolute Error (years)
- **Biological interpretation:** Mutation patterns encode temporal signal — the virus accumulates specific mutations over time

#### 6C: Emerging Mutation Detection
- Compares mutation frequency between earliest and latest years
- Ranks by frequency shift `Δ = freq_late − freq_early`
- Identifies mutations that are "emerging" (rising in prevalence)

---

### Section 7 — Spike Protein Domain Analysis

Filters `mut_df` to Spike gene mutations, maps each to a Spike domain, then computes:

- **Mutation density** — number of mutations per domain
- **Unique positions** — how many distinct residues are mutated
- **Mean stability score** — average destabilization proxy
- **RBD hotspots** — positions in the Receptor Binding Domain with highest mutation counts

Visualized as a heatmap (Figure 2): rows = Spike domains, columns = variants, cell = mean stability score.

---

### Section 8 — Mutation Co-occurrence Network

Builds a graph where:
- **Nodes** = top-20 most important mutations (from XGBoost feature importance)
- **Edges** = pairs that co-occur in ≥60% of genomes where either is present

**Community detection** via `networkx.algorithms.community.greedy_modularity_communities()` — finds groups of mutations that travel together (potential epistatic interactions or lineage-defining modules).

Visualized in Figure 4 with spring layout, edge width proportional to co-occurrence rate.

---

### Section 9 — Visualization

| Figure | File | Description |
|---|---|---|
| 1 | `01_mutation_burden.png` | Bar chart of average mutation count per genome by year with error bars |
| 2 | `02_spike_domain_heatmap.png` | Heatmap of mean stability impact score per Spike domain × variant |
| 3 | `03_temporal_trajectories.png` | Line plot of top-8 mutation frequencies over time |
| 4 | `04_cooccurrence_network.png` | Network graph of mutation co-occurrence with communities |
| 5 | `05_feature_importance.png` | Horizontal bar chart of top-15 XGBoost feature importances |
| 6 | `06_dnds_heatmap.png` | Heatmap of dN/dS ratios per gene × variant |

All figures saved at 150 DPI in `covid_advanced_output/`.

---

## 5. Configuration

Modify these constants at the top of the script:

```python
OUTPUT_DIR = 'covid_advanced_output'      # Output directory
GENE_BOUNDARIES = { ... }                 # 11 SARS-CoV-2 genes
SPIKE_DOMAINS_AA = { ... }                # 9 Spike domains
cooc_threshold = 0.6                      # Co-occurrence rate threshold
```

---

## 6. Using Real Sequence Data

Replace `generate_synthetic_genomes()` with real FASTA data via Biopython Entrez:

```python
from Bio import Entrez, SeqIO

Entrez.email = "your@email.com"

# Fetch reference
handle = Entrez.efetch(db="nucleotide", id="NC_045512", rettype="fasta")
REFERENCE = str(SeqIO.read(handle, "fasta").seq)

# Search for sequences
search = Entrez.esearch(db="nucleotide",
    term="SARS-CoV-2[Organism] AND 29000:30000[SLEN]",
    retmax=100)
id_list = Entrez.read(search)["IdList"]

# Fetch and parse
handle = Entrez.efetch(db="nucleotide", id=id_list, rettype="fasta")
records = list(SeqIO.parse(handle, "fasta"))

# Format as genomes_list
genomes_list = [{
    'id': r.id,
    'seq': str(r.seq),
    'variant': 'Unknown',
    'year': extract_year_from_description(r.description)
} for r in records]
```

---

## 7. Comparison: Original vs Advanced

| Feature | Original `Covid_virus.py` | Advanced `Covid_virus_advanced.py` |
|---|---|---|
| Environment | Google Colab only | Standalone Python |
| Data source | Manual FASTA upload | Synthetic + real NCBI option |
| Alignment | MAFFT (external) | Not needed (synthetic) |
| ML model | Random Forest | XGBoost (classifier + regressor) |
| Protein translation | No | Yes — all coding genes |
| Spike domains | No | 9 domains mapped |
| dN/dS selection | No | Yes |
| Co-occurrence network | No | Yes |
| Stability scoring | No | AA property-based |
| Transition/transversion | No | Yes |
| Figures | 2 (PCA + KMeans) | 6 (publication quality) |
| Mutation types | N/A | Synonymous/non-synonymous |
| Temporal forecasting | Basic mean | XGBoost regression + emerging mutations |

---

## 8. Dependencies

| Package | Version (min) | Purpose |
|---|---|---|
| `numpy` | 1.21 | Numerical arrays |
| `pandas` | 1.3 | DataFrames |
| `scikit-learn` | 1.0 | Train/test split, metrics |
| `xgboost` | 1.5 | Gradient-boosted ML models |
| `networkx` | 2.6 | Mutation co-occurrence graph |
| `matplotlib` | 3.4 | Plotting |
| `seaborn` | 0.11 | Statistical visualizations |
| `biopython` | 1.79 | (optional) Real data via Entrez |

---

## 9. Extension Guide

### Add 3D structural mapping
```python
from Bio.PDB import PDBParser
parser = PDBParser()
structure = parser.get_structure("6M0J", "6m0j.pdb")  # Spike RBD
# Overlay mutation positions on 3D structure
```

### Add FoldX/Rosetta stability prediction
- Export mutated Spike sequences as PDB
- Run FoldX `BuildModel` or Rosetta `ddg_monomer`
- Compare with the AA property-based stability scores

### Integrate ESM-2 protein language model
```python
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
# Score each Spike mutation → compare with dN/dS and stability score
```

### Replace with real deep mutational scanning data
- Use ProteinGym (https://proteingym.org/) substitutions for Spike RBD
- Compare ML-predicted mutation effects against experimental measurements

---

## 10. File Listing

```
Covid_virus_advanced.py          — Main pipeline script (741 lines)
covid_advanced_output/
├── 01_mutation_burden.png
├── 02_spike_domain_heatmap.png
├── 03_temporal_trajectories.png
├── 04_cooccurrence_network.png
├── 05_feature_importance.png
└── 06_dnds_heatmap.png
```
