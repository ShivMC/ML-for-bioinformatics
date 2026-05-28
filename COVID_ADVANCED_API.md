# Covid_virus_advanced.py — Function API Reference

Auto-generated from docstrings and code structure.

---

## Global Constants

### `GENE_BOUNDARIES`
```python
{
    'ORF1ab':    (266,   21555),   # Replicase polyprotein
    'Spike':     (21563, 25384),   # Surface glycoprotein
    'ORF3a':     (25393, 26220),   # Accessory protein
    'E':         (26245, 26472),   # Envelope protein
    'M':         (26523, 27191),   # Membrane protein
    'ORF6':      (27202, 27387),
    'ORF7a':     (27394, 27759),
    'ORF7b':     (27756, 27887),
    'ORF8':      (27894, 28259),
    'N':         (28274, 29533),   # Nucleocapsid
    'ORF10':     (29558, 29674),
}
```

### `SPIKE_DOMAINS_AA`
```python
{
    'NTD':              (14,   305),  # N-terminal domain
    'RBD':              (319,  541),  # Receptor binding domain
    'SD1':              (542,  591),  # Subdomain 1
    'SD2':              (592,  647),  # Subdomain 2
    'S1_S2_cleavage':   (675,  695),  # Furin cleavage site
    'FP':               (788,  806),  # Fusion peptide
    'HR1':              (912,  984),  # Heptad repeat 1
    'HR2':              (1163, 1213), # Heptad repeat 2
    'TM':               (1213, 1237), # Transmembrane domain
}
```

### `AA_PROPERTIES`
Each of the 20 amino acids mapped to:
```python
{
    'hydropathy': float,   # Kyte-Doolittle scale (-4.5 to 4.5)
    'charge':     int,     # -1, 0, or +1
    'volume':     float,   # van der Waals volume (A^3)
    'group':      str,     # nonpolar, polar, basic, acidic, aromatic, stop
}
```

### `CODON_TABLE`
64 DNA codons → single-letter amino acid code (e.g. `'ATG'` → `'M'`).

---

## Functions

### Data Generation

#### `generate_synthetic_genomes(n_per_variant=50)`
| | |
|---|---|
| **Purpose** | Generate synthetic SARS-CoV-2 genomes with variant-specific mutations |
| **Input** | `n_per_variant` (int) — sequences per variant group |
| **Output** | `(reference, genomes_list, metadata_df)` |
| **Variants** | Wuhan_2020, Alpha_2021, Delta_2021, Omicron_2022 |

---

### Mutation Analysis

#### `map_gene(pos)`
| | |
|---|---|
| **Purpose** | Map 1-indexed genomic position to SARS-CoV-2 gene |
| **Input** | `pos` (int) — genomic position |
| **Output** | Gene name (str) or `'Intergenic'` |

#### `translate_codon(codon)`
| | |
|---|---|
| **Purpose** | Translate 3-nt DNA codon to amino acid |
| **Input** | `codon` (str) — 3 nucleotides |
| **Output** | Single-letter amino acid code (str) or `'X'` |

#### `classify_mutation(ref_nt, alt_nt)`
| | |
|---|---|
| **Purpose** | Classify nucleotide substitution type |
| **Input** | `ref_nt`, `alt_nt` (str) — single nucleotides |
| **Output** | `'transition'` or `'transversion'` |

#### `get_spike_domain(aa_pos)`
| | |
|---|---|
| **Purpose** | Map Spike amino acid position to functional domain |
| **Input** | `aa_pos` (int) — Spike protein position |
| **Output** | Domain name (str) or `'Other'` |

#### `mutation_stability_score(wt_aa, mut_aa)`
| | |
|---|---|
| **Purpose** | Score mutation destabilization from AA property changes |
| **Input** | `wt_aa` (str), `mut_aa` (str) — single-letter codes |
| **Output** | `float` (0–5), higher = more destabilizing |
| **Formula** | `0.3×|Δhydropathy| + |Δvolume|/200 + charge_penalty(1.0) + group_penalty(0.5)` |

---

### ML Models (Section 6)

#### XGBClassifier (variant classification)
```python
params = {
    'n_estimators': 150,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
}
```
**Task:** 4-class variant classification (Wuhan, Alpha, Delta, Omicron)

#### XGBRegressor (year prediction)
```python
params = {
    'n_estimators': 150,
    'max_depth': 4,
    'learning_rate': 0.1,
}
```
**Task:** Regression of collection year from mutation profile

---

### Data Structures (major)

#### `mut_df` (DataFrame)
Per-mutation detail with 15 columns (see Section 4 of main docs).

#### `presence_df` (DataFrame)
Binary mutation presence matrix: samples × mutations (0/1), merged with metadata.

#### `dnds_df` (DataFrame)
Selection analysis: gene, variant, n_syn, n_non, dnds ratio, selection class.

#### `cooc_graph` (networkx.Graph)
Mutation co-occurrence network with edge weights = co-occurrence rate.

---

## Output Files

| File | Dependency | When Generated |
|---|---|---|
| `covid_advanced_output/01_mutation_burden.png` | burden_data exists | Always |
| `covid_advanced_output/02_spike_domain_heatmap.png` | spike_muts > 0 | Always |
| `covid_advanced_output/03_temporal_trajectories.png` | importance_df non-empty | Always |
| `covid_advanced_output/04_cooccurrence_network.png` | cooc_graph has edges | If co-occurrences ≥ threshold |
| `covid_advanced_output/05_feature_importance.png` | importance_df non-empty | Always |
| `covid_advanced_output/06_dnds_heatmap.png` | dnds_df non-empty | If dN/dS pairs found |
