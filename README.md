# Pattern Discovery in Alzheimer's Disease
### A Knowledge Discovery and Data Mining Approach

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.1.0-150458?style=flat&logo=pandas&logoColor=white)
![mlxtend](https://img.shields.io/badge/mlxtend-0.22.0-blue?style=flat)
![Accuracy](https://img.shields.io/badge/Accuracy-93.8%25-brightgreen?style=flat)
![Rules](https://img.shields.io/badge/Association%20Rules-39-orange?style=flat)
![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat)

Alzheimer’s Disease is a progressive neurodegenerative disorder affecting millions worldwide. Early detection is vital, yet many cases go undiagnosed until symptoms worsen. This study applies a data‑mining pipeline to patient data with 35 clinical and lifestyle features to identify patterns linked to AD.

This is a comprehensive **knowledge discovery and data mining (KDD) pipeline** applied to a real-world
Alzheimer's Disease dataset of **2,149 patients and 35 clinical features** — comparing unsupervised
and supervised learning methods to discover clinically actionable patterns for early AD detection.

> **Team:** Cynthia Mutua · Halee Belghouthi · Fedi Naimi · Jhansi Nalla
> **Course:** CIS 635 — Knowledge Discovery and Data Mining, Grand Valley State University
> 
> **Cynthia's contributions:** K-means implementation, 4 clustering validation metrics, cluster
> profiling, PCA visualization (42 commits)

---

## Key Findings

| Finding | Result |
|---|---|
| Decision Tree Accuracy | **93.8%** (F1: 0.912) |
| Association Rules Discovered | **39 rules** (confidence up to 84%, lift up to 2.36) |
| Strongest Predictive Pattern | MemoryComplaints + Severe MMSE → AD (conf=84%, lift=2.36) |
| Clustering Result | Weak separation (silhouette ≈ 0.06) → **AD lies on a continuum** |
| Top Predictive Feature | FunctionalAssessment (23.3% tree importance) |
| Convergent Features | MemoryComplaints & BehavioralProblems across all methods |

---

## Analytical Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAW DATASET                                   │
│                2,149 patients · 35 features                          │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       PREPROCESSING                                  │
│  Median imputation · Z-score scaling · Feature discretization        │
│  Stratified 70-30 split (χ²=0.001, p=0.975)                        │
└──────────┬──────────────────────────────────┬───────────────────────┘
           │                                  │
           ▼                                  ▼
┌──────────────────────┐          ┌───────────────────────┐
│   UNSUPERVISED        │          │    SUPERVISED          │
│                      │          │                        │
│  1. EDA              │          │  4. Decision Tree      │
│     Correlation      │          │     Grid search        │
│     analysis         │          │     depth ∈ {3,5,7,10} │
│                      │          │     93.8% accuracy     │
│  2. K-Means          │          │     0.912 F1-score     │
│     k=2 through 7    │          │                        │
│     Silhouette ≈0.06 │          └───────────┬───────────┘
│     → Continuum      │                      │
│                      │                      │
│  3. Association      │                      │
│     Rules (Apriori)  │                      │
│     39 rules         │                      │
│     lift up to 2.36  │                      │
└──────────┬───────────┘                      │
           │                                  │
           └──────────────┬───────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  CROSS-METHOD INTEGRATION                            │
│   Convergent features: MemoryComplaints · BehavioralProblems         │
│   Validated across correlation, association rules, and decision tree │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
Pattern-Discovery-in-Alzheimer-s-Disease/
│
├── KDD_FINAL_GROUP_PROJECT.ipynb   # Full pipeline notebook
├── data/
│   └── alzheimers_disease_data.csv # Source dataset (Kaggle)
├── outputs/
│   ├── figures/                    # All visualizations
│   └── rules/                      # Association rules export
└── README.md
```

---

## Dataset

**Source:** [Alzheimer's Disease Dataset — El Kharoua (2024), Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset)

| Property | Value |
|---|---|
| Patients | 2,149 |
| Features | 35 |
| Missing values | 0.10% |
| AD diagnosed | 35.4% (760 patients) |
| Age range | 60–90 years |

**Feature categories:** Demographics · Lifestyle · Cardiovascular · Metabolic ·
Medical History · Cognitive (MMSE) · Functional (ADL) · Behavioral/Symptomatic

---

## Results

### 1. Exploratory Data Analysis

Top 5 features correlated with AD diagnosis:

| Feature | Correlation | Direction |
|---|---|---|
| FunctionalAssessment | r = −0.36 | Protective |
| ADL | r = −0.33 | Protective |
| MemoryComplaints | r = +0.31 | Risk factor |
| MMSE | r = −0.24 | Protective |
| BehavioralProblems | r = +0.22 | Risk factor |

> No single feature perfectly predicts diagnosis (max r = 0.36), motivating multivariate methods.

---

### 2. Clustering Analysis — A Scientific Finding

We tested k-means (k=2–7) and hierarchical clustering (Ward linkage) using four validation metrics:

| k | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---|---|---|
| 2 | 0.059 | 3.921 | 135.8 |
| 3 | 0.052 | 3.495 | 116.2 |
| 5 | 0.051 | 3.040 | 96.2 |
| 7 | 0.053 | 2.753 | 84.1 |

**Result:** All metrics indicate poor cluster separation. K-means and hierarchical clustering
disagree substantially (Adjusted Rand Index = 0.025). PCA retains only 16.8% variance in 2D.

**Interpretation:** Rather than a methodological failure, this is a **meaningful scientific finding** —
AD severity in this population lies on a **continuum** rather than forming discrete phenotypes.
This aligns with the NIA-AA Research Framework (Jack et al., 2018) and biological heterogeneity
studies in ADNI (Nettiksimmons et al., 2014).

---

### 3. Association Rule Mining (Apriori)

**39 interpretable IF-THEN rules** discovered predicting AD diagnosis:

| Metric | Range | Median |
|---|---|---|
| Confidence | 60–84% | 77% |
| Lift | 1.5–2.36 | 2.1 |
| Support | 3–9% | — |

**Top Rule:**
```
IF MemoryComplaints = 1 AND MMSE_Category = Severe_Impairment
→ Diagnosis = 1
Support = 3.8% · Confidence = 84% · Lift = 2.36
```

**Most frequent antecedent features:**
1. MemoryComplaints (appears in 51% of rules)
2. MMSE_Category = Severe Impairment (31% of rules)
3. BehavioralProblems (21% of rules)

---

### 4. Decision Tree Classification

**Best configuration:** max_depth=5, min_samples_leaf=10, class_weight='balanced'

| Metric | Baseline (Majority) | Logistic Regression | **Decision Tree** |
|---|---|---|---|
| Accuracy | 64.7% | 81.6% | **93.8%** |
| F1-Score | 0.000 | 0.762 | **0.912** |
| Sensitivity | — | — | **91%** |
| Specificity | — | — | **95%** |

**Confusion Matrix (test set, n=645):**
```
              Predicted: No AD    Predicted: AD
Actual: No AD      397                20
Actual: AD          20               208
```

**Top 5 features by importance:**

| Feature | Importance |
|---|---|
| FunctionalAssessment | 23.3% |
| MMSE | 21.2% |
| ADL | 18.4% |
| BehavioralProblems | 18.3% |
| MemoryComplaints | 17.2% |

> The top 5 features account for **98.4% of total importance** — a small set of functional,
> cognitive, and behavioral measures drives nearly all predictive power.

**Sample decision path:**
```
IF FunctionalAssessment ≤ 4.97
  AND MMSE ≤ 24.02
  AND ADL ≤ 5.12
→ PREDICT: Alzheimer's Diagnosis
```

---

### 5. Cross-Method Integration

Features validated across **multiple independent methods:**

| Feature | Correlation | Assoc. Rules | Decision Tree |
|---|---|---|---|
| FunctionalAssessment | r=−0.36 (rank 1) | Rare | 23.3% (rank 1) |
| ADL | r=−0.33 (rank 2) | Rare | 18.4% (rank 3) |
| MemoryComplaints | r=+0.31 (rank 3) | 51% of rules | 17.2% (rank 5) |
| MMSE | r=−0.24 (rank 4) | 31% of rules | 21.2% (rank 2) |
| BehavioralProblems | r=+0.22 (rank 5) | 21% of rules | 18.3% (rank 4) |

**MemoryComplaints** and **BehavioralProblems** appear in both association rule antecedents
and top decision tree features — providing **convergent evidence** from independent methods.

---

## Clinical Implications

1. **Functional assessment > cognitive testing alone** — FunctionalAssessment (23.3% importance)
outranks MMSE (21.2%) as a predictor, suggesting daily living ability is a more sensitive indicator
2. **Subjective complaints matter** — Patient/caregiver-reported MemoryComplaints is the
third-strongest predictor, validating the importance of qualitative clinical intake
3. **Behavioral changes are early warning signs** — BehavioralProblems ranks 4th across methods;
family-reported behavioral changes may precede measurable cognitive decline
4. **Multivariate screening outperforms single-test approaches** — No single feature achieves
r > 0.36; combining functional, cognitive, and behavioral measures is essential

---

## How to Run

### Prerequisites
```bash
Python 3.10+
```

### Install dependencies
```bash
git clone https://github.com/cece15/Pattern-Discovery-in-Alzheimer-s-Disease-A-Knowledge-Discovery-and-Data-Mining-Approach.git
cd Pattern-Discovery-in-Alzheimer-s-Disease-A-Knowledge-Discovery-and-Data-Mining-Approach
pip install -r requirements.txt
```

### Run in Google Colab
Open the notebook directly:
👉 [ALZHEIMER'S DISEASE PATTERN DISCOVERY.ipynb](https://colab.research.google.com/drive/1Q83Mx_Dgv_biAxytV0Q9fcuJ7QvXdlj6?usp=sharing)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10 |
| Data Processing | pandas 2.1.0, NumPy 1.24.3 |
| Machine Learning | scikit-learn 1.3.0 |
| Association Rules | mlxtend 0.22.0 (Apriori) |
| Visualization | matplotlib 3.7.2, seaborn 0.12.2 |
| Environment | Google Colab |
| Version Control | Git / GitHub |

---

## References

1. El Kharoua, R. (2024). Alzheimer's Disease Dataset. Kaggle.
2. Jack, C.R., et al. (2018). NIA-AA Research Framework. *Alzheimer's & Dementia*, 14(4), 535–562.
3. Nettiksimmons, J., et al. (2014). Biological heterogeneity in ADNI. *Alzheimer's & Dementia*, 10(5), 511–521.
4. Gamberger, D., et al. (2017). Homogeneous clusters of Alzheimer's disease. *Biomedical Engineering Online*, 16(1).
5. Podgorelec, V., et al. (2002). Decision trees in medicine. *Journal of Medical Systems*, 26(5), 445–463.

---

## Team & Contributions

**Cynthia Mutua** *(Primary contributor)*
K-means clustering implementation, all 4 clustering validation metrics
(silhouette, Davies-Bouldin, Calinski-Harabasz, inertia), cluster profiling,
PCA visualization, exploratory data analysis, association rule mining,
decision tree implementation, cross-method integration, final report writing — 42 commits

**Halee Belghouthi** — Feature discretization, preprocessing pipeline

**Fedi Naimi** — Hierarchical clustering, dendrograms, chi-square validation

**Jhansi Nalla** — Repository setup, notebook consolidation, documentation

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Cynthia%20Mutua-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/cynthia-mutua)
[![GitHub](https://img.shields.io/badge/GitHub-cece15-181717?style=flat&logo=github)](https://github.com/cece15)
