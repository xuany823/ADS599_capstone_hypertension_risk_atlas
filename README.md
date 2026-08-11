# 💓 Hypertension Risk Atlas 🗺️

**ADS-599 Capstone Project — University of San Diego**
**Project Group 2:** Nancy Walker & Michelle Wang

An interpretable machine learning project that predicts county-level hypertension prevalence across the United States from place-based social and environmental determinants, and explains the specific drivers behind each county's risk.

---

## 🟢 Project Status

**Current stage: Project Complete** 🟢 

| Phase | Status |
|-------|--------|
| Team formation & proposal | Complete |
| Data acquisition (4 public sources) | Complete |
| ETL / merge to county-level dataset (3 sources) | Complete |
| Exploratory data analysis (`eda_master.ipynb`) | Complete |
| Data cleaning & feature engineering | Complete |
| CHR&R feature selection & integration | Complete |
| Modeling & evaluation | Complete |
| Interactive atlas (Streamlit) | Complete |

---

## 🎯 Problem & Goal

Hypertension affects nearly half of U.S. adults and is unevenly distributed across geography. Public health agencies repeatedly face the same question: where should limited prevention resources be directed to reduce hypertension burden most effectively?

This project reframes that question as a data science problem — predicting county-level hypertension prevalence from social, environmental, and food-access indicators — and uses interpretable machine learning (SHAP) to expose *why* each county is at risk, not just *where* risk is high.

---

## 📊 Data Sources

All four sources are public, free, and joined on the 5-digit county FIPS / GEOID.

| Source | Contribution | Size (raw) | Access |
|--------|-------------|------------|--------|
| CDC PLACES | Health outcomes incl. target (hypertension prevalence) | 3,144 counties × 24 cols | Socrata API (RSocrata) / download; no key |
| USDA Food Access Research Atlas | Food environment & access indicators | 72,531 tracts × 147 cols | Direct download; no key |
| U.S. Census Bureau (ACS) | Socioeconomic context (income, poverty, housing) | 6,444 records × 5 cols | Census Data API; free key |
| County Health Rankings (CHR&R) | Social & clinical community health metrics | 3,204 counties × 796 cols | Download / `countyhealthR`; no key |

**Active analytic dataset (`master_dataset_all_variables.csv`):** three sources — CDC PLACES, USDA FARA, and Census ACS — standardized to county-level GEOID and merged into a single table of **3,231 rows × 120 columns**. Target variable: county-level hypertension prevalence (**BPHIGH**), observed for 2,956 counties (91.5%), which is the effective sample size for modeling.

**CHR&R (not merged):** its 796 raw columns carry high per-column missingness (suppression flags) and substantial redundancy, so it requires a dedicated feature-selection pass before it can be folded into the master table without overwhelming the model. A specific release year is pinned for reproducibility (CHR&R funding concludes December 2026). Dataset used in dashboard only as a independent holdout set. 

---

## 📈 EDA Highlights (`notebooks/EDA/eda_master.ipynb`)

- **Join integrity:** zero duplicate rows or FIPS keys; USDA and Census income estimates agree at r ≈ 0.91, corroborating the merge.
- **Target (BPHIGH):** mean 33.5%, range 21%–53.1%, moderate right skew (0.83) — log-transform is a modeling candidate, not a requirement. High-prevalence outliers are genuine counties, not errors, and are retained.
- **Missingness:** no column exceeds ~29% missing; nothing qualifies for the ≥ 60% drop threshold. Gaps will be imputed per the tiered policy in `notebooks/EDA/README` (validation EDAs).
- **Leakage:** no predictor exceeds |r| = 0.95 with the target — no exclusions required.
- **Strongest predictors:** `DIABETES` (r ≈ 0.88), `MOBILITY` (r ≈ 0.86), `FOODINSECU` (r ≈ 0.84) on the risk side; `median_income` / `MedianFamilyIncome` (r ≈ −0.6) on the protective side.
- **Multicollinearity:** top predictors resolve into two clusters (CDC comorbidity/SDoH; income/food access). 77 of 78 USDA `la*`/`Tract*` columns exceed VIF > 10, many at near-exact linear dependence — the highest-priority target for feature selection or dimensionality reduction.
- **Geography:** prevalence visibly clusters by region, supporting the project's core hypothesis and motivating residual checks for geographic bias.

---

## ⚙️ Methodology

1. **Data acquisition** — Programmatic ingestion from the primary sources via R scripts (`pipeline/R_src/`) that retrieve raw data through API calls and downloads, with logging of source versions and retrieval dates. CHR&R uses a pinned annual release for reproducibility.
2. **Data preparation** — Python ETL scripts (`pipeline/DataIngestion/`) standardize each source to the 5-digit `fipscode` (GEOID, read as a zero-padded string), aggregate USDA tract-level data to counties, resolve data-grain differences, and merge into the master table; merge validation covers duplicate keys and a cross-source income consistency check (r ≈ 0.91).
3. **Exploratory data analysis** — Completed against the merged table: distribution and skewness assessment of the target, tiered missingness audit, leakage screening (|r| > 0.95), correlation and VIF analysis, and geographic pattern mapping (see EDA Highlights above).
4. **Relational integration** — The cleaned master dataset is loaded into a SQLite relational database (`hypertension_atlas.db`), enabling reproducible analysis and dashboard-ready SQL views; modeling reads the CSV directly for speed.
5. **Feature engineering and dimensionality reduction** — Resolution of the documented USDA `la*`/`Tract*` redundancy through deliberate feature selection or PCA; creation of derived features (e.g., vehicle-access rate, food-desert density, composite socioeconomic indices); Lasso/PCA applied to retain predictive variance while stabilizing the feature space. Transformations are fitted on training data only.
6. **Modeling** — Supervised regression on the 2,956 counties with an observed target (target-missing rows are dropped, never imputed). A Linear Regression baseline — supported by the roughly linear DIABETES–BPHIGH relationship — is benchmarked against tree-based ensembles (Random Forest, XGBoost), with train/test splits and cross-validation for model selection and hyperparameter tuning.
7. **Evaluation** — Performance assessed with R², RMSE, and MAE against a mean-prediction baseline on held-out data; residual analysis checks for geographic bias motivated by the regional clustering observed in the EDA.
8. **Interpretation** — SHAP (SHapley Additive exPlanations) traces predictions to place-based drivers, with attention to the two predictor clusters identified in the EDA so importance rankings reflect underlying factors rather than redundant columns.

---

## 🗂️ Repository Structure

```
Hypertension-Risk-Atlas/
├── .devcontainer/       # devcontainer.json for VS Code Remote Containers 
├── github/workflow/     # GitHub Actions workflows for CI/CD (e.g., run pipeline on push)
├── dashboard/           # Streamlit app and assets 
├── data/
│   ├── final/           # final analytic dataset (cleaned, merged, and feature-engineered)
│   ├── processed/       # datasets created by the ETL pipeline (incl. master_dataset_all_variables.csv)
│   ├── raw/             # source files (not version-controlled if large)
│   └── validation/      # validation datasets (raw source datasets with validation checks)
├── notebooks/
│   ├── 01_DataIngestion/   # .qmd files that ingest data from CDC, USDA, Census, and CHR&R
│   ├── 02_validation/      # .ipynb notebooks for validation of raw source datasets
│   ├── 03_EDA/             # .ipynb notebooks for exploratory data analysis (eda_master.ipynb active)
│   ├── 04_FeatureEngineering/  # .ipynb notebooks for feature engineering and dimensionality reduction 
│   └── 05_Modeling/         # .ipynb notebooks for model training, evaluation, and interpretation
├── pipeline/
│   ├── 01_R_src/           # R scripts to retrieve raw data via API calls and downloads
│   ├── 02_DataIngestion/  # Python cleaning & merge scripts (01_process_cdc.py, 02_process_usda.py, 03_process_census.py, 07_run_ingestion.py)
│   ├── 03_FeatureEngineering/ # Python script for feature enginnering and dimensionality reduction
│   ├── 04_generate_eda_report.py # Python script to generate EDA report (figures, tables, and summary statistics) in HTML format
│   └── 05_generate_model_report.py # Python script to generate model report (figures, tables, and summary statistics) in HTML format
├── .gitignore            # controls which files are ignored by Git (e.g., large raw data, virtual environment, etc.)
├── audit_data.py         # checks for data integrity and completeness
├── hypertension_atlas.db # SQLite database of merged datasets
├── hypertension_risk_atlas.ipynb # Summary notebook presenting the project's methodology, analyses, and key findings
├── main.py               # main script to run the entire pipeline
├── paths.py              # filepaths for data ingestion and modeling
├── README.md
├── requirements.txt      # Python dependencies
└── utils.py              # utility functions for data ingestion and modeling
```

---

## 📄 Class Deliverables (Module 7)

- **Capstone Article** — full methodology and results
- **Capstone GitHub** — documented, interview-ready repository
- **Capstone Presentation** — narrated pitch
- **Capstone User Tool** — interactive Hypertension Risk Atlas (Streamlit web app)

---

## 🛠️ Tools & Workflow

- **Languages/Environments:** R (raw data retrieval via APIs), Python (ETL, analysis & modeling), SQLite (relational integration), Jupyter Notebook, VS Code
- **Version control:** GitHub (with GitHub Projects Kanban board for task tracking)
- **Collaboration:** Slack (coordination), Zoom (working sessions), shared Google Drive (documents)

---
## 🚀 Usage Instructions
___

### Activate Virtual Environment (Python)

Use Python 3.10+ to create and activate a virtual environment for this project.

```bash
# Create virtual environment (if not already created)
python -m venv .venv
# Activate virtual environment
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

---

### 📦 Install Dependencies (Python)

```bash
# Install dependencies from requirements.txt
pip install -r requirements.txt
```
---
### ⚡ Run Data Pipeline

```bash
# Run the main.py script to execute the entire data pipeline
python main.py
```

---

### 📁 Pipeline Outputs & Artifacts

| Directory / File |	Description |
| `data/processed/`	| Contains the cleaned, merged master datasets ready for analytical modeling.|
| `data/validation/` |	Houses CHR&R (County Health Rankings & Roadmaps) validation data and associated metadata files. |
| `data/final/` | Stores final outputs including HTML-formatted EDA and Modeling reports, train-test split datasets, and final model selection files. |
| `hypertension_atlas.db` | The core SQLite database housing structured project data for querying and downstream application use.|

---

### 🖥️ Dashboard Usage & Application

The project includes an interactive Streamlit dashboard designed to explore county-level health indicators, environmental factors, and machine learning predictions.

👉 Access the Live Dashboard Here: [<ads599capstonehypertensionriskatlas-3tepewveftselpfgfsn9nc.streamlit.app>](https://ads599capstonehypertensionriskatlas-b6cz6ygt7lui5xt4s4awuk.streamlit.app/)

(or launch it locally using the instructions below)

```bash
# Run the main.py script to execute the entire data pipeline
streamlit run dashboard/app.py
```
___

## 🎯 Scope & Intended Use

The Hypertension Risk Atlas is designed to **identify and prioritize** high-risk counties and inform the planning of targeted prevention activities. Consistent with the intended use of the underlying CDC PLACES small-area estimates, it is **not** intended to evaluate the effectiveness of specific programs or policies.

---

*Note on AI assistance: AI tools were used to support drafting and code scaffolding during this project. During the preparation of this work, the authors used AI-based large language models to assist with initial research, the synthesis of the literature review, and the development of data ingestion strategies. After using these tools, the authors validated and edited all output content, including technical terminology and research summaries, to ensure accuracy and take full responsibility for the final content of the publication.*
