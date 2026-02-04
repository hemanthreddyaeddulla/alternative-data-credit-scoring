# Credit Risk Analysis with Alternative Data

A comprehensive credit risk modeling pipeline that compares the predictive power of **traditional credit bureau data** versus **alternative data sources** for predicting loan defaults.

---

## Research Objective

> **Can alternative data sources improve credit risk prediction, especially for "thin-file" customers who lack extensive credit history?**

---

## Key Findings

| Metric | Best Model | Feature Set | Score |
|--------|------------|-------------|-------|
| **Overall AUC** | LightGBM | All (381 features) | **0.7742** |
| **Acceptance Rate @ 5% BR** | LightGBM | All | **84.0%** |
| **Alternative Only** | Random Forest | Alternative (47 features) | 0.7290 |
| **Traditional Only** | LightGBM | Traditional (334 features) | 0.7387 |

### Key Insight

**Alternative data alone (47 features) achieves higher average AUC than traditional data (334 features)**, validating that alternative data provides valuable signals for credit risk assessment, especially for thin-file customers.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Dataset](#dataset)
3. [Pipeline Overview](#pipeline-overview)
4. [Models](#models)
5. [Results](#results)
6. [Installation & Usage](#installation--usage)
7. [Documentation](#documentation)

---

## Project Structure

```
Credit-Risk-Alternative-Data/
|
+-- run.py                    # Main entry point
+-- requirements.txt          # Python dependencies
+-- README.md                 # This file
|
+-- data/
|   +-- data_added_now/       # Raw CSV files (~2.6 GB)
|   |   +-- application_train.csv    # 307K rows - Main training data
|   |   +-- bureau.csv               # 1.7M rows - External credit history
|   |   +-- bureau_balance.csv       # 27M rows - Monthly bureau snapshots
|   |   +-- previous_application.csv # 1.67M rows - Prior HC applications
|   |   +-- credit_card_balance.csv  # 3.8M rows - Credit card data
|   |   +-- POS_CASH_balance.csv     # 10M rows - POS/cash loans
|   |   +-- installments_payments.csv # 13.6M rows - Payment history
|   +-- preprocessor.pkl
|   +-- preprocessed_data_sample_1pct/
|
+-- src/
|   +-- __init__.py
|   +-- utils/
|   |   +-- __init__.py
|   |   +-- paths.py              # Path utilities
|   +-- pipeline/
|       +-- __init__.py
|       +-- credit_pipeline.py    # Main orchestrator
|       +-- data_preprocessor.py  # Data processing
|       +-- trainer.py            # Model training
|       +-- custom_models.py      # Custom classifiers
|       +-- analysis.py           # Thin-file analysis
|       +-- visualize.py          # Plot generation
|
+-- models/                   # Saved model files (24 .pkl files)
+-- artifact/                 # Output files
|   +-- 01_Model_results.csv
|   +-- 02_model_comparison.png
|   +-- 03_thin_file_analysis.png
|   +-- EDA_output/           # EDA visualizations
|
+-- notebooks/                # Documentation notebooks
|   +-- 00_Project_Overview.ipynb
|   +-- 01_Data_Documentation.ipynb
```

---

## Dataset

### Data Source

**Home Credit Default Risk** dataset - consumer finance data for populations with little or no credit history.

### Data Schema

```
application_train.csv (SK_ID_CURR) - 307,511 rows
    |
    +-> bureau.csv (1.7M records) - External credit bureau history
    |       |
    |       +-> bureau_balance.csv (27.3M records) - Monthly bureau snapshots
    |
    +-> previous_application.csv (1.67M records) - Prior Home Credit applications
            |
            +-> credit_card_balance.csv (3.8M records) - Credit card data
            +-> POS_CASH_balance.csv (10M records) - POS/cash loan balances
            +-> installments_payments.csv (13.6M records) - Payment history
```

### Data Files

| File | Rows | Size | Description |
|------|------|------|-------------|
| `application_train.csv` | 307,511 | 158 MB | Main training data with TARGET |
| `bureau.csv` | 1.7M | 162 MB | External credit bureau history |
| `bureau_balance.csv` | 27.3M | 358 MB | Monthly bureau balance snapshots |
| `previous_application.csv` | 1.67M | 386 MB | Prior Home Credit applications |
| `credit_card_balance.csv` | 3.8M | 405 MB | Credit card monthly data |
| `POS_CASH_balance.csv` | 10M | 375 MB | POS/cash loan balances |
| `installments_payments.csv` | 13.6M | 690 MB | Payment history |

**Total:** ~58 million records, ~2.6 GB

---

## Pipeline Overview

### High-Level Architecture

```
Raw CSV Files (2.6 GB)
        |
        v
+-------------------+
| Data Preprocessor |  <- Merge, Engineer, Transform, Encode, Split, Balance
+-------------------+
        |
        v
+-------------------+
| Feature Sets      |  <- All (381), Traditional (334), Alternative (47)
+-------------------+
        |
        v
+-------------------+
| Model Trainer     |  <- Train 8 models x 3 feature sets = 24 configurations
+-------------------+
        |
        v
+-------------------+
| Analysis          |  <- Thin-file analysis, Feature set comparison
+-------------------+
        |
        v
+-------------------+
| Visualization     |  <- Generate comparison plots
+-------------------+
```

### Preprocessing Steps

1. **Load & Merge**: Combine 7 CSV files by SK_ID_CURR
2. **Feature Engineering**: Create ratios, age, employment features
3. **Windowizing**: Yeo-Johnson power transformation for skewed features
4. **Categorical Encoding**: LabelEncoder (binary) or Top-5 One-Hot (multi-category)
5. **Train/Validation Split**: 80/20 stratified
6. **SMOTE Balancing**: 50% sampling ratio for class imbalance
7. **StandardScaler**: Normalize features

### Train/Validation Split

```
application_train.csv (307,511 rows with TARGET)
         |
         +--- 80% ---> Training Set (~246,000 rows)
         |                    |
         |                    +---> SMOTE Applied (50% ratio)
         |                    +---> ~368,000 rows after balancing
         |
         +--- 20% ---> Validation Set (~61,500 rows)
                              +---> NO SMOTE (evaluate on real distribution)
```

**Note:** `application_test.csv` is NOT used - it has no TARGET column.

---

## Models

### Implemented Models (8 total)

| Category | Model | Description |
|----------|-------|-------------|
| Linear | Linear Regression | Wrapper as classifier |
| Linear | Logistic Regression | Standard binary classifier |
| Tree | Decision Tree | Single tree (max_depth=10) |
| Tree | Random Forest | 100 trees (max_depth=10) |
| Tree | Gradient Boosting | 100 estimators |
| Tree | LightGBM | Fast gradient boosting |
| Tree | Extra Trees | Extremely randomized trees |
| Other | SVM | SGDClassifier with log_loss |

### Feature Sets

| Feature Set | Count | Description |
|-------------|-------|-------------|
| **All** | 381 | Combined traditional + alternative |
| **Traditional** | 334 | Standard credit bureau data |
| **Alternative** | 47 | Non-traditional data sources |

**Total Configurations:** 8 models x 3 feature sets = **24 trained models**

---

## Results

### Overall Model Performance (All Features)

| Rank | Model | AUC | Acceptance Rate |
|------|-------|-----|-----------------|
| 1 | **LightGBM** | **0.7742** | **84.0%** |
| 2 | Gradient Boosting | 0.7659 | 82.7% |
| 3 | Linear Regression | 0.7632 | 82.5% |
| 4 | Logistic Regression | 0.7620 | 82.6% |
| 5 | SVM | 0.7538 | 81.0% |
| 6 | Extra Trees | 0.7418 | 78.5% |
| 7 | Random Forest | 0.7403 | 77.8% |
| 8 | Decision Tree | 0.7034 | 71.9% |

### Feature Set Comparison

| Feature Set | Features | Avg AUC | Avg Acceptance |
|-------------|----------|---------|----------------|
| **All** | 381 | 0.7387 | 78.0% |
| **Alternative** | 47 | 0.7177 | 73.4% |
| **Traditional** | 334 | 0.6985 | 69.9% |

### Business Impact

| Metric | Without Alt. Data | With Alt. Data |
|--------|-------------------|----------------|
| Acceptance Rate | 70% | 84% |
| Bad Rate | 5% | 5% |
| Impact | Baseline | **+14% more approvals** |

---

## Installation & Usage

### Prerequisites

- Python 3.8+
- 16GB+ RAM (for processing 2.6GB of data)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-repo/Credit-Risk-Alternative-Data.git
cd Credit-Risk-Alternative-Data

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# Run with all models
python run.py

# Or with environment variables for non-interactive mode
set MODEL_SELECTION=0     # All models
set REPROCESS=n           # Use cached preprocessed data
python run.py
```

### Model Selection Options

| Option | Description |
|--------|-------------|
| `0` | All 8 models |
| `99` | Quick mode (LightGBM, Random Forest, Logistic Regression) |
| `1,3,5` | Custom selection by number |

### Expected Runtime

| Component | Time |
|-----------|------|
| Data Preprocessing | 10-15 minutes |
| All Models Training | 20-30 minutes |
| Quick Mode | 5-10 minutes |

---

## Documentation

Detailed documentation is available in the `notebooks/` folder:

| Notebook | Description |
|----------|-------------|
| `00_Project_Overview.ipynb` | Complete project documentation with workflow |
| `01_Data_Documentation.ipynb` | Detailed data file descriptions |

### Output Files

| File | Description |
|------|-------------|
| `artifact/01_Model_results.csv` | All model performance metrics |
| `artifact/02_model_comparison.png` | 4-panel visualization |
| `artifact/03_thin_file_analysis.png` | Thin-file vs regular comparison |
| `artifact/EDA_output/` | Exploratory analysis visualizations |

---

## Future Work

- Hyperparameter tuning with grid search
- Feature selection to reduce dimensionality
- Ensemble methods combining top models
- Fairness analysis for protected groups
- Time-based validation for realistic evaluation

---

## License

This project is for educational and research purposes.

## Acknowledgments

- Data: Home Credit Default Risk (Kaggle)
- Goal: Advancing financial inclusion through alternative data
