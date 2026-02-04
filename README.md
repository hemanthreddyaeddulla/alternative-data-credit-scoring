# Credit Risk Analysis with Alternative Data

A comprehensive machine learning pipeline for credit risk assessment that compares the predictive power of **traditional credit bureau data** versus **alternative data sources** for predicting loan defaults.

---

## Research Objective

> **Can alternative data sources (digital footprints, behavioral signals, external scores) improve credit risk prediction, especially for "thin-file" customers who lack extensive credit history?**

---

## Project Overview

Traditional credit scoring relies heavily on credit bureau data - payment history, outstanding debts, length of credit history, etc. However, this approach fails for **thin-file customers** who lack sufficient credit history:
- Young adults with no prior loans
- Immigrants new to the credit system
- People who prefer cash transactions

This project investigates whether **alternative data sources** can fill this gap and enable better credit decisions for underserved populations.

### What is Alternative Data?

| Type | Examples |
|------|----------|
| **External Scores** | Third-party credit scores, risk indicators |
| **Digital Footprint** | Email verification, phone verification, contact flags |
| **Document Verification** | Document submission patterns |
| **Behavioral Signals** | Application behavior, regional indicators |

---

## Key Results

### Best Model Performance

| Metric | Value |
|--------|-------|
| **Best Model** | LightGBM |
| **AUC Score** | **0.7742** |
| **Acceptance Rate @ 5% Bad Rate** | **84.0%** |

### Model Comparison (All Features)

| Rank | Model | AUC | Acceptance Rate |
|------|-------|-----|-----------------|
| 1 | **LightGBM** | **0.7742** | **84.0%** |
| 2 | Gradient Boosting | 0.7659 | 82.7% |
| 3 | Linear Regression | 0.7632 | 82.5% |
| 4 | Logistic Regression | 0.7620 | 82.7% |
| 5 | SVM | 0.7538 | 81.0% |
| 6 | Extra Trees | 0.7418 | 78.3% |
| 7 | Random Forest | 0.7403 | 77.8% |
| 8 | Decision Tree | 0.7034 | 71.9% |

### Feature Set Comparison

| Feature Set | # Features | Avg AUC | Avg Acceptance Rate |
|-------------|------------|---------|---------------------|
| **All (Combined)** | 381 | 0.7507 | 80.1% |
| **Alternative Only** | 47 | 0.7177 | 76.1% |
| **Traditional Only** | 334 | 0.7029 | 69.1% |

### Key Finding

**Alternative data alone (47 features) outperforms traditional data (334 features)!**

This validates the hypothesis that alternative data provides valuable predictive signals, achieving better results with **7× fewer features**.

---

## Thin-File Customer Analysis

A critical aspect of this research is evaluating model performance on thin-file customers.

| Model | Regular AUC | Thin-File AUC | Thin-File Acceptance |
|-------|-------------|---------------|---------------------|
| LightGBM | 0.7742 | 0.7689 | **89.4%** |
| Gradient Boosting | 0.7659 | 0.7561 | 88.0% |
| Logistic Regression | 0.7620 | 0.7548 | 87.5% |

**Key Insight:** Models maintain strong performance on thin-file customers, with acceptance rates actually **higher** than the general population (89.4% vs 84.0%).

---

## Business Impact

| Metric | Without Alternative Data | With Alternative Data |
|--------|-------------------------|----------------------|
| Best AUC | 0.7387 | **0.7742** |
| Acceptance Rate | 77.7% | **84.0%** |
| Thin-File Acceptance | 86.5% | **89.4%** |
| Bad Rate | 5% (controlled) | 5% (controlled) |

**Bottom Line:**
- **+6.3% more loan approvals** at the same risk level
- **+2.9% more thin-file customers** included
- Enables **financial inclusion** for underserved populations

---

## Pipeline Features

- **Data Preprocessing**: Automated merging, cleaning, and feature engineering
- **Yeo-Johnson Transformation**: Handles skewed distributions
- **SMOTE Balancing**: Addresses class imbalance (8% default rate)
- **Multiple Feature Sets**: Compare traditional vs alternative vs combined
- **8 ML Models**: From simple (Logistic Regression) to complex (LightGBM)
- **Thin-File Analysis**: Evaluate performance on credit-invisible customers

---

## Models Implemented

### Linear Models
- Linear Regression (with optimal threshold)
- Logistic Regression

### Tree-Based Models
- Decision Tree
- Random Forest
- Gradient Boosting
- LightGBM
- Extra Trees

### Other
- SVM (SGDClassifier)

---

## Installation

```bash
# Create conda environment
conda create -n credit_risk python=3.10 -y
conda activate credit_risk

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python run.py
```

Follow the interactive prompts to:
1. Choose whether to reprocess data
2. Select which models to train
3. View results and analysis

---

## Data Source

This project uses the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset from Kaggle.

> Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience.

**Note:** Data files are not included due to size (~2.6 GB). Download from Kaggle and place in the `data/` folder.

---

## Project Structure

```
├── run.py              # Main entry point
├── requirements.txt    # Python dependencies
├── src/
│   ├── pipeline/       # ML pipeline modules
│   └── utils/          # Utility functions
└── data/               # Data folder (download from Kaggle)
```

---

## Requirements

- Python 3.10+
- 16GB+ RAM (for processing large datasets)
- See `requirements.txt` for package dependencies

---

## Conclusions

1. **Alternative data is valuable** - 47 features outperform 334 traditional features
2. **Combining data is best** - All features yield 0.7742 AUC
3. **LightGBM wins** - Best balance of speed and accuracy
4. **Thin-file customers benefit most** - 89.4% acceptance rate
5. **Financial inclusion** - Alternative data enables lending to underserved populations

---

## Future Improvements

- Hyperparameter optimization
- Feature importance analysis (SHAP values)
- Ensemble methods
- Fairness analysis for protected groups
- Time-based validation

---

## License

MIT License

## Acknowledgments

- **Data Source:** [Home Credit Default Risk - Kaggle](https://www.kaggle.com/c/home-credit-default-risk)
- **Goal:** Advancing financial inclusion through alternative data
