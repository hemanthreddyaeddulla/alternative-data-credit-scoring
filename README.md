# Credit Risk Analysis with Alternative Data

A machine learning pipeline for credit risk assessment comparing traditional credit bureau features with alternative data sources.

## Overview

This project explores whether **alternative data sources** can improve credit risk prediction, particularly for "thin-file" customers with limited credit history.

### Key Findings

- Alternative data provides valuable predictive signals for credit risk
- Combining traditional and alternative features yields the best results
- Thin-file customers benefit significantly from alternative data inclusion

## Features

- Automated data preprocessing pipeline
- Multiple ML models (Linear, Tree-based, Ensemble)
- Comparison across different feature sets
- Thin-file customer analysis
- SMOTE for handling class imbalance

## Models

| Category | Models |
|----------|--------|
| Linear | Logistic Regression, Linear Regression |
| Tree-based | Decision Tree, Random Forest, Gradient Boosting, LightGBM, Extra Trees |
| Other | SVM |

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

Follow the interactive prompts to select models and run the pipeline.

## Data

This project uses the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset from Kaggle.

**Note:** Data files are not included in this repository due to size. Download from Kaggle and place in `data/` folder.

## Project Structure

```
├── run.py              # Entry point
├── requirements.txt    # Dependencies
├── src/
│   ├── pipeline/       # ML pipeline modules
│   └── utils/          # Utilities
└── data/               # Data folder (not included)
```

## Requirements

- Python 3.10+
- 16GB+ RAM recommended
- See `requirements.txt` for packages

## License

MIT License

## Acknowledgments

- Data source: [Home Credit Default Risk - Kaggle](https://www.kaggle.com/c/home-credit-default-risk)
