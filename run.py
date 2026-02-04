# -*- coding: utf-8 -*-
"""
Credit Risk Pipeline - Main Entry Point

This script runs the complete credit risk modeling pipeline:
1. Load and preprocess data from 7 CSV files (~2.6 GB)
2. Train 8 ML models on 3 feature sets (all, traditional, alternative)
3. Analyze thin-file customer performance
4. Generate comparison visualizations

Usage:
    python run.py                    # Interactive mode

    # Non-interactive mode with environment variables:
    set MODEL_SELECTION=0            # 0=all, 99=quick mode
    set REPROCESS=n                  # y=reprocess, n=use cache
    python run.py

Models (8 total):
    - Linear: Linear Regression, Logistic Regression
    - Tree-based: Decision Tree, Random Forest, Gradient Boosting, LightGBM, Extra Trees
    - Other: SVM
"""

import os
from src.pipeline.credit_pipeline import CreditRiskPipeline
from src.utils.paths import data_path


def main():
    """Run the credit risk pipeline."""

    # Get configuration from environment variables or use defaults
    model_selection = os.getenv('MODEL_SELECTION', None)
    reprocess_choice = os.getenv('REPROCESS', None)

    print("=" * 80)
    print("CREDIT RISK ANALYSIS WITH ALTERNATIVE DATA")
    print("=" * 80)
    print("\nThis pipeline compares Traditional vs Alternative data for credit scoring.")
    print(f"Models: 8 ML algorithms × 3 feature sets = 24 configurations\n")

    # Define data paths using data_path() function
    train_paths = {
        'application': data_path('application_train.csv'),
        'bureau': data_path('bureau.csv'),
        'bureau_balance': data_path('bureau_balance.csv'),
        'previous_application': data_path('previous_application.csv'),
        'credit_card_balance': data_path('credit_card_balance.csv'),
        'pos_cash_balance': data_path('POS_CASH_balance.csv'),
        'installments_payments': data_path('installments_payments.csv')
    }

    # Define test data paths (not used - no TARGET column)
    test_paths = {
        'application': data_path('application_test.csv'),
        'bureau': data_path('bureau.csv'),
        'bureau_balance': data_path('bureau_balance.csv'),
        'previous_application': data_path('previous_application.csv'),
        'credit_card_balance': data_path('credit_card_balance.csv'),
        'pos_cash_balance': data_path('POS_CASH_balance.csv'),
        'installments_payments': data_path('installments_payments.csv')
    }

    # Run pipeline
    try:
        pipeline = CreditRiskPipeline()
        results = pipeline.run(
            train_paths,
            test_paths,
            selection=model_selection,
            reprocess_choice=reprocess_choice
        )

        print("\n" + "=" * 80)
        print("[OK] PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nGenerated files:")
        print("  - artifact/01_Model_results.csv    : Model performance metrics")
        print("  - artifact/02_model_comparison.png : Performance visualization")
        print("  - artifact/03_thin_file_analysis.png : Thin-file analysis")
        print("  - models/*.pkl                     : Saved model files (24)")
        print("  - data/preprocessor.pkl            : Preprocessor object")

    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
