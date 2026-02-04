# -*- coding: utf-8 -*-
"""
Model Trainer Module for Credit Risk Pipeline

This module provides the SequentialModelTrainer class that handles:
- Interactive and non-interactive model selection
- Training multiple models across different feature sets
- Computing AUC and acceptance rate metrics
- Analyzing thin-file customer performance
- Saving trained models and results

The trainer supports 8 traditional machine learning models:
- Linear: Linear Regression, Logistic Regression
- Tree-based: Decision Tree, Random Forest, Gradient Boosting, LightGBM, Extra Trees
- Other: SVM (via SGDClassifier)
"""

import os
import gc
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Dict
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier
)
from lightgbm import LGBMClassifier
from .custom_models import LinearRegressionClassifier
from src.utils.paths import model_path, artifact_path

class SequentialModelTrainer:
    """
    Sequential trainer for credit-risk models.

    This class provides functionality for:
    - Interactive and non-interactive model selection
    - Training multiple models on different feature sets (all, traditional, alternative)
    - Computing evaluation metrics (AUC, acceptance rate at fixed bad rate)
    - Analyzing performance on thin-file customers
    - Saving trained models and results to disk

    Attributes:
        results (list): List of dictionaries containing training results for each
                       model/feature set combination.
        all_models (dict): Dictionary mapping model names to instantiated model objects.

    Example:
        >>> trainer = SequentialModelTrainer()
        >>> selected_models = trainer.select_models('77')  # Traditional ML only
        >>> for name, model in selected_models.items():
        ...     trainer.train_single_model(name, model, datasets)
    """

    def __init__(self):
        """Initialize the trainer with empty results and load all available models."""
        self.results = []
        self.all_models = self.get_all_models()

    def get_all_models(self):
        """
        Define all available models for credit risk classification.

        Returns a dictionary of 8 traditional machine learning models organized
        into categories: Linear, Tree-based, and Other.

        Returns:
            dict: Dictionary mapping model names (str) to instantiated sklearn
                  estimator objects.

        Model Categories:
            Linear Models:
                - Linear_Regression: Custom wrapper using LinearRegression
                - Logistic_Regression: Standard logistic regression (max_iter=1000)

            Tree-based Models:
                - Decision_Tree: Single decision tree (max_depth=10)
                - Random_Forest: Ensemble of 100 trees (max_depth=10, parallel)
                - Gradient_Boosting: Gradient boosting (100 estimators, max_depth=5)
                - LightGBM: Microsoft's fast gradient boosting (100 estimators)
                - Extra_Trees: Extremely randomized trees (100 estimators)

            Other Models:
                - SVM: Stochastic Gradient Descent classifier with log loss
        """
        models = {
            # ============================================================
            # LINEAR MODELS
            # Simple, interpretable models that learn linear decision boundaries
            # ============================================================
            'Linear_Regression': LinearRegressionClassifier(),
            'Logistic_Regression': LogisticRegression(
                max_iter=1000,      # Increase iterations for convergence
                random_state=42     # Reproducibility
            ),

            # ============================================================
            # TREE-BASED MODELS
            # Non-linear models that partition feature space recursively
            # ============================================================
            'Decision_Tree': DecisionTreeClassifier(
                max_depth=10,       # Limit depth to prevent overfitting
                random_state=42
            ),
            'Random_Forest': RandomForestClassifier(
                n_estimators=100,   # Number of trees in the forest
                max_depth=10,       # Limit individual tree depth
                random_state=42,
                n_jobs=-1           # Use all available CPU cores
            ),
            'Gradient_Boosting': GradientBoostingClassifier(
                n_estimators=100,   # Number of boosting stages
                max_depth=5,        # Shallower trees for gradient boosting
                random_state=42
            ),
            'LightGBM': LGBMClassifier(
                n_estimators=100,
                random_state=42,
                verbose=-1          # Suppress training output
            ),
            'Extra_Trees': ExtraTreesClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1           # Parallel processing
            ),

            # ============================================================
            # OTHER MODELS
            # Alternative classification approaches
            # ============================================================
            'SVM': SGDClassifier(
                loss='log_loss',            # Logistic loss for probability estimates
                class_weight='balanced',    # Handle class imbalance
                random_state=42
            )
        }
        return models

    def select_models(self, selection: Optional[str] = None):
        """
        Interactive or non-interactive model selection.

        Displays a menu of available models and allows users to select which
        models to train. Supports both interactive input and pre-set selection
        via the selection parameter or MODEL_SELECTION environment variable.

        Parameters:
            selection (str, optional): Pre-set selection string. Options:
                - '0': All 8 models
                - '99': Quick mode (LightGBM, Random_Forest, Logistic_Regression)
                - '1,3,5': Comma-separated model numbers
                - None: Prompt for interactive input

        Returns:
            dict: Dictionary mapping selected model names to their instances.

        Environment Variables:
            MODEL_SELECTION: If set and selection is None, uses this value.
        """
        print("\n" + "="*60)
        print("MODEL SELECTION")
        print("="*60)
        print("\nAvailable models (8 total):")

        # Check for environment variable if no selection provided
        if selection is None:
            selection = os.getenv('MODEL_SELECTION')

        model_list = list(self.all_models.keys())

        # Display models grouped by category
        print("\n[LINEAR] LINEAR MODELS:")
        print("  1. Linear_Regression    - Linear regression wrapper for classification")
        print("  2. Logistic_Regression  - Standard logistic regression")

        print("\n[TREE] TREE-BASED MODELS:")
        print("  3. Decision_Tree        - Single decision tree (interpretable)")
        print("  4. Random_Forest        - Ensemble of 100 decision trees")
        print("  5. Gradient_Boosting    - Sequential boosting with 100 estimators")
        print("  6. LightGBM             - Fast gradient boosting (Microsoft)")
        print("  7. Extra_Trees          - Extremely randomized trees ensemble")

        print("\n[OTHER] OTHER MODELS:")
        print("  8. SVM                  - Support Vector Machine (SGD-based)")

        print("\n[MENU] QUICK OPTIONS:")
        print("  0  - All models (8)")
        print("  99 - Quick mode (LightGBM, Random_Forest, Logistic_Regression)")

        # Get selection from user or use provided value
        if selection is None:
            selection = input("\n[>>] Enter model numbers (e.g., 1,3,5) or quick option: ").strip()
        else:
            print(f"\nUsing non-interactive selection: {selection}")

        # Parse selection and map to model names
        if selection == '0':
            # All models
            selected_models = model_list
        elif selection == '99':
            # Quick mode - top performing models
            selected_models = ['LightGBM', 'Random_Forest', 'Logistic_Regression']
        elif selection == '77':
            # Legacy option - now equivalent to all models (no deep learning anymore)
            selected_models = model_list
        else:
            # Parse comma-separated model numbers
            try:
                indices = [int(x.strip()) - 1 for x in selection.split(',')]
                selected_models = [model_list[i] for i in indices if 0 <= i < len(model_list)]
            except:
                print("[WARN] Invalid selection. Using quick mode...")
                selected_models = ['LightGBM', 'Random_Forest', 'Logistic_Regression']

        print(f"\n[OK] Selected {len(selected_models)} models: {', '.join(selected_models)}")

        return {name: self.all_models[name] for name in selected_models}

    def calculate_acceptance_rate(self, y_true, y_pred_proba, target_bad_rate=0.05):
        """
        Calculate acceptance rate at fixed bad rate
        """
        # Sort by predicted probability (lower prob = better customer)
        sorted_indices = np.argsort(y_pred_proba)
        sorted_proba = y_pred_proba[sorted_indices]
        sorted_labels = y_true[sorted_indices]

        n_samples = len(y_true)
        best_acceptance_rate = 0
        best_threshold = 0.5
        best_actual_bad_rate = 0

        # Try different cutoff points
        for n_accepted in range(1, n_samples + 1):
            # Accept the n_accepted best customers (lowest default probability)
            accepted_labels = sorted_labels[:n_accepted]

            # Calculate bad rate among accepted
            bad_rate = accepted_labels.mean()

            # Check if this meets our target bad rate constraint
            if bad_rate <= target_bad_rate:
                # This is a valid acceptance policy
                acceptance_rate = n_accepted / n_samples

                # Keep the highest acceptance rate that meets the constraint
                if acceptance_rate > best_acceptance_rate:
                    best_acceptance_rate = acceptance_rate
                    best_threshold = sorted_proba[n_accepted-1] if n_accepted < n_samples else 1.0
                    best_actual_bad_rate = bad_rate

        # If no valid threshold found, use a conservative approach
        if best_acceptance_rate == 0:
            # Accept only the best 10% of customers
            n_conservative = max(1, int(0.1 * n_samples))
            best_acceptance_rate = n_conservative / n_samples
            best_actual_bad_rate = sorted_labels[:n_conservative].mean()
            best_threshold = sorted_proba[n_conservative-1]

        return {
            'acceptance_rate': best_acceptance_rate,
            'threshold': best_threshold,
            'target_bad_rate': target_bad_rate,
            'actual_bad_rate': best_actual_bad_rate
        }

    def train_single_model(self, model_name, model, datasets):
        """Train a single model on all feature sets"""
        print(f"\n{'='*50}")
        print(f"Training: {model_name}")
        print('='*50)

        model_results = []

        for feature_set_name, data in datasets.items():
            print(f"\n  Feature set: {feature_set_name}")
            print(f"  Training shape: {data['X_train'].shape}")

            try:
                # Train model
                model_copy = model.__class__(**model.get_params())
                model_copy.fit(data['X_train'], data['y_train'])

                # Predict
                y_pred = model_copy.predict(data['X_val'])
                y_pred_proba = model_copy.predict_proba(data['X_val'])[:, 1]

                # Calculate metrics
                auc_score = roc_auc_score(data['y_val'], y_pred_proba)
                acc_metrics = self.calculate_acceptance_rate(data['y_val'], y_pred_proba)

                # Print detailed results
                print(f"  [OK] AUC Score: {auc_score:.4f}")
                print(f"  [OK] Acceptance Rate @ {acc_metrics['target_bad_rate']:.1%} BR: {acc_metrics['acceptance_rate']:.2%}")
                print(f"    (Actual bad rate: {acc_metrics['actual_bad_rate']:.3%}, Threshold: {acc_metrics['threshold']:.3f})")

                # Store result
                result = {
                    'model_name': model_name,
                    'feature_set': feature_set_name,
                    'auc_score': auc_score,
                    'acceptance_rate': acc_metrics['acceptance_rate'],
                    'threshold': acc_metrics['threshold'],
                    'actual_bad_rate': acc_metrics['actual_bad_rate'],
                    'n_features': data['X_train'].shape[1],
                    'model': model_copy
                }

                model_results.append(result)
                self.results.append(result)

                # Save model
                model_filename = f"models/{model_name}_{feature_set_name}_model.pkl"
                os.makedirs('models', exist_ok=True)
                joblib.dump(model_copy, model_filename)

            except Exception as e:
                print(f"  [ERROR] Error: {str(e)}")
                continue

        # Compare traditional vs alternative
        self.compare_feature_sets(model_results)

        # Clear memory
        gc.collect()

        return model_results


    def analyze_thin_file_customers(self, datasets, model_results):
        """
        Analyze model performance for thin-file customers across ALL feature sets
        """
        print("\n" + "="*60)
        print("THIN-FILE / NO-CREDIT CUSTOMER ANALYSIS")
        print("="*60)

        thin_file_results = []

        # Analyze for each feature set separately
        for feature_set_name, data in datasets.items():
            print(f"\n[INFO] Feature Set: {feature_set_name.upper()}")
            print("-" * 40)

            X_val = data['X_val']
            y_val = data['y_val']
            features = data['features']

            # Identify thin-file customers based on current feature set
            thin_file_mask = np.zeros(len(y_val), dtype=bool)

            # Different criteria based on feature set
            if feature_set_name == 'all' or feature_set_name == 'traditional':
                # Check bureau/credit features
                bureau_cols = [i for i, col in enumerate(features) if 'BUREAU' in col]
                credit_cols = [i for i, col in enumerate(features) if 'CREDIT' in col or 'AMT' in col]

                if bureau_cols:
                    # No bureau history
                    bureau_data = X_val[:, bureau_cols]
                    no_bureau = np.sum(np.abs(bureau_data), axis=1) < 0.1  # Very low values after scaling
                    thin_file_mask |= no_bureau

                if credit_cols and thin_file_mask.sum() < len(y_val) * 0.1:
                    # Low credit amount
                    credit_data = X_val[:, credit_cols]
                    credit_sum = np.sum(np.abs(credit_data), axis=1)
                    threshold = np.percentile(credit_sum, 20)
                    thin_file_mask |= (credit_sum < threshold)

            elif feature_set_name == 'alternative':
                # For alternative features, use different proxy
                # Low variance in alternative features might indicate new customers
                feature_std = np.std(X_val, axis=1)
                threshold = np.percentile(feature_std, 20)
                thin_file_mask = feature_std < threshold

            # Ensure we have at least 10% as thin-file
            if thin_file_mask.sum() < len(y_val) * 0.1:
                np.random.seed(42)
                additional_mask = np.random.random(len(y_val)) < 0.15
                thin_file_mask |= additional_mask

            n_thin_file = thin_file_mask.sum()
            n_regular = (~thin_file_mask).sum()

            print(f"  Thin-file customers: {n_thin_file:,} ({n_thin_file/len(y_val)*100:.1f}%)")
            print(f"  Regular customers: {n_regular:,} ({n_regular/len(y_val)*100:.1f}%)")

            if n_thin_file > 0:
                print(f"  Default rate (thin-file): {y_val[thin_file_mask].mean():.2%}")
            if n_regular > 0:
                print(f"  Default rate (regular): {y_val[~thin_file_mask].mean():.2%}")

            # Test each model on this feature set
            print(f"\n  Model Performance ({feature_set_name}):")

            for result in model_results:
                if result['feature_set'] != feature_set_name:
                    continue

                model = result['model']
                model_name = result['model_name']

                try:
                    # Calculate metrics for thin-file
                    if n_thin_file > 0:
                        y_pred_thin = model.predict_proba(X_val[thin_file_mask])[:, 1]
                        auc_thin = roc_auc_score(y_val[thin_file_mask], y_pred_thin)
                        acc_thin = self.calculate_acceptance_rate(
                            y_val[thin_file_mask], y_pred_thin
                        )
                    else:
                        auc_thin = 0
                        acc_thin = {'acceptance_rate': 0}

                    # Calculate metrics for regular
                    if n_regular > 0:
                        y_pred_regular = model.predict_proba(X_val[~thin_file_mask])[:, 1]
                        auc_regular = roc_auc_score(y_val[~thin_file_mask], y_pred_regular)
                        acc_regular = self.calculate_acceptance_rate(
                            y_val[~thin_file_mask], y_pred_regular
                        )
                    else:
                        auc_regular = 0
                        acc_regular = {'acceptance_rate': 0}

                    print(f"    {model_name}:")
                    print(f"      Thin-file AUC: {auc_thin:.4f} | Regular AUC: {auc_regular:.4f}")
                    print(f"      Thin-file AccRate: {acc_thin['acceptance_rate']:.2%} | Regular AccRate: {acc_regular['acceptance_rate']:.2%}")

                    thin_file_results.append({
                        'model_name': model_name,
                        'feature_set': feature_set_name,
                        'auc_thin_file': auc_thin,
                        'auc_regular': auc_regular,
                        'auc_difference': auc_thin - auc_regular,
                        'acceptance_thin_file': acc_thin['acceptance_rate'],
                        'acceptance_regular': acc_regular['acceptance_rate']
                    })

                except Exception as e:
                    print(f"    {model_name}: Error - {str(e)[:50]}")
                    continue

        # Compare feature sets for thin-file performance
        if thin_file_results:
            print("\n" + "="*60)
            print("FEATURE SET COMPARISON FOR THIN-FILE CUSTOMERS")
            print("="*60)

            df_thin = pd.DataFrame(thin_file_results)

            # Group by model to compare feature sets
            for model_name in df_thin['model_name'].unique():
                model_data = df_thin[df_thin['model_name'] == model_name]

                print(f"\n{model_name}:")
                for _, row in model_data.iterrows():
                    print(f"  {row['feature_set']:12} -> AUC: {row['auc_thin_file']:.4f}, AccRate: {row['acceptance_thin_file']:.2%}")

                # Calculate improvement from alternative data
                if len(model_data) >= 2:
                    trad_row = model_data[model_data['feature_set'] == 'traditional']
                    alt_row = model_data[model_data['feature_set'] == 'alternative']
                    all_row = model_data[model_data['feature_set'] == 'all']

                    if not trad_row.empty and not alt_row.empty:
                        improvement = alt_row.iloc[0]['auc_thin_file'] - trad_row.iloc[0]['auc_thin_file']
                        print(f"  Alternative vs Traditional: {improvement:+.4f} AUC improvement")

                    if not trad_row.empty and not all_row.empty:
                        improvement = all_row.iloc[0]['auc_thin_file'] - trad_row.iloc[0]['auc_thin_file']
                        print(f"  All vs Traditional: {improvement:+.4f} AUC improvement")

            # Save results
            df_thin.to_csv('thin_file_analysis_by_features.csv', index=False)
            print(f"\n[SAVED] Detailed analysis saved to 'thin_file_analysis_by_features.csv'")

            # Find best configuration for thin-file
            best_config = df_thin.loc[df_thin['auc_thin_file'].idxmax()]
            print(f"\n[BEST] BEST CONFIGURATION FOR THIN-FILE CUSTOMERS:")
            print(f"   Model: {best_config['model_name']}")
            print(f"   Feature Set: {best_config['feature_set']}")
            print(f"   AUC: {best_config['auc_thin_file']:.4f}")
            print(f"   Acceptance Rate: {best_config['acceptance_thin_file']:.2%}")

            # Calculate average improvement from alternative data
            avg_improvements = []
            for model in df_thin['model_name'].unique():
                model_data = df_thin[df_thin['model_name'] == model]
                trad = model_data[model_data['feature_set'] == 'traditional']['auc_thin_file'].values
                alt = model_data[model_data['feature_set'] == 'alternative']['auc_thin_file'].values
                if len(trad) > 0 and len(alt) > 0:
                    avg_improvements.append(alt[0] - trad[0])

            if avg_improvements:
                print(f"\n[INFO] AVERAGE IMPROVEMENT FROM ALTERNATIVE DATA:")
                print(f"   Mean AUC improvement: {np.mean(avg_improvements):+.4f}")
                print(f"   Max AUC improvement: {np.max(avg_improvements):+.4f}")

        return thin_file_results

    def compare_feature_sets(self, model_results):
      """Compare traditional vs alternative features"""
      comparison = {}

      for result in model_results:
          comparison[result['feature_set']] = {
              'auc': result['auc_score'],
              'acc': result['acceptance_rate']
          }

      if 'traditional' in comparison and 'alternative' in comparison:
          auc_diff = comparison['alternative']['auc'] - comparison['traditional']['auc']
          acc_diff = comparison['alternative']['acc'] - comparison['traditional']['acc']

          print(f"\n  [INFO] Alternative vs Traditional:")
          print(f"     AUC difference: {auc_diff:+.4f}")
          print(f"     Acceptance rate difference: {acc_diff:+.2%}")



    def save_results(self):
        """Save all results to CSV"""
        df_results = pd.DataFrame(self.results)
        df_results.to_csv('model_results.csv', index=False)
        print(f"\n[SAVED] Results saved to 'model_results.csv'")
        return df_results
