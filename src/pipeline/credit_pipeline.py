# -*- coding: utf-8 -*-
import os, gc, pickle
import numpy as np
import pandas as pd
from typing import Optional
from .data_preprocessor import DataPreprocessor
from .trainer import SequentialModelTrainer
from .analysis import compare_alternative_impact
from .visualize import create_thin_file_plots, create_comparison_plots
from src.utils.paths import artifact_path

class CreditRiskPipeline:
    """
    Encapsulated credit risk modeling pipeline with preprocessing, model selection,
    training, analysis, and visualization.
    """

    def preprocess(self, train_paths, test_paths=None, reprocess_choice: Optional[str] = None):
        """Run preprocessing; reuse cache if available and user declines reprocess."""
        print("\n" + "="*60)
        print("STEP 1: DATA PREPROCESSING")
        print("="*60)
        if not os.path.exists('preprocessed_data.pkl'):
            preprocessor = DataPreprocessor()
            datasets = preprocessor.preprocess_and_save(train_paths, test_paths)
        else:
            print("\n[FILE] Loading existing preprocessed data...")
            with open('preprocessed_data.pkl', 'rb') as f:
                datasets = pickle.load(f)
            print("[OK] Data loaded successfully")

            if reprocess_choice is None:
                reprocess_choice = os.getenv('REPROCESS')
            if reprocess_choice is None:
                reprocess_choice = input("\nDo you want to reprocess the data? (y/n): ").lower()
            else:
                print(f"\nUsing non-interactive reprocess choice: {reprocess_choice}")
            if reprocess_choice == 'y':
                preprocessor = DataPreprocessor()
                datasets = preprocessor.preprocess_and_save(train_paths, test_paths)
        return datasets

    def select_models(self, selection: Optional[str] = None):
        """Return a dictionary of selected model instances by delegating to trainer."""
        trainer = SequentialModelTrainer()
        selected = trainer.select_models(selection)
        return trainer, selected

    def train_models(self, trainer, selected_models, datasets):
        """Train selected models and persist intermediate results."""
        print("\n" + "="*60)
        print(f"TRAINING {len(selected_models)} SELECTED MODELS")
        print("="*60)
        all_results = []
        for idx, (model_name, model) in enumerate(selected_models.items(), 1):
            print(f"\n[MODEL] MODEL {idx}/{len(selected_models)}: {model_name}")
            model_results = trainer.train_single_model(model_name, model, datasets)
            all_results.extend(model_results)
            trainer.save_results()
            print(f"\n[OK] {model_name} complete!")
            gc.collect()
        return all_results

    def analyze_thin_file(self, trainer, datasets, all_results):
        """Analyze thin-file performance using trainer's method."""
        print("\n" + "="*60)
        print("ANALYZING THIN-FILE/NO-CREDIT CUSTOMERS")
        print("="*60)
        return trainer.analyze_thin_file_customers(datasets, all_results)

    def compare_alternative_impact(self, all_results, datasets):
        """Compare alternative vs traditional features for thin-file customers."""
        return compare_alternative_impact(all_results, datasets, None)

    def create_thin_file_plots(self, thin_file_results):
        """Create and save thin-file analysis plots."""
        return create_thin_file_plots(thin_file_results)

    def create_comparison_plots(self, df_results):
        """Create and save overall comparison plots."""
        return create_comparison_plots(df_results)

    def run(self, train_paths, test_paths=None, selection: Optional[str] = None, reprocess_choice: Optional[str] = None):
        """
        Execute the complete sequential pipeline and return results DataFrame.

        Args:
            train_paths: Dict of training paths
            test_paths: Optional dict of test paths
            selection: Non-interactive model selection string (e.g., '99')
            reprocess_choice: 'y'/'n' to control reprocessing without prompt
        """
        print("\n" + "="*80)
        print("SEQUENTIAL CREDIT RISK PIPELINE WITH WINDOWIZING")
        print("="*80)

        datasets = self.preprocess(train_paths, test_paths, reprocess_choice=reprocess_choice)
        trainer, selected_models = self.select_models(selection=selection)
        all_results = self.train_models(trainer, selected_models, datasets)

        thin_file_results = self.analyze_thin_file(trainer, datasets, all_results)

        print("\n" + "="*60)
        print("ALTERNATIVE DATA IMPACT ON THIN-FILE CUSTOMERS")
        print("="*60)
        self.compare_alternative_impact(all_results, datasets)

        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        df_results = pd.DataFrame(all_results).sort_values('auc_score', ascending=False)

        print("\n[BEST] Top 10 Model Performances (Overall):")
        print(df_results[['model_name', 'feature_set', 'auc_score', 'acceptance_rate']].head(10).to_string())

        if not df_results.empty:
            best_overall = df_results.iloc[0]
            print(f"\n[STAR] BEST MODEL OVERALL: {best_overall['model_name']} with {best_overall['feature_set']} features")
            print(f"   AUC Score: {best_overall['auc_score']:.4f}")
            print(f"   Acceptance Rate: {best_overall['acceptance_rate']:.2%}")

        if thin_file_results:
            df_thin = pd.DataFrame(thin_file_results)
            best_thin = df_thin.loc[df_thin['auc_thin_file'].idxmax()]
            print(f"\n[STAR] BEST MODEL FOR THIN-FILE: {best_thin['model_name']}")
            print(f"   AUC Score (thin-file): {best_thin['auc_thin_file']:.4f}")
            print(f"   Acceptance Rate (thin-file): {best_thin['acceptance_thin_file']:.2%}")

        self.create_comparison_plots(df_results)
        if thin_file_results:
            self.create_thin_file_plots(thin_file_results)

        return df_results
