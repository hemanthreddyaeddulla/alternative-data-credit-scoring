# -*- coding: utf-8 -*-
"""
Credit Risk Pipeline Package

This package provides a complete pipeline for credit risk modeling, including:
- Data preprocessing and feature engineering
- Model training with multiple algorithms
- Performance analysis including thin-file customer evaluation
- Visualization of results

Main Components:
    CreditRiskPipeline: Main orchestrator for the entire pipeline
    DataPreprocessor: Handles data loading, cleaning, and feature engineering
    SequentialModelTrainer: Trains and evaluates multiple ML models
    LinearRegressionClassifier: Custom sklearn-compatible classifier

Analysis & Visualization:
    compare_alternative_impact: Compares traditional vs alternative data impact
    create_thin_file_plots: Generates thin-file customer analysis plots
    create_comparison_plots: Creates model comparison visualizations
"""

from .credit_pipeline import CreditRiskPipeline
from .data_preprocessor import DataPreprocessor
from .trainer import SequentialModelTrainer
from .custom_models import LinearRegressionClassifier
from .analysis import compare_alternative_impact
from .visualize import create_thin_file_plots, create_comparison_plots

__all__ = [
    'CreditRiskPipeline',
    'DataPreprocessor',
    'SequentialModelTrainer',
    'LinearRegressionClassifier',
    'compare_alternative_impact',
    'create_thin_file_plots',
    'create_comparison_plots'
]
