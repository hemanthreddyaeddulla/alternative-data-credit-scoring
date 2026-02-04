# -*- coding: utf-8 -*-
"""
Custom Model Wrappers for Credit Risk Pipeline

This module provides custom sklearn-compatible model wrappers for credit risk
classification. These wrappers ensure consistent interfaces (fit, predict,
predict_proba) across all models used in the pipeline.

Classes:
    LinearRegressionClassifier: Wraps sklearn's LinearRegression for binary
                                classification tasks with automatic threshold
                                optimization.

Note:
    All classifiers in this module implement the sklearn BaseEstimator and
    ClassifierMixin interfaces for compatibility with sklearn pipelines,
    cross-validation, and grid search utilities.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression


class LinearRegressionClassifier(BaseEstimator, ClassifierMixin):
    """
    Linear Regression wrapper for binary classification.

    This classifier uses sklearn's LinearRegression model to predict continuous
    values, then converts them to class predictions using an optimized threshold.
    The threshold is automatically determined during training to maximize accuracy.

    This approach can be useful as a baseline model or when interpretability is
    important, as the relationship between features and the target remains linear.

    Attributes:
        model (LinearRegression): The underlying sklearn LinearRegression model.
        threshold (float): The decision threshold for converting continuous
                          predictions to binary classes. Optimized during fit().
        classes_ (np.ndarray): Array of class labels [0, 1] for sklearn compatibility.

    Example:
        >>> from custom_models import LinearRegressionClassifier
        >>> clf = LinearRegressionClassifier()
        >>> clf.fit(X_train, y_train)
        >>> predictions = clf.predict(X_test)
        >>> probabilities = clf.predict_proba(X_test)
    """

    def __init__(self):
        """
        Initialize the LinearRegressionClassifier.

        Creates a new LinearRegression model instance and sets default values
        for threshold (0.5) and class labels.
        """
        self.model = LinearRegression()
        self.threshold = 0.5
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        """
        Train the linear regression model and optimize the decision threshold.

        The method first fits the LinearRegression model to the training data,
        then searches for the optimal threshold that maximizes classification
        accuracy by testing 100 evenly-spaced thresholds between the minimum
        and maximum predicted values.

        Parameters:
            X (array-like): Training features of shape (n_samples, n_features).
            y (array-like): Target values of shape (n_samples,). Should contain
                           binary labels (0 or 1).

        Returns:
            self: Returns the fitted classifier instance.
        """
        # Fit the underlying linear regression model
        self.model.fit(X, y)

        # Find optimal threshold by testing multiple values
        y_pred_cont = self.model.predict(X)
        thresholds = np.linspace(y_pred_cont.min(), y_pred_cont.max(), 100)

        best_acc = 0
        for t in thresholds:
            # Calculate accuracy at this threshold
            acc = ((y_pred_cont > t).astype(int) == y).mean()
            if acc > best_acc:
                best_acc = acc
                self.threshold = t

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for input samples.

        The continuous predictions from the linear regression are clipped to
        the [0, 1] range to serve as pseudo-probabilities. These values are
        then formatted as a 2-column array for sklearn compatibility.

        Parameters:
            X (array-like): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Array of shape (n_samples, 2) where:
                       - Column 0: Probability of class 0 (1 - prediction)
                       - Column 1: Probability of class 1 (prediction)
        """
        y_pred = self.model.predict(X)
        # Clip to [0, 1] range to ensure valid probabilities
        y_pred = np.clip(y_pred, 0, 1)
        return np.column_stack([1 - y_pred, y_pred])

    def predict(self, X):
        """
        Predict binary class labels for input samples.

        Uses the optimized threshold from fit() to convert continuous
        predictions into binary class labels.

        Parameters:
            X (array-like): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Array of predicted class labels (0 or 1) with shape
                       (n_samples,).
        """
        y_pred = self.model.predict(X)
        return (y_pred > self.threshold).astype(int)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator (sklearn compatibility).

        Parameters:
            deep (bool): If True, will return the parameters for this estimator
                        and contained subobjects. Default is True.

        Returns:
            dict: Empty dictionary as this classifier has no hyperparameters.
        """
        return {}

    def set_params(self, **params):
        """
        Set parameters for this estimator (sklearn compatibility).

        Parameters:
            **params: Estimator parameters to set.

        Returns:
            self: Returns the estimator instance.
        """
        return self
