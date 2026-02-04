# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.utils.paths import artifact_path

def compare_alternative_impact(all_results, datasets, trainer):
    """Compare how alternative data helps thin-file customers"""

    # Get validation data
    X_val_all = datasets['all']['X_val']
    y_val = datasets['all']['y_val']

    # Identify thin-file customers (simplified)
    feature_variance = np.var(X_val_all, axis=1)
    threshold = np.percentile(feature_variance, 20)
    thin_file_mask = feature_variance < threshold

    print(f"\n[INFO] Comparing Feature Sets for Thin-File Customers:")
    print("-" * 50)

    # Group results by model name
    model_groups = {}
    for result in all_results:
        model_name = result['model_name']
        if model_name not in model_groups:
            model_groups[model_name] = {}
        model_groups[model_name][result['feature_set']] = result

    # Compare each model's performance across feature sets
    comparison_results = []

    for model_name, feature_results in model_groups.items():
        if 'traditional' in feature_results and 'alternative' in feature_results:
            print(f"\n{model_name}:")

            # Traditional features
            trad_model = feature_results['traditional']['model']
            X_val_trad = datasets['traditional']['X_val']
            y_pred_trad = trad_model.predict_proba(X_val_trad[thin_file_mask])[:, 1]
            auc_trad = roc_auc_score(y_val[thin_file_mask], y_pred_trad)

            # Alternative features
            alt_model = feature_results['alternative']['model']
            X_val_alt = datasets['alternative']['X_val']
            y_pred_alt = alt_model.predict_proba(X_val_alt[thin_file_mask])[:, 1]
            auc_alt = roc_auc_score(y_val[thin_file_mask], y_pred_alt)

            # All features
            if 'all' in feature_results:
                all_model = feature_results['all']['model']
                X_val_all = datasets['all']['X_val']
                y_pred_all = all_model.predict_proba(X_val_all[thin_file_mask])[:, 1]
                auc_all = roc_auc_score(y_val[thin_file_mask], y_pred_all)
            else:
                auc_all = 0

            improvement = auc_alt - auc_trad
            print(f"  Traditional AUC: {auc_trad:.4f}")
            print(f"  Alternative AUC: {auc_alt:.4f}")
            print(f"  All features AUC: {auc_all:.4f}")
            print(f"  Alternative improvement: {improvement:+.4f} ({improvement/auc_trad*100:+.1f}%)")

            comparison_results.append({
                'model': model_name,
                'auc_traditional': auc_trad,
                'auc_alternative': auc_alt,
                'auc_all': auc_all,
                'improvement': improvement
            })

    if comparison_results:
        df_comp = pd.DataFrame(comparison_results)
        df_comp.to_csv('alternative_impact_thin_file.csv', index=False)
        print(f"\n[SAVED] Alternative data impact saved to 'alternative_impact_thin_file.csv'")

        # Summary
        avg_improvement = df_comp['improvement'].mean()
        print(f"\n[INFO] SUMMARY:")
        print(f"  Average improvement from alternative data: {avg_improvement:+.4f}")
        print(f"  Best improvement: {df_comp['improvement'].max():+.4f} ({df_comp.loc[df_comp['improvement'].idxmax(), 'model']})")
