# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
from src.utils.paths import fig_path

def create_thin_file_plots(thin_file_results):
    """Create visualizations for thin-file analysis"""
    df = pd.DataFrame(thin_file_results)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: AUC comparison
    ax1 = axes[0]
    x = range(len(df))
    width = 0.35

    bars1 = ax1.bar([i - width/2 for i in x], df['auc_thin_file'], width, label='Thin-file', color='orange')
    bars2 = ax1.bar([i + width/2 for i in x], df['auc_regular'], width, label='Regular', color='blue')

    ax1.set_xlabel('Model')
    ax1.set_ylabel('AUC Score')
    ax1.set_title('Model Performance: Thin-file vs Regular Customers')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['model_name'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Acceptance rate comparison
    ax2 = axes[1]
    bars3 = ax2.bar([i - width/2 for i in x], df['acceptance_thin_file']*100, width, label='Thin-file', color='orange')
    bars4 = ax2.bar([i + width/2 for i in x], df['acceptance_regular']*100, width, label='Regular', color='blue')

    ax2.set_xlabel('Model')
    ax2.set_ylabel('Acceptance Rate (%)')
    ax2.set_title('Acceptance Rate @ 5% Bad Rate')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['model_name'], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('thin_file_analysis.png', dpi=100, bbox_inches='tight')
    plt.show()
    print("\n[INFO] Thin-file visualization saved to 'thin_file_analysis.png'")

def create_comparison_plots(df_results):
    """Create comparison visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: AUC by Model and Feature Set
    ax1 = axes[0, 0]
    pivot_auc = df_results.pivot_table(index='model_name', columns='feature_set',
                                       values='auc_score', aggfunc='mean')
    pivot_auc.plot(kind='bar', ax=ax1)
    ax1.set_title('AUC Score Comparison')
    ax1.set_ylabel('AUC Score')
    ax1.set_xlabel('Model')
    ax1.legend(title='Feature Set')
    ax1.grid(axis='y', alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 2: Acceptance Rate
    ax2 = axes[0, 1]
    pivot_acc = df_results.pivot_table(index='model_name', columns='feature_set',
                                       values='acceptance_rate', aggfunc='mean')
    pivot_acc.plot(kind='bar', ax=ax2)
    ax2.set_title('Acceptance Rate @ 5% Bad Rate')
    ax2.set_ylabel('Acceptance Rate')
    ax2.set_xlabel('Model')
    ax2.legend(title='Feature Set')
    ax2.grid(axis='y', alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 3: Feature Set Performance
    ax3 = axes[1, 0]
    feature_avg = df_results.groupby('feature_set')[['auc_score', 'acceptance_rate']].mean()
    feature_avg.plot(kind='bar', ax=ax3)
    ax3.set_title('Average Performance by Feature Set')
    ax3.set_ylabel('Score')
    ax3.set_xlabel('Feature Set')
    ax3.legend(['AUC', 'Acceptance Rate'])
    ax3.grid(axis='y', alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0)

    # Plot 4: Top Models (dynamic based on available data)
    ax4 = axes[1, 1]

    # Get top models (up to 5)
    n_top = min(5, len(df_results))
    top_models = df_results.nlargest(n_top, 'auc_score')

    if len(top_models) > 0:
        y_pos = range(len(top_models))
        bars = ax4.barh(y_pos, top_models['auc_score'].values)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels([f"{row['model_name'][:10]}\n({row['feature_set']})"
                             for _, row in top_models.iterrows()])
        ax4.set_xlabel('AUC Score')
        ax4.set_title(f'Top {len(top_models)} Model Configurations')
        ax4.grid(axis='x', alpha=0.3)

        # Color bars by feature set
        colors = {'all': 'blue', 'traditional': 'green', 'alternative': 'orange'}
        for bar, (_, row) in zip(bars, top_models.iterrows()):
            bar.set_color(colors.get(row['feature_set'], 'gray'))
    else:
        ax4.text(0.5, 0.5, 'No data available', ha='center', va='center')
        ax4.set_title('Top Models')

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=100, bbox_inches='tight')
    plt.show()
    print("\n[INFO] Visualization saved to 'model_comparison.png'")
