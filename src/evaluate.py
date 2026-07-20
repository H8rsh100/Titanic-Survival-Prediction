import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
FIGURES_DIR = "reports/figures"

def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Compute comprehensive classification metrics.
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
    }
    if y_prob is not None:
        metrics['ROC-AUC'] = roc_auc_score(y_true, y_prob)
    return metrics

def plot_survival_by_demographics(df, save_dir=FIGURES_DIR):
    """
    Generate & save EDA figure for survival by Sex and Pclass.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sex distribution
    sns.barplot(data=df, x='Sex', y='Survived', palette='Set2', ax=axes[0], errorbar=None)
    axes[0].set_title('Survival Rate by Sex', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Survival Rate')
    axes[0].set_ylim(0, 1)
    for p in axes[0].patches:
        axes[0].annotate(f'{p.get_height():.2%}', (p.get_x() + p.get_width() / 2., p.get_height() / 2.),
                         ha='center', va='center', fontsize=11, color='white', fontweight='bold')
        
    # Pclass distribution
    sns.barplot(data=df, x='Pclass', y='Survived', hue='Sex', palette='crest', ax=axes[1], errorbar=None)
    axes[1].set_title('Survival Rate by Pclass and Sex', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Survival Rate')
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    output_path = os.path.join(save_dir, 'survival_by_sex_pclass.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path

def plot_correlation_matrix(df, save_dir=FIGURES_DIR):
    """
    Generate & save correlation heatmap of numeric features.
    """
    os.makedirs(save_dir, exist_ok=True)
    numeric_df = df.select_dtypes(include=[np.number])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', square=True, linewidths=0.5)
    plt.title('Numeric Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(save_dir, 'correlation_heatmap.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path

def plot_confusion_matrix(y_true, y_pred, model_name="Best Model", save_dir=FIGURES_DIR):
    """
    Generate & save confusion matrix display.
    """
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Perished (0)', 'Survived (1)'])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path

def plot_feature_importance(feature_names, importances, top_n=15, save_dir=FIGURES_DIR):
    """
    Plot and save top N feature importances.
    """
    os.makedirs(save_dir, exist_ok=True)
    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    fi_df = fi_df.sort_values(by='Importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=fi_df, x='Importance', y='Feature', palette='viridis')
    plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    
    output_path = os.path.join(save_dir, 'feature_importance.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path

def plot_cv_comparison(cv_results, save_dir=FIGURES_DIR):
    """
    Plot cross-validation accuracy across evaluated models.
    """
    os.makedirs(save_dir, exist_ok=True)
    df_cv = pd.DataFrame(cv_results)
    
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=df_cv, x='Model', y='Mean_CV_Accuracy', palette='magma')
    plt.title('5-Fold Stratified CV Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Mean CV Accuracy')
    plt.ylim(0.70, 0.90)
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.4f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
                    
    plt.tight_layout()
    output_path = os.path.join(save_dir, 'cv_model_comparison.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path

if __name__ == "__main__":
    print("Evaluation module loaded.")
