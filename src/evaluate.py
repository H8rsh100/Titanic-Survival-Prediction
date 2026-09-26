import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve,
    average_precision_score, precision_recall_curve
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

def plot_roc_curve(y_true, y_prob, model_name="Best Model", save_dir=FIGURES_DIR):
    """
    Plot the ROC curve alongside the precision-recall curve.

    Accuracy alone hides where a classifier actually fails. ROC-AUC measures
    ranking quality across all thresholds, while the PR curve exposes precision
    at the operating point, which matters more under class imbalance.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auc_value = roc_auc_score(y_true, y_prob)
    ap_value = average_precision_score(y_true, y_prob)
    prevalence = float(np.mean(y_true))
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    
    axes[0].plot(fpr, tpr, color='#D4AF37', linewidth=2.2,
                 label=f'{model_name} (AUC = {auc_value:.4f})')
    axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1.2, label='Random guess')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve (out-of-fold)', fontsize=13, fontweight='bold')
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1.02)
    axes[0].legend(loc='lower right', fontsize=9)
    
    axes[1].plot(recall, precision, color='#00A6C8', linewidth=2.2,
                 label=f'{model_name} (AP = {ap_value:.4f})')
    axes[1].axhline(prevalence, linestyle='--', color='gray', linewidth=1.2,
                    label=f'Baseline (prevalence = {prevalence:.4f})')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve (out-of-fold)', fontsize=13, fontweight='bold')
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1.02)
    axes[1].legend(loc='lower left', fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(save_dir, 'roc_pr_curves.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path

def plot_threshold_sweep(y_true, y_prob, save_dir=FIGURES_DIR, n_points=101):
    """
    Sweep the decision threshold and plot accuracy, F1 and FPR against it.

    The 0.5 default is rarely optimal for an imbalanced target, and this makes
    the trade-off explicit rather than implicit.
    """
    os.makedirs(save_dir, exist_ok=True)
    thresholds = np.linspace(0.05, 0.95, n_points)
    rows = []
    for threshold in thresholds:
        preds = (y_prob >= threshold).astype(int)
        rows.append({
            'Threshold': threshold,
            'Accuracy': accuracy_score(y_true, preds),
            'F1': f1_score(y_true, preds, zero_division=0),
            'Precision': precision_score(y_true, preds, zero_division=0),
            'Recall': recall_score(y_true, preds, zero_division=0),
        })
    sweep = pd.DataFrame(rows)
    
    best_row = sweep.loc[sweep['F1'].idxmax()]
    default_index = int(np.abs(thresholds - 0.5).argmin())
    default_row = sweep.iloc[default_index]
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(sweep['Threshold'], sweep['Accuracy'], linewidth=2, label='Accuracy', color='#1f77b4')
    ax.plot(sweep['Threshold'], sweep['F1'], linewidth=2, label='F1 Score', color='#d62728')
    ax.plot(sweep['Threshold'], sweep['Precision'], linewidth=1.6, linestyle='--',
            label='Precision', color='#2ca02c')
    ax.plot(sweep['Threshold'], sweep['Recall'], linewidth=1.6, linestyle='--',
            label='Recall', color='#9467bd')
    ax.axvline(0.5, color='gray', linestyle=':', linewidth=1.4, label='Default 0.50')
    ax.axvline(best_row['Threshold'], color='black', linestyle='-.', linewidth=1.4,
               label=f"Best F1 @ {best_row['Threshold']:.2f} ({best_row['F1']:.4f})")
    
    ax.set_xlabel('Decision Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Decision Threshold Sweep (out-of-fold)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.02)
    ax.legend(loc='lower left', fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(save_dir, 'threshold_sweep.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"  Threshold {default_row['Threshold']:.2f} -> Accuracy {default_row['Accuracy']:.4f}, "
          f"F1 {default_row['F1']:.4f} (default)")
    print(f"  Threshold {best_row['Threshold']:.2f} -> Accuracy {best_row['Accuracy']:.4f}, "
          f"F1 {best_row['F1']:.4f} (best F1)")
    
    return output_path

if __name__ == "__main__":
    print("Evaluation module loaded.")
