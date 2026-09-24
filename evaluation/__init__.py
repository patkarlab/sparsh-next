"""
SPARSH Evaluation module with classification metrics and CSV export utilities.
"""

from .metrics import (
    compute_classification_metrics,
    compute_per_class_metrics,
    print_classification_report,
    get_confusion_matrix_data,
    get_pr_curve_data,
    get_roc_curve_data,
    export_confusion_matrix_csv,
    export_confusion_matrix_normalized_csv,
    export_classification_report_csv,
    export_per_class_metrics_csv,
    export_roc_curve_csv,
    export_roc_auc_summary_csv,
    export_pr_curve_csv,
    export_summary_metrics_csv,
    export_predictions_csv,
    export_predictions_with_folds_csv,
    export_per_fold_metrics_csv,
)

__all__ = [
    "compute_classification_metrics",
    "compute_per_class_metrics",
    "print_classification_report",
    "get_confusion_matrix_data",
    "get_pr_curve_data",
    "get_roc_curve_data",
    "export_confusion_matrix_csv",
    "export_confusion_matrix_normalized_csv",
    "export_classification_report_csv",
    "export_per_class_metrics_csv",
    "export_roc_curve_csv",
    "export_roc_auc_summary_csv",
    "export_pr_curve_csv",
    "export_summary_metrics_csv",
    "export_predictions_csv",
    "export_predictions_with_folds_csv",
    "export_per_fold_metrics_csv",
]
