# src/utils/metrics.py

import pandas as pd


def micro_f1_at_k(pred_labels: pd.DataFrame, true_labels: pd.DataFrame, top_k=1):
    """
    Computes quantitative evaluation metrics, such as k-precision, k-recall,
    and k-F1-score, by comparing true labels against the top-k predicted
    labels for each text unit.

    Parameters
    ----------
    pred_labels : pandas.DataFrame
        DataFrame containing model-predicted labels. Must include columns:
        - '*_id': identifier columns for report/document (report_id), sentence (sentence_id), ....
        - 'icd_code': predicted ICD-10 code(s) for each row.
        Each row represents one predicted label per text unit.
        - 'distance': model output score or inverse-proximity for each label.
        - 'threshold_passed': indicator if predicted code is within decision boundary

    true_labels : pandas.DataFrame
        DataFrame of ground truth labels. Must include:
        - 'report_id': identifier for the report/document.
        - 'sentence_id': identifier for the sentence or text unit.
        - 'icd_code': ground truth ICD-10 code(s) for each text unit.
        Each row denotes one reference label per text unit.

    top_k : int, default=1
        Number of top-ranked predictions per text unit to consider as "retrieved".

    level : {'leaf', 'subcategory', 'category', ''}, default='leaf'
        Granularity level for the evaluation (e.g., leaf ICD-10 codes).
        As there can be multiple subcategories for a single code, the one
        closest to the root is chosen. If a code is leaf node and highest
        subcategory this is identical to setting level to leaf.

    granularity : {'sentence', 'report'}, default='sentence'
        Text granularity for which to aggregate labels.

    Notes
    -----
    For top-k metrics:
        - Precision@k: Fraction of top-k predictions that are true labels.
        - Recall@k: Fraction of ground truth labels present in top-k predictions.
        - F1@k: Harmonic mean of Precision@k and Recall@k.

    Returns
    -------
    scores : dict
        Dictionary with precision_at_k, recall_at_k, f1_at_k and k (=top_k).
    """
    # Filter out predictions where threshold_passed is False
    pred_labels = pred_labels[pred_labels["threshold_passed"]]

    group_keys = [c for c in pred_labels.columns if c.endswith("_id")]
    # Prepare ground truth nested sets
    true_grouped = true_labels.groupby(group_keys)["icd_code"].apply(set)

    # Prepare predicted top-k per group (lowest distance = highest score)
    preds_sorted = pred_labels.sort_values(group_keys + ["distance"])
    top_k_preds = preds_sorted.groupby(group_keys).head(top_k)
    pred_grouped = top_k_preds.groupby(group_keys)["icd_code"].apply(set)

    # Union with empty sets where missing
    keys = sorted(set(true_grouped.index).union(set(pred_grouped.index)))
    precisions, recalls = [], []

    for key in keys:
        y_pred = pred_grouped.get(key, set())
        y_true = true_grouped.get(key, set())
        tp = len(y_pred & y_true)
        precision = tp / top_k if top_k > 0 else 0.0
        recall = tp / len(y_true) if len(y_true) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)

    precision_at_k = sum(precisions) / len(precisions) if precisions else 0.0
    recall_at_k = sum(recalls) / len(recalls) if recalls else 0.0
    if precision_at_k + recall_at_k > 0:
        f1_at_k = 2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k)
    else:
        f1_at_k = 0.0

    return {
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "micro_f1_at_k": f1_at_k,
        "k": top_k,
    }


def macro_f1_at_k(
    pred_labels: pd.DataFrame,
    true_labels: pd.DataFrame,
    top_k=1,
    group_keys=["report_id", "sentence_id"],
):
    """
    Computes macro-F1@k: averages F1@k across all ICD-10 codes.
    Additionally returns precision, recall and F1 at k per true code.
    """
    # Filter out predictions where threshold_passed is False
    pred_labels = pred_labels[pred_labels["threshold_passed"]]

    # Sort predictions & select top_k per text unit
    preds_sorted = pred_labels.sort_values(group_keys + ["distance"])
    top_k_preds = preds_sorted.groupby(group_keys).head(top_k)

    # Create sets of predictions/truth per group
    pred_grouped = top_k_preds.groupby(group_keys)["code"].apply(set)
    true_grouped = true_labels.groupby(group_keys)["y_true"].apply(set)
    keys = sorted(set(true_grouped.index).union(set(pred_grouped.index)))

    # Flatten: build code-to-sample sets
    all_codes = set(true_labels["y_true"].unique()).union(
        set(pred_labels["code"].unique())
    )
    code_metrics = {}
    for code in all_codes:
        # For each sample, is the code present in predicted top-k and in ground truth?
        tps = fps = fns = 0
        for key in keys:
            pred_set = pred_grouped.get(key, set())
            true_set = true_grouped.get(key, set())
            if code in pred_set and code in true_set:
                tps += 1
            if code in pred_set and code not in true_set:
                fps += 1
            if code not in pred_set and code in true_set:
                fns += 1
        precision = tps / (tps + fps) if (tps + fps) > 0 else 0.0
        recall = tps / (tps + fns) if (tps + fns) > 0 else 0.0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        code_metrics[code] = {"precision": precision, "recall": recall, "f1": f1}

    macro_f1 = (
        sum([v["f1"] for v in code_metrics.values()]) / len(code_metrics)
        if code_metrics
        else 0.0
    )

    return {"macro_f1_at_k": macro_f1, "per_code": code_metrics, "k": top_k}
