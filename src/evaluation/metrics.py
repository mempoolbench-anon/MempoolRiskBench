"""metrics + bootstrap CIs.

per task:
    revert    -> F1, ROC-AUC (decisive), accuracy (secondary)
    mev       -> PR-AUC, P@100, P@1000 (decisive), R@1000 (secondary)
    drop      -> F1, ROC-AUC

decisive metrics get 1k-resample bootstrap 95% CIs on the test set.
everything is model-agnostic -- inputs are (scores, labels) numpy arrays.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def precision_at_k(scores, labels, k):
    """frac of top-k scored that are positive."""
    if len(scores) < k:
        k = len(scores)
    top = np.argsort(scores)[-k:]
    return float(labels[top].sum()) / k


def recall_at_k(scores, labels, k):
    """frac of all positives captured in top-k."""
    if len(scores) < k:
        k = len(scores)
    n_pos = labels.sum()
    if n_pos == 0:
        return 0.0
    top = np.argsort(scores)[-k:]
    return float(labels[top].sum()) / n_pos


def _revert_metrics(scores, labels):
    valid = labels >= 0      # -1 means no receipt observed
    if valid.sum() < 10:
        return {}
    s = scores[valid]
    l = labels[valid]
    pred = (s > 0.5).astype(int)
    out = {
        "revert_f1": float(f1_score(l, pred, zero_division=0)),
        "revert_accuracy": float(accuracy_score(l, pred)),
    }
    if len(np.unique(l)) > 1:
        out["revert_auc"] = float(roc_auc_score(l, s))
    return out


def _mev_metrics(scores, labels):
    out = {}
    if labels.sum() < 1:
        return out
    if len(np.unique(labels)) > 1:
        out["mev_pr_auc"] = float(average_precision_score(labels, scores))
    out["mev_p_at_100"] = precision_at_k(scores, labels, 100)
    out["mev_p_at_1000"] = precision_at_k(scores, labels, 1000)
    out["mev_recall_at_1000"] = recall_at_k(scores, labels, 1000)
    return out


def _drop_metrics(scores, labels):
    valid = labels >= 0      # -1 means right-censored past 24h maturity
    if valid.sum() < 10:
        return {}
    s = scores[valid]
    l = labels[valid]
    pred = (s > 0.5).astype(int)
    out = {"drop_f1": float(f1_score(l, pred, zero_division=0))}
    if len(np.unique(l)) > 1:
        out["drop_auc"] = float(roc_auc_score(l, s))
    return out


def compute_all_metrics(preds_dict, labels, task="all"):
    """preds_dict: {'revert','mev','drop'} -> scores. labels: (N, 3).
    task: 'revert' | 'mev' | 'drop' | 'all'.
    """
    result = {}

    if task in ("all", "revert"):
        result.update(_revert_metrics(preds_dict["revert"], labels[:, 0]))
    if task in ("all", "mev"):
        result.update(_mev_metrics(preds_dict["mev"], labels[:, 1]))
    if task in ("all", "drop"):
        result.update(_drop_metrics(preds_dict["drop"], labels[:, 2]))

    return result


def bootstrap_ci(metric_fn, preds, labels, n_bootstrap=1000, ci=0.95, seed=42):
    """percentile bootstrap CI for a single metric. returns (point, lo, hi)."""
    rng = np.random.RandomState(seed)
    n = len(preds)
    point = metric_fn(preds, labels)

    boots = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        try:
            boots.append(metric_fn(preds[idx], labels[idx]))
        except (ValueError, ZeroDivisionError):
            # eg. a resample with a single class — skip
            continue

    if not boots:
        return point, point, point

    boots = np.array(boots)
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boots, 100 * alpha))
    hi = float(np.percentile(boots, 100 * (1 - alpha)))
    return point, lo, hi


def compute_all_metrics_with_ci(preds_dict, labels, n_bootstrap=1000, seed=42):
    """same as compute_all_metrics but each value is (point, lo, hi)."""
    result = {}

    # Revert metrics with CI
    valid = labels[:, 0] >= 0
    if valid.sum() >= 10:
        rev_scores = preds_dict["revert"][valid]
        rev_labels = labels[:, 0][valid]

        def rev_f1(p, l):
            return float(f1_score(l, (p > 0.5).astype(int), zero_division=0))

        def rev_auc(p, l):
            if len(np.unique(l)) > 1:
                return float(roc_auc_score(l, p))
            return 0.0

        result["revert_f1"] = bootstrap_ci(rev_f1, rev_scores, rev_labels, n_bootstrap, seed=seed)
        result["revert_auc"] = bootstrap_ci(rev_auc, rev_scores, rev_labels, n_bootstrap, seed=seed)

    # MEV metrics with CI
    mev_scores = preds_dict["mev"]
    mev_labels = labels[:, 1]
    if mev_labels.sum() >= 1:
        def mev_prauc(p, l):
            if len(np.unique(l)) > 1:
                return float(average_precision_score(l, p))
            return 0.0

        def mev_p100(p, l):
            return precision_at_k(p, l, 100)

        def mev_p1000(p, l):
            return precision_at_k(p, l, 1000)

        result["mev_pr_auc"] = bootstrap_ci(mev_prauc, mev_scores, mev_labels, n_bootstrap, seed=seed)
        result["mev_p_at_100"] = bootstrap_ci(mev_p100, mev_scores, mev_labels, n_bootstrap, seed=seed)
        result["mev_p_at_1000"] = bootstrap_ci(mev_p1000, mev_scores, mev_labels, n_bootstrap, seed=seed)

    # Drop metrics with CI (filter -1 right-censored labels first)
    drop_valid = labels[:, 2] >= 0
    if drop_valid.sum() >= 10:
        drop_scores = preds_dict["drop"][drop_valid]
        drop_labels = labels[:, 2][drop_valid]

        def drop_f1(p, l):
            return float(f1_score(l, (p > 0.5).astype(int), zero_division=0))

        def drop_auc(p, l):
            if len(np.unique(l)) > 1:
                return float(roc_auc_score(l, p))
            return 0.0

        result["drop_f1"] = bootstrap_ci(drop_f1, drop_scores, drop_labels, n_bootstrap, seed=seed)
        result["drop_auc"] = bootstrap_ci(drop_auc, drop_scores, drop_labels, n_bootstrap, seed=seed)

    return result
