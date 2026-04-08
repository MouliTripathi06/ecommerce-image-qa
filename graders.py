"""
Deterministic agent graders for all 3 tasks.
Each grader returns a float in [0.0, 1.0] with partial credit.
"""
from typing import List, Optional, Dict, Tuple


def _f1_score(predicted: List[str], ground_truth: List[str]) -> float:
    """Compute F1 score over label sets."""
    if not ground_truth and not predicted:
        return 1.0  # both clean
    if not ground_truth:
        return 0.0  # hallucinated defects
    if not predicted:
        return 0.0  # missed all defects

    pred_set = set(predicted)
    gt_set = set(ground_truth)

    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def grade_easy(
    predicted_defects: List[str],
    ground_truth_defects: List[str],
) -> Tuple[float, Dict]:
    """
    Easy task: detect the single defect type (or confirm clean).
    Binary correct/wrong with small partial credit for near-misses.
    """
    pred = set(predicted_defects)
    gt = set(ground_truth_defects)

    if pred == gt:
        score = 1.0
        detail = "exact_match"
    elif len(gt) == 0 and len(pred) > 0:
        score = 0.0
        detail = "false_positive"
    elif len(gt) > 0 and len(pred) == 0:
        score = 0.0
        detail = "missed_defect"
    elif gt.issubset(pred):
        # Got the right one but also added extras
        score = 0.5
        detail = "correct_with_extras"
    elif pred.issubset(gt):
        # Partial hit — found some but not all
        score = 0.5
        detail = "partial_hit"
    else:
        score = 0.0
        detail = "wrong_label"

    return score, {"detail": detail, "predicted": list(pred), "ground_truth": list(gt)}


def grade_medium(
    predicted_defects: List[str],
    ground_truth_defects: List[str],
) -> Tuple[float, Dict]:
    """
    Medium task: identify ALL defects (multi-label).
    Scored with F1 over the label set.
    """
    f1 = _f1_score(predicted_defects, ground_truth_defects)
    pred_set = set(predicted_defects)
    gt_set = set(ground_truth_defects)

    return f1, {
        "f1": f1,
        "predicted": list(pred_set),
        "ground_truth": list(gt_set),
        "correct_labels": list(pred_set & gt_set),
        "missed_labels": list(gt_set - pred_set),
        "extra_labels": list(pred_set - gt_set),
    }


def grade_hard(
    predicted_defects: List[str],
    ground_truth_defects: List[str],
    predicted_severity: Optional[float],
    ground_truth_severity: float,
    predicted_recommendation: Optional[str],
    ground_truth_recommendation: str,
) -> Tuple[float, Dict]:
    """
    Hard task: multi-label F1 + severity MAE + recommendation accuracy.
    Weighted blend: 40% F1, 30% severity, 30% recommendation.
    """
    # Component 1: F1 on defect labels (40%)
    f1 = _f1_score(predicted_defects, ground_truth_defects)

    # Component 2: Severity score (30%) — 1 - normalized MAE
    if predicted_severity is None:
        severity_score = 0.0
    else:
        mae = abs(predicted_severity - ground_truth_severity)
        severity_score = round(max(0.0, 1.0 - mae), 4)

    # Component 3: Recommendation accuracy (30%)
    if predicted_recommendation is None:
        rec_score = 0.0
    elif predicted_recommendation == ground_truth_recommendation:
        rec_score = 1.0
    else:
        # Partial: retouch vs approve/reject is closer than approve vs reject
        combos = {
            ("approve", "retouch"): 0.3,
            ("retouch", "approve"): 0.3,
            ("retouch", "reject"): 0.3,
            ("reject", "retouch"): 0.3,
            ("approve", "reject"): 0.0,
            ("reject", "approve"): 0.0,
        }
        rec_score = combos.get((predicted_recommendation, ground_truth_recommendation), 0.0)

    total = round(0.40 * f1 + 0.30 * severity_score + 0.30 * rec_score, 4)

    return total, {
        "total": total,
        "f1_defects": f1,
        "f1_weight": 0.40,
        "severity_score": severity_score,
        "severity_weight": 0.30,
        "rec_score": rec_score,
        "rec_weight": 0.30,
        "predicted_severity": predicted_severity,
        "ground_truth_severity": ground_truth_severity,
        "predicted_recommendation": predicted_recommendation,
        "ground_truth_recommendation": ground_truth_recommendation,
    }
