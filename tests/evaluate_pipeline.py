"""Comprehensive evaluation of the complete hallucination detection and claim verification pipeline."""
import sys
import os
import json
import re
from typing import Dict, Any, List

# Ensure project root is in sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from services.evidence import retrieve_evidence
from services.verification import (
    verify_claim,
    SUPPORTED,
    CONTRADICTED,
    INSUFFICIENT_EVIDENCE
)

# Labeled evaluation dataset with balanced class distribution covering Section 8 regression claims
EVALUATION_DATASET = [
    # Class: SUPPORTED (8 claims)
    {"claim": "The boiling point of nitrogen is approximately -196°C at atmospheric pressure.", "expected_verdict": SUPPORTED, "has_correction": False},
    {"claim": "The chemical formula for ammonia is NH3.", "expected_verdict": SUPPORTED, "has_correction": False},
    {"claim": "The currency of Bangladesh is the taka.", "expected_verdict": SUPPORTED, "has_correction": False},
    {"claim": "A light-year measures distance rather than time.", "expected_verdict": SUPPORTED, "has_correction": False},
    {"claim": "The human stomach produces hydrochloric acid.", "expected_verdict": SUPPORTED, "has_correction": False},
    {"claim": "The capital of Portugal is Lisbon.", "expected_verdict": SUPPORTED, "has_correction": False},
    {"claim": "The square root of 225 is 15.", "expected_verdict": SUPPORTED, "has_correction": False},
    {"claim": "The chemical symbol for calcium is Ca.", "expected_verdict": SUPPORTED, "has_correction": False},
    {"claim": "The first photograph of a black hole was released in 2019.", "expected_verdict": SUPPORTED, "has_correction": False},
    {"claim": "A standard piano has 88 keys.", "expected_verdict": SUPPORTED, "has_correction": False},

    # Class: CONTRADICTED (8 claims)
    {"claim": "All planets in the Solar System have rings.", "expected_verdict": CONTRADICTED, "has_correction": True, "expected_correction_contains": "giant planets"},
    {"claim": "Every species of penguin lives in the Arctic.", "expected_verdict": CONTRADICTED, "has_correction": True, "expected_correction_contains": "Southern Hemisphere"},
    {"claim": "The capital of South Korea is Busan.", "expected_verdict": CONTRADICTED, "has_correction": True, "expected_correction_contains": "Seoul"},
    {"claim": "The capital of Switzerland is Zurich.", "expected_verdict": CONTRADICTED, "has_correction": True, "expected_correction_contains": "Bern"},
    {"claim": "A regular pentagon has six sides.", "expected_verdict": CONTRADICTED, "has_correction": True, "expected_correction_contains": "five"},
    {"claim": "The chemical symbol for gold is Ag.", "expected_verdict": CONTRADICTED, "has_correction": True, "expected_correction_contains": "Au"},
    {"claim": "The largest planet in the Solar System is Mercury.", "expected_verdict": CONTRADICTED, "has_correction": True, "expected_correction_contains": "Jupiter"},
    {"claim": "The Sun revolves around the Earth.", "expected_verdict": CONTRADICTED, "has_correction": True, "expected_correction_contains": "Earth"},

    # Class: INSUFFICIENT_EVIDENCE (8 claims)
    {"claim": "My best friend moved to Canada last month.", "expected_verdict": INSUFFICIENT_EVIDENCE, "has_correction": False},
    {"claim": "My college has exactly 7,500 enrolled students this academic year.", "expected_verdict": INSUFFICIENT_EVIDENCE, "has_correction": False},
    {"claim": "A company announced a commercially available quantum computer with unlimited processing power today.", "expected_verdict": INSUFFICIENT_EVIDENCE, "has_correction": False},
    {"claim": "Scientists discovered a new human organ in Hyderabad yesterday.", "expected_verdict": INSUFFICIENT_EVIDENCE, "has_correction": False},
    {"claim": "My professor had oatmeal for breakfast this morning.", "expected_verdict": INSUFFICIENT_EVIDENCE, "has_correction": False},
    {"claim": "An unknown microscopic organism was spotted on my desk.", "expected_verdict": INSUFFICIENT_EVIDENCE, "has_correction": False},
    {"claim": "A person named Alexander visited a cafe near my house today.", "expected_verdict": INSUFFICIENT_EVIDENCE, "has_correction": False},
    {"claim": "Scientists announced a warp drive breakthrough today.", "expected_verdict": INSUFFICIENT_EVIDENCE, "has_correction": False}
]

def run_evaluation() -> Dict[str, Any]:
    print("=================================================================")
    print("Evaluating Complete Hallucination Detection & Verification Pipeline")
    print("=================================================================\n")

    classes = [SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE]
    confusion_matrix = {t: {p: 0 for p in classes} for t in classes}
    total = len(EVALUATION_DATASET)
    correct = 0
    evidence_relevance_scores = []
    correction_success = 0
    contradicted_count = 0
    confidences_correct = []
    confidences_incorrect = []

    details = []

    for item in EVALUATION_DATASET:
        claim = item["claim"]
        expected = item["expected_verdict"]
        ev = retrieve_evidence(claim)
        res = verify_claim(claim, ev)
        predicted = res["verdict"]
        confidence = float(res.get("confidence", 0.0))
        corr = res.get("correction")

        is_match = (predicted == expected)
        if is_match:
            correct += 1
            confidences_correct.append(confidence)
        else:
            confidences_incorrect.append(confidence)

        confusion_matrix[expected][predicted] += 1

        # Track evidence relevance
        rel = float(ev.get("relevance_score", 0.0))
        evidence_relevance_scores.append(rel)

        # Track correction relevance
        if expected == CONTRADICTED:
            contradicted_count += 1
            if item.get("has_correction"):
                check_term = item.get("expected_correction_contains", "").lower()
                if corr and (not check_term or check_term in corr.lower()):
                    correction_success += 1

        details.append({
            "claim": claim,
            "expected": expected,
            "predicted": predicted,
            "confidence": confidence,
            "relevance": rel,
            "correction": corr,
            "is_correct": is_match
        })

    accuracy = correct / total

    # Per-class metrics
    per_class_metrics = {}
    for cls in classes:
        tp = confusion_matrix[cls][cls]
        fp = sum(confusion_matrix[o][cls] for o in classes if o != cls)
        fn = sum(confusion_matrix[cls][o] for o in classes if o != cls)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class_metrics[cls] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "support": sum(confusion_matrix[cls].values())
        }

    avg_evidence_relevance = sum(evidence_relevance_scores) / len(evidence_relevance_scores) if evidence_relevance_scores else 0.0
    corr_relevance_rate = correction_success / contradicted_count if contradicted_count > 0 else 0.0
    mean_conf_correct = sum(confidences_correct) / len(confidences_correct) if confidences_correct else 0.0
    mean_conf_incorrect = sum(confidences_incorrect) / len(confidences_incorrect) if confidences_incorrect else 0.0

    report = {
        "total_claims": total,
        "correct_predictions": correct,
        "accuracy": round(accuracy, 4),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion_matrix,
        "average_evidence_relevance": round(avg_evidence_relevance, 4),
        "correction_relevance_rate": round(corr_relevance_rate, 4),
        "mean_confidence_correct": round(mean_conf_correct, 4),
        "mean_confidence_incorrect": round(mean_conf_incorrect, 4),
        "details": details
    }

    # Print summary table
    print(f"Overall Accuracy: {accuracy * 100:.2f}% ({correct}/{total})\n")

    print(f"{'Class':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
    print("-" * 69)
    for cls in classes:
        m = per_class_metrics[cls]
        print(f"{cls:<25} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1_score']:<12.4f} {m['support']:<8}")
    print("-" * 69)

    header_label = "Actual \\ Predicted"
    print("\nConfusion Matrix:")
    print(f"{header_label:<25} {'SUPPORTED':<14} {'CONTRADICTED':<14} {'INSUFFICIENT':<14}")
    for act in classes:
        row = confusion_matrix[act]
        print(f"{act:<25} {row[SUPPORTED]:<14} {row[CONTRADICTED]:<14} {row[INSUFFICIENT_EVIDENCE]:<14}")

    print(f"\nAverage Evidence Relevance: {avg_evidence_relevance:.4f}")
    print(f"Correction Relevance Rate:  {corr_relevance_rate * 100:.2f}%")
    print(f"Mean Confidence (Correct):  {mean_conf_correct:.4f}")
    print(f"Mean Confidence (Incorrect):{mean_conf_incorrect:.4f}")

    return report

if __name__ == "__main__":
    run_evaluation()
