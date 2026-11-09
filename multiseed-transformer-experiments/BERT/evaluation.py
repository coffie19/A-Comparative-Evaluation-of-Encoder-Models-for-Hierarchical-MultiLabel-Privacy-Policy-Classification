"""
evaluation.py
=============
All evaluation metrics used in the paper:

Section 4.3   – Precision, Recall, macro-F1, and balanced accuracy
Section 6.1   – Privacy icon prediction accuracy, Cohen's κ, Hellinger distance
Section 8.3   – PriBot predictive accuracy: top-k score, NDCG@k, MAP
Section 8.4   – User-perceived utility (top-k relevance score)

Run this module directly to reproduce Table 1, Table 3, and Fig. 8 results
once the models have been trained and the Twitter QA dataset is available.
"""

import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    HIGH_LEVEL_CATEGORIES, ATTRIBUTES, OPP115_TEST_POLICIES,
    RESULTS_DIR, MODELS_DIR,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Classifier evaluation  (Table 1 / Appendix of the paper)
# ─────────────────────────────────────────────────────────────────────────────

def compute_multilabel_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
) -> Dict:
    """Compute the primary macro-F1 and secondary balanced accuracy.

    Balanced accuracy for a multilabel target is defined here as the mean of
    the ten independent label-wise balanced accuracies. Each label-wise value
    is ``(sensitivity + specificity) / 2``. Labels missing either outcome are
    reported as undefined and excluded from that mean; the full OPP-115 50-
    policy evaluation split contains both outcomes for all ten labels.
    """

    try:
        from sklearn.metrics import (
            balanced_accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required: pip install scikit-learn"
        ) from exc

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError("y_true and y_pred must both be 2-D multilabel matrices")
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred shapes differ: {y_true.shape} vs {y_pred.shape}"
        )
    if y_true.shape[1] != len(label_names):
        raise ValueError("label_names length must match the number of columns")
    if not np.isin(y_true, [0, 1]).all() or not np.isin(y_pred, [0, 1]).all():
        raise ValueError("y_true and y_pred must contain only binary 0/1 values")

    per_label: Dict[str, Dict] = {}
    defined_balanced_accuracies: List[float] = []
    for label_index, label_name in enumerate(label_names):
        true_column = y_true[:, label_index]
        pred_column = y_pred[:, label_index]
        balanced_accuracy = None
        if np.unique(true_column).size == 2:
            balanced_accuracy = float(
                balanced_accuracy_score(true_column, pred_column)
            )
            defined_balanced_accuracies.append(balanced_accuracy)
        per_label[label_name] = {
            "precision": float(
                precision_score(true_column, pred_column, zero_division=0)
            ),
            "recall": float(
                recall_score(true_column, pred_column, zero_division=0)
            ),
            "f1": float(f1_score(true_column, pred_column, zero_division=0)),
            "balanced_accuracy": balanced_accuracy,
            "positive_support": int(true_column.sum()),
            "negative_support": int((true_column == 0).sum()),
        }

    macro_f1 = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )
    mean_balanced_accuracy = (
        float(np.mean(defined_balanced_accuracies))
        if defined_balanced_accuracies
        else None
    )
    return {
        "primary_metric": "macro_f1",
        "macro_f1": macro_f1,
        "balanced_accuracy": mean_balanced_accuracy,
        "balanced_accuracy_definition": (
            "mean over labels of (sensitivity + specificity) / 2"
        ),
        "balanced_accuracy_labels_defined": len(defined_balanced_accuracies),
        "label_count": len(label_names),
        "per_label": per_label,
    }

def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
) -> Dict:
    """
    Compute per-label and macro-averaged Precision, Recall, F1, and balanced
    accuracy. Macro-F1 is the primary result.

    Parameters
    ----------
    y_true : np.ndarray, shape (N, C)
        Ground-truth binary multi-hot vectors.
    y_pred : np.ndarray, shape (N, C)
        Predicted binary multi-hot vectors (already thresholded).
    label_names : List[str]
        Human-readable class names aligned to columns of y_true/y_pred.

    Returns
    -------
    dict with keys per label and "Average".
    """
    metric_summary = compute_multilabel_classification_metrics(
        y_true, y_pred, label_names
    )

    results = {}
    n_labels = len(label_names)

    # Per-label metrics
    for i, name in enumerate(label_names):
        label_metrics = metric_summary["per_label"][name]
        p = label_metrics["precision"]
        r = label_metrics["recall"]
        f = label_metrics["f1"]
        balanced_accuracy = label_metrics["balanced_accuracy"]

        support = int(y_true[:, i].sum())
        results[name] = {
            "precision": round(p, 2),
            "recall": round(r, 2),
            "f1": round(f, 2),
            "balanced_accuracy": (
                round(balanced_accuracy, 2)
                if balanced_accuracy is not None
                else None
            ),
            "support": support,
        }

    # Macro averages
    macro_p = np.mean([results[n]["precision"] for n in label_names])
    macro_r = np.mean([results[n]["recall"] for n in label_names])
    macro_f = metric_summary["macro_f1"]
    macro_balanced_accuracy = metric_summary["balanced_accuracy"]

    results["Average"] = {
        "precision": round(float(macro_p), 2),
        "recall": round(float(macro_r), 2),
        "f1": round(float(macro_f), 2),
        "macro_f1": round(float(macro_f), 2),
        "balanced_accuracy": (
            round(float(macro_balanced_accuracy), 2)
            if macro_balanced_accuracy is not None
            else None
        ),
        "primary_metric": "macro_f1",
        "support": int(y_true.sum()),
    }

    return results


def print_classifier_table(results: Dict, title: str = "") -> None:
    """Pretty-print classifier results matching the paper's table format."""
    header = f"\n{'=' * 72}\n{title}\n{'=' * 72}"
    print(header)
    print(f"{'Category':<35} {'Prec':>6} {'Recall':>7} {'F1':>5} "
          f"{'BalAcc':>7} {'Support':>8}")
    print("-" * 74)
    for name, vals in results.items():
        balanced_accuracy = vals.get("balanced_accuracy")
        balanced_text = (
            f"{balanced_accuracy:.2f}" if balanced_accuracy is not None else "n/a"
        )
        print(
            f"{name:<35} {vals['precision']:>6.2f} {vals['recall']:>7.2f} "
            f"{vals['f1']:>5.2f} {balanced_text:>7} "
            f"{vals['support']:>8d}"
        )
    print("=" * 74)


# ─────────────────────────────────────────────────────────────────────────────
# Privacy Icon evaluation  (Table 3)
# ─────────────────────────────────────────────────────────────────────────────

def cohens_kappa(labels_a: List, labels_b: List) -> float:
    """
    Compute Cohen's κ between two annotators' categorical labels.
    Handles binary (two-class) as well as multi-class problems.
    """
    try:
        from sklearn.metrics import cohen_kappa_score
        if labels_a == labels_b and len(set(labels_a)) == 1:
            return 1.0
        kappa = cohen_kappa_score(
            labels_a,
            labels_b,
            labels=["Red", "Yellow", "Green"],
        )
        return 0.0 if math.isnan(kappa) else float(kappa)
    except ImportError:
        # Manual computation
        cats = list(set(labels_a) | set(labels_b))
        n = len(labels_a)
        agree = sum(a == b for a, b in zip(labels_a, labels_b))
        p_o = agree / n
        # Expected agreement
        p_e = sum(
            (labels_a.count(c) / n) * (labels_b.count(c) / n)
            for c in cats
        )
        return (p_o - p_e) / (1 - p_e + 1e-12)


def hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Hellinger distance between two discrete probability distributions.
        H(P, Q) = (1/√2) * ||√P − √Q||_2
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    # Normalise
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    return float((1.0 / math.sqrt(2)) * np.linalg.norm(np.sqrt(p) - np.sqrt(q)))


def evaluate_privacy_icons(
    polisis_icons: List[Dict[str, str]],
    expert_icons: List[Dict[str, str]],
    icon_names: Optional[List[str]] = None,
) -> Dict:
    """
    Reproduce Table 3 of the paper.

    Parameters
    ----------
    polisis_icons : List[Dict[str, str]]
        [{icon_name: colour}, ...] – one dict per policy – from Polisis.
    expert_icons : List[Dict[str, str]]
        [{icon_name: colour}, ...] – one dict per policy – from expert labels.
    icon_names : List[str], optional
        Icon names to evaluate (defaults to all five Disconnect icons).

    Returns
    -------
    dict: {icon_name: {accuracy, cohen_kappa, hellinger_distance, counts}}
    """
    if icon_names is None:
        icon_names = [
            "Expected Use", "Expected Collection",
            "Precise Location", "Data Retention", "Children Privacy",
        ]

    results = {}
    for icon in icon_names:
        polisis = [d[icon] for d in polisis_icons if icon in d]
        experts = [d[icon] for d in expert_icons if icon in d]
        n = min(len(polisis), len(experts))
        if n == 0:
            continue

        polisis = polisis[:n]
        experts = experts[:n]

        accuracy = sum(p == e for p, e in zip(polisis, experts)) / n
        kappa = cohens_kappa(experts, polisis)

        # Hellinger distance between colour distributions
        colours = ["Red", "Yellow", "Green"]

        def dist_vec(lst):
            return np.array([lst.count(c) for c in colours], dtype=float)

        h_dist = hellinger_distance(dist_vec(experts), dist_vec(polisis))

        counts = {
            "N_Red": experts.count("Red"),
            "N_Yellow": experts.count("Yellow"),
            "N_Green": experts.count("Green"),
        }
        results[icon] = {
            "accuracy": round(accuracy, 2),
            "cohen_kappa": round(kappa, 2),
            "hellinger_distance": round(h_dist, 2),
            **counts,
        }

    return results


def print_icon_table(results: Dict) -> None:
    print("\n" + "=" * 75)
    print("Table 3: Privacy Icon Prediction Accuracy")
    print("=" * 75)
    print(f"{'Icon':<22} {'Acc':>5} {'κ':>6} {'H-dist':>7} "
          f"{'N(R)':>6} {'N(G)':>6} {'N(Y)':>6}")
    print("-" * 75)
    for icon, vals in results.items():
        print(
            f"{icon:<22} {vals['accuracy']:>5.2f} {vals['cohen_kappa']:>6.2f} "
            f"{vals['hellinger_distance']:>7.2f} "
            f"{vals['N_Red']:>6d} {vals['N_Green']:>6d} {vals['N_Yellow']:>6d}"
        )
    avg_acc = np.mean([v["accuracy"] for v in results.values()])
    print(f"\nAverage accuracy: {avg_acc:.2f}")
    print("=" * 75)


# ─────────────────────────────────────────────────────────────────────────────
# PriBot evaluation metrics  (Section 8.3)
# ─────────────────────────────────────────────────────────────────────────────

def top_k_score(
    ranked_lists: List[List[str]],
    ground_truths: List[List[str]],
    k: int,
) -> float:
    """
    Top-k score: fraction of questions with ≥1 correct answer in top-k.

    Parameters
    ----------
    ranked_lists : list of ranked segment lists (one per question)
    ground_truths : list of ground-truth answer sets (one per question)
    k : cut-off

    Returns
    -------
    float in [0, 1]
    """
    correct = 0
    for ranked, gt in zip(ranked_lists, ground_truths):
        if any(seg in gt for seg in ranked[:k]):
            correct += 1
    return correct / len(ranked_lists) if ranked_lists else 0.0


def ndcg_at_k(
    ranked_lists: List[List[str]],
    ground_truths: List[List[str]],
    k: int,
) -> float:
    """
    Normalised Discounted Cumulative Gain @ k.
    rel_i = 1 if answer_i ∈ ground_truth, else 0.
    DCG@k = Σ_{i=1}^{k} rel_i / log2(i + 1)
    NDCG@k = DCG@k / IDCG@k   (IDCG assumes all relevant items ranked first)
    """
    scores = []
    for ranked, gt in zip(ranked_lists, ground_truths):
        gt_set = set(gt)
        dcg = sum(
            1.0 / math.log2(i + 2)
            for i, seg in enumerate(ranked[:k])
            if seg in gt_set
        )
        # Ideal DCG: perfect ranking
        n_rel = min(len(gt_set), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
        scores.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(scores)) if scores else 0.0


def mean_average_precision(
    ranked_lists: List[List[str]],
    ground_truths: List[List[str]],
) -> float:
    """
    MAP = mean of per-query Average Precision (area under precision-recall).

    AP@Q = Σ_{k=1}^{|R|} P@k * rel(k) / |ground_truth|
    """
    aps = []
    for ranked, gt in zip(ranked_lists, ground_truths):
        gt_set = set(gt)
        if not gt_set:
            aps.append(0.0)
            continue
        hits = 0
        ap = 0.0
        for i, seg in enumerate(ranked):
            if seg in gt_set:
                hits += 1
                ap += hits / (i + 1)
        aps.append(ap / len(gt_set))

    return float(np.mean(aps)) if aps else 0.0


def evaluate_qa_model(
    ranked_lists: List[List[str]],
    ground_truths: List[List[str]],
    max_k: int = 7,
    model_name: str = "",
) -> Dict:
    """
    Compute top-k, NDCG@k, and MAP for a QA model over a range of k values.

    Returns dict:
        {
          "top_k": {1: 0.68, 2: 0.75, ...},
          "ndcg": {1: 0.62, 2: 0.70, ...},
          "map": 0.54,
          "model": model_name,
        }
    """
    top_k_dict = {}
    ndcg_dict = {}

    for k in range(1, max_k + 1):
        top_k_dict[k] = round(top_k_score(ranked_lists, ground_truths, k), 3)
        ndcg_dict[k] = round(ndcg_at_k(ranked_lists, ground_truths, k), 3)

    map_score = round(mean_average_precision(ranked_lists, ground_truths), 3)

    return {
        "top_k": top_k_dict,
        "ndcg": ndcg_dict,
        "map": map_score,
        "model": model_name,
    }


def print_qa_table(
    model_results: List[Dict],
    k_values: List[int] = [1, 2, 3, 4, 5, 6, 7],
) -> None:
    """Print top-k and NDCG scores as in Fig. 8 of the paper."""
    print("\n" + "=" * 70)
    print("QA Evaluation Results (top-k score and NDCG)")
    print("=" * 70)

    # Top-k scores
    print("\nTop-k Scores:")
    header = f"{'Model':<15}" + "".join(f"{'k='+str(k):>8}" for k in k_values)
    print(header)
    print("-" * (15 + 8 * len(k_values)))
    for result in model_results:
        row = f"{result['model']:<15}" + "".join(
            f"{result['top_k'].get(k, 0):>8.3f}" for k in k_values
        )
        print(row)

    # NDCG scores
    print("\nNDCG Scores:")
    print(header)
    print("-" * (15 + 8 * len(k_values)))
    for result in model_results:
        row = f"{result['model']:<15}" + "".join(
            f"{result['ndcg'].get(k, 0):>8.3f}" for k in k_values
        )
        print(row)

    # MAP
    print("\nMAP:")
    for result in model_results:
        print(f"  {result['model']:<15}: {result['map']:.3f}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Twitter QA Dataset  (Section 8.1)
# ─────────────────────────────────────────────────────────────────────────────

class TwitterQADataset:
    """
    Loader for the Twitter QA Dataset described in Section 8.1.

    Expected JSON format (one entry per question):
    {
      "question": "Are Evernote notes encrypted at rest?",
      "policy_url": "http://bit.ly/1tauyfh",
      "policy_segments": ["...", "...", ...],
      "ground_truth": ["Segment text that answers the question", ...]
    }

    The dataset is publicly available at:
        https://pribot.org/data.html
    """

    def __init__(self, json_path: str):
        self.json_path = json_path
        self.data: List[Dict] = []

    def load(self) -> "TwitterQADataset":
        with open(self.json_path, encoding="utf-8") as stream:
            self.data = json.load(stream)
        logger.info("Twitter QA Dataset loaded: %d questions.", len(self.data))
        return self

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def questions(self) -> List[str]:
        return [item["question"] for item in self.data]

    def policy_segments(self, idx: int) -> List[str]:
        return self.data[idx]["policy_segments"]

    def ground_truth(self, idx: int) -> List[str]:
        return self.data[idx]["ground_truth"]


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

class PolisisEvaluator:
    """
    Orchestrates the complete evaluation pipeline:
    1. Classifier evaluation  (Table 1)
    2. Privacy icon accuracy  (Table 3)
    3. PriBot QA evaluation   (Fig. 8, Fig. 9)

    Usage
    -----
        evaluator = PolisisEvaluator(hierarchical_clf, vectoriser)
        evaluator.run_classifier_eval(test_samples)
        evaluator.run_icon_eval(test_policies)
        evaluator.run_qa_eval(twitter_qa_dataset)
    """

    def __init__(self, hierarchical_clf, vectoriser):
        self.clf = hierarchical_clf
        self.vectoriser = vectoriser

    # ── 1. Classifier evaluation ─────────────────────────────────────────────

    def run_classifier_eval(self, test_samples: List) -> Dict:
        from ml_layer_embeddings import OPP115Dataset

        texts = [sample.text for sample in test_samples]
        X_test = self.vectoriser.transform(texts)

        # Category classifier
        y_true_cat = np.vstack([
            OPP115Dataset.encode_categories(sample) for sample in test_samples
        ])
        y_pred_cat = self.clf._qry_cat_clf.predict(X_test)
        qry_labels = [category for category in HIGH_LEVEL_CATEGORIES if category != "Other"]
        y_true_cat_qry = y_true_cat[:, [
            HIGH_LEVEL_CATEGORIES.index(category) for category in qry_labels
        ]]
        clf_results = evaluate_classifier(y_true_cat_qry, y_pred_cat, qry_labels)
        print_classifier_table(clf_results, "Table 1: Category Classifier (Query)")

        # Save
        out = os.path.join(RESULTS_DIR, "classifier_results.json")
        with open(out, "w") as stream:
            json.dump(clf_results, stream, indent=2)
        logger.info("Classifier results saved to %s", out)
        return clf_results

    # ── 2. Privacy icon evaluation ────────────────────────────────────────────

    def run_icon_eval(
        self,
        test_policy_annotations: List[Tuple[List[Dict], List[Dict]]],
    ) -> Dict:
        """
        Parameters
        ----------
        test_policy_annotations : List of (polisis_annotations, expert_annotations)
            One tuple per test policy. Each annotation list is the output of
            HierarchicalClassifier.classify_batch() or expert-derived labels.
        """
        from application_layer import PrivacyIconAssigner

        polisis_icons_list = []
        expert_icons_list = []

        for polisis_anns, expert_anns in test_policy_annotations:
            polisis_icons_list.append(
                PrivacyIconAssigner(polisis_anns, strategy="conservative").assign_all()
            )
            expert_icons_list.append(
                PrivacyIconAssigner(expert_anns, strategy="conservative").assign_all()
            )

        results = evaluate_privacy_icons(polisis_icons_list, expert_icons_list)
        print_icon_table(results)

        out = os.path.join(RESULTS_DIR, "icon_results.json")
        with open(out, "w") as stream:
            json.dump(results, stream, indent=2)
        logger.info("Icon results saved to %s", out)
        return results

    # ── 3. PriBot QA evaluation ────────────────────────────────────────────────

    def run_qa_eval(
        self,
        qa_dataset: "TwitterQADataset",
        pribot: "PriBot",
        bm25: "BM25Retrieval",
        max_k: int = 7,
    ) -> Dict:
        from application_layer import PriBot, BM25Retrieval

        pribot_ranked = []
        bm25_ranked = []
        random_ranked = []
        ground_truths = []

        for idx in range(len(qa_dataset)):
            entry = qa_dataset.data[idx]
            question = entry["question"]
            segments = entry["policy_segments"]
            gt = entry["ground_truth"]
            ground_truths.append(gt)

            # PriBot
            answers = pribot.answer(question, segments, top_k=max_k)
            pribot_ranked.append([answer["segment"] for answer in answers])

            # BM25
            bm25.fit(segments)
            bm25_answers = bm25.rank(question, top_k=max_k)
            bm25_ranked.append([answer["segment"] for answer in bm25_answers])

            # Random
            rng = np.random.default_rng(seed=42 + idx)
            rnd_idx = rng.choice(
                len(segments), size=min(max_k, len(segments)), replace=False
            )
            random_ranked.append([segments[index] for index in rnd_idx])

        results = {
            "PriBot": evaluate_qa_model(
                pribot_ranked, ground_truths, max_k, "PriBot"
            ),
            "Retrieval": evaluate_qa_model(
                bm25_ranked, ground_truths, max_k, "Retrieval"
            ),
            "Random": evaluate_qa_model(
                random_ranked, ground_truths, max_k, "Random"
            ),
        }

        print_qa_table(list(results.values()))

        out = os.path.join(RESULTS_DIR, "qa_results.json")
        with open(out, "w") as stream:
            json.dump(results, stream, indent=2)
        logger.info("QA results saved to %s", out)
        return results
