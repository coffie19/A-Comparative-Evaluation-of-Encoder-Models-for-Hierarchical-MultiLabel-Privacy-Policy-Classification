"""Compare all four completed studies; never pool smoke or unmatched runs."""
import argparse
import csv
from pathlib import Path
import statistics
from multiseed import METRICS, aggregate, read, write

MODELS = ["BERT", "PrivBERT", "RoBERTa", "LegalBERT"]


def compare(output_root):
    root = Path(output_root) / "full"
    reports = {}
    common = None
    for model in MODELS:
        folder = root / model
        spec = read(folder / "study_specification.json")
        if spec["mode"] != "full" or spec["model"] != model:
            raise ValueError("Only correctly identified full-model studies may be compared.")
        contract = {k: spec[k] for k in ["seeds", "data", "training", "max_length", "expected_steps", "pretraining", "metric_contract"]}
        contract["common_code"] = {k: spec["source_hashes"][k] for k in ["multiseed.py", "evaluation.py", "opp115_hierarchy.py", "opp115_csv_loader.py", "opp115_policy_split.py"]}
        if common is not None and contract != common:
            raise ValueError("Models differ in seeds, data, training, or metric contract.")
        common = contract
        reports[model] = aggregate(folder, spec)
    lines = ["# Matched multi-seed OPP-115 comparison", "", "Mean ± sample standard deviation (ddof=1); scores on the 0–1 scale.", "",
             "| Model | Category Macro-F1 | Hierarchical Macro-F1 | Gated fine-grained Macro-F1 | n |", "|---|---:|---:|---:|---:|"]
    csv_rows = []
    for model, report in reports.items():
        values = report["metrics"]
        formatted = [f"{values[k]['mean']:.6f} ± {values[k]['std']:.6f}" for k in METRICS]
        lines.append("| " + " | ".join([model, *formatted, str(len(report["seeds"]))]) + " |")
        csv_rows.extend(dict(model=model, metric=k, **v) for k, v in values.items())
    paired = {}
    for metric in METRICS:
        priv = {r["seed"]: r[metric] for r in reports["PrivBERT"]["per_seed"]}
        legal = {r["seed"]: r[metric] for r in reports["LegalBERT"]["per_seed"]}
        differences = [priv[s] - legal[s] for s in common["seeds"]]
        paired[metric] = dict(mean_difference=statistics.mean(differences), sample_std=statistics.stdev(differences),
                              differences_by_seed=dict(zip(map(str, common["seeds"]), differences)), n=len(differences))
    lines += ["", "All models used the same seed list and fixed policy split. Each seed starts from its model's original pinned pretrained checkpoint.",
              "Identical seed numbers control each run's random generators; they do not imply identical parameter tensors across architectures.",
              "The paired PrivBERT-minus-LegalBERT differences are saved in comparison.json.",
              "These are descriptive results, not a significance test or proof of model superiority. Variation is across fine-tuning seeds on one fixed test set."]
    write(root / "comparison.json", dict(status="complete", reports=reports, paired_privbert_minus_legalbert=paired, contract=common))
    (root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "metric", "mean", "std", "n", "ddof"])
        writer.writeheader(); writer.writerows(csv_rows)
    return root / "comparison.md"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True, help="Same shared output root passed to multiseed.py.")
    print(compare(parser.parse_args().output_root))
