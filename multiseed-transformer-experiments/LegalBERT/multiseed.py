"""Matched OPP-115 fine-tuning runs and strictly validated descriptive reports."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

SEEDS = [17011, 29023, 41047, 53069, 65089]
SCHEMA_SHA = "90627a25efe47a217e03a2f09431b3256bc254867cd8f9aee675a1eafbb93b9b"
METRICS = {
    "category_macro_f1": ("category", "macro_f1"),
    "hierarchical_macro_f1": ("hierarchical_macro_f1",),
    "gated_fine_grained_macro_f1": ("fine_grained", "fine_grained_macro_f1"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def recipe(mode):
    smoke = mode == "smoke"
    return dict(batch_size=4 if smoke else 8, epochs=5, max_steps=1 if smoke else 0,
                gradient_accumulation_steps=4, learning_rate=2.5e-5, weight_decay=.01,
                warmup_ratio=.1, threshold=.5, classifier_dropout=.15,
                attribute_loss_weight=1., positive_weight_power=.5, max_positive_weight=10.,
                max_train_examples=8 if smoke else 0, max_eval_examples=32 if smoke else 0)


def prepare(root, data, config, mode, seeds):
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(s < 1000 or s >= 2**32 for s in seeds):
        raise ValueError("Use at least two distinct seeds in [1000, 2**32), shared across all models.")
    backend = importlib.import_module(config["backend"])
    from opp115_hierarchy import fixed_policy_split, derive_hierarchy_schema
    train_ids, eval_ids = fixed_policy_split(data / "annotations")
    schema = derive_hierarchy_schema(data / "annotations", train_ids)
    examples = backend.load_segment_examples(data, schema)
    train, evaluation, actual_train, actual_eval = backend.split_segment_examples(examples)
    if (train_ids, eval_ids) != (actual_train, actual_eval):
        raise ValueError("Inconsistent policy split.")
    if (len(train), len(evaluation), schema.schema_sha256) != (2352, 1440, SCHEMA_SHA):
        raise ValueError("Dataset differs from the matched study.")
    def rows(items):
        return [dict(policy_id=x.policy_id, segment_id=x.segment_id, text_sha256=hashlib.sha256(x.text.encode()).hexdigest(),
                     categories=list(x.category_labels), attributes=list(x.attribute_labels), masks=list(x.attribute_mask)) for x in items]
    source_names = [config["backend"] + ".py", "multiseed.py", "model_config.json", "evaluation.py",
                    "opp115_hierarchy.py", "opp115_csv_loader.py", "opp115_policy_split.py", "config.py"]
    checkpoint = config["smoke"] if mode == "smoke" else config["full"]
    return dict(format_version=1, mode=mode, model=config["model"], seeds=seeds,
                model_name=checkpoint["name"], model_revision=checkpoint["revision"], model_type=config["model_type"],
                source_hashes={n: sha(root / n) for n in source_names},
                data=dict(train_policy_ids=list(train_ids), eval_policy_ids=list(eval_ids),
                          train_instances_sha256=fingerprint(rows(train)), eval_instances_sha256=fingerprint(rows(evaluation)),
                          schema_sha256=schema.schema_sha256),
                training=recipe(mode), max_length=64 if mode == "smoke" else 384,
                expected_steps=1 if mode == "smoke" else 370,
                pretraining=False, metric_contract="gold-applicable predicted-parent-gated macro-F1; zero_division=0")


def validate_summary(summary, specification, seed):
    def require(actual, expected, field):
        if actual != expected:
            raise ValueError(f"{field}: expected {expected!r}, got {actual!r}")
    for field in ("model_name", "model_revision", "model_type", "max_length"):
        require(summary.get(field), specification[field], field)
    require(summary.get("seed"), seed, "seed")
    require(summary.get("completed_optimizer_steps"), specification["expected_steps"], "completed_optimizer_steps")
    for field, value in specification["training"].items():
        require(summary.get("training_configuration", {}).get(field), value, f"training_configuration.{field}")
    for field in ("train_policy_ids", "eval_policy_ids"):
        require(summary.get(field), specification["data"][field], field)
    require(summary.get("label_schema", {}).get("schema_sha256"), specification["data"]["schema_sha256"], "schema")
    smoke = specification["mode"] == "smoke"
    for field, value in dict(segments=3792, train_segments=2352, eval_segments=1440,
                             train_policies=65, eval_policies=50, train_eval_policy_ids_disjoint=True,
                             category_count=10, attribute_head_count=20, fine_grained_label_count=138,
                             training_examples_used=8 if smoke else 2352,
                             evaluation_examples_used=32 if smoke else 1440, sampling_applied=smoke).items():
        require(summary.get(field), value, field)
    if not smoke:
        require(summary.get("training_policies_used"), 65, "training_policies_used")
        require(summary.get("evaluation_policies_used"), 50, "evaluation_policies_used")
    require(summary.get("results", {}).get("evaluated_examples"), 32 if smoke else 1440, "evaluated_examples")
    require(summary.get("results", {}).get("threshold"), .5, "results.threshold")
    require(summary.get("results", {}).get("primary_metric"), "hierarchical_macro_f1", "primary_metric")
    values = {}
    for name, keys in METRICS.items():
        value = summary["results"]
        for key in keys:
            value = value[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid {name}: {value!r}")
        values[name] = float(value)
    return values


def reduce_metrics(rows):
    if len(rows) < 2 or len({r["seed"] for r in rows}) != len(rows):
        raise ValueError("At least two distinct completed seeds are required.")
    return {name: dict(mean=statistics.mean(r[name] for r in rows),
                       std=statistics.stdev(r[name] for r in rows), n=len(rows), ddof=1)
            for name in METRICS}


def load_seed(run_root, spec, seed):
    folder = run_root / f"seed_{seed}"
    receipt = read(folder / "complete.json")
    if receipt["specification_sha256"] != fingerprint(spec) or receipt["seed"] != seed:
        raise ValueError(f"Seed {seed} belongs to a different study.")
    summary_path = folder / "summary.json"
    if receipt["summary_sha256"] != sha(summary_path):
        raise ValueError(f"Seed {seed} summary changed after completion.")
    return dict(seed=seed, **validate_summary(read(summary_path), spec, seed))


def aggregate(run_root, spec):
    rows = [load_seed(run_root, spec, seed) for seed in spec["seeds"]]
    result = dict(status="complete", reportable=spec["mode"] == "full", mode=spec["mode"],
                  model=spec["model"], seeds=spec["seeds"], specification_sha256=fingerprint(spec),
                  standard_deviation="sample (ddof=1), across independent fine-tuning runs",
                  metrics=reduce_metrics(rows), per_seed=rows)
    write(run_root / "completed_report.json", result)
    with (run_root / "per_seed.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["seed", *METRICS]); writer.writeheader(); writer.writerows(rows)
    lines = [f"# {spec['model']} — {'FULL STUDY' if result['reportable'] else 'SMOKE CHECK ONLY — NOT RESEARCH RESULTS'}",
             "", f"Seeds: {', '.join(map(str, spec['seeds']))}", "",
             "| Metric | Mean ± sample standard deviation | n |", "|---|---:|---:|"]
    for name, value in result["metrics"].items():
        lines.append(f"| {name} | {value['mean']:.6f} ± {value['std']:.6f} | {value['n']} |")
    lines += ["", "Scores are on the 0–1 scale. Standard deviation uses ddof=1; it is not a confidence interval.",
              "The policy split is fixed. Category and hierarchical Macro-F1 are different metrics.",
              "No claim of model superiority is established by this descriptive report."]
    (run_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def command(root, data, spec, seed, folder):
    config = read(root / "model_config.json")
    cmd = [sys.executable, str(root / (config["backend"] + ".py")), "--data-dir", str(data),
           "--model-name", spec["model_name"], "--model-revision", spec["model_revision"],
           "--seed", str(seed), "--max-length", str(spec["max_length"]),
           "--summary-file", str(folder / "summary.json"), "--device", "cuda" if spec["mode"] == "full" else "cpu"]
    for key, value in spec["training"].items():
        cmd += ["--" + key.replace("_", "-"), str(value)]
    if spec["mode"] == "full":
        cmd += ["--output-dir", str(folder / "checkpoint"), "--overwrite-output-dir"]
    else:
        cmd += ["--no-save"]
    return cmd


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["full", "smoke"], default="full")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-root", type=Path, required=True, help="Shared study directory; model/mode subfolders are automatic.")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    data = (args.data_dir or root / "data/opp115").resolve()
    config = read(root / "model_config.json")
    spec = prepare(root, data, config, args.mode, args.seeds)
    run_root = args.output_root.expanduser().resolve() / args.mode / config["model"]
    run_root.mkdir(parents=True, exist_ok=True)
    spec_path = run_root / "study_specification.json"
    if spec_path.exists() and read(spec_path) != spec:
        raise ValueError("Existing study settings/data/source differ. Choose a new output root.")
    write(spec_path, spec)
    if args.prepare_only:
        print("PREPARED; no training performed:", spec_path)
        return 0
    if args.aggregate_only:
        aggregate(run_root, spec)
        print(run_root / "report.md")
        return 0
    import torch
    if args.mode == "full" and not torch.cuda.is_available():
        raise RuntimeError("Full runs require CUDA. Select a Colab GPU runtime; no results have been fabricated.")
    for seed in spec["seeds"]:
        folder = run_root / f"seed_{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "complete.json").exists():
            load_seed(run_root, spec, seed)
            print(f"Verified completed seed {seed}; reusing.", flush=True)
            continue
        cmd = command(root, data, spec, seed, folder)
        write(folder / "invocation.json", dict(command=cmd, seed=seed, specification_sha256=fingerprint(spec)))
        write(run_root / "status.json", dict(status="running", seed=seed, reportable=False))
        env = dict(os.environ, PYTHONHASHSEED=str(seed), CUBLAS_WORKSPACE_CONFIG=":4096:8", TOKENIZERS_PARALLELISM="false")
        print(f"Starting {config['model']} seed {seed}; log: {folder / 'training.log'}", flush=True)
        try:
            with (folder / "training.log").open("w", encoding="utf-8") as log:
                subprocess.run(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            validate_summary(read(folder / "summary.json"), spec, seed)
            if args.mode == "full":
                checkpoint = folder / "checkpoint"
                for relative in ["classifier_config.json", "hierarchy_schema.json", "hierarchy_heads.safetensors", "encoder/config.json", "encoder/model.safetensors"]:
                    if not (checkpoint / relative).is_file():
                        raise ValueError(f"Missing completed checkpoint file: {relative}")
            write(folder / "complete.json", dict(seed=seed, specification_sha256=fingerprint(spec), summary_sha256=sha(folder / "summary.json")))
        except Exception as error:
            write(run_root / "status.json", dict(status="failed", seed=seed, error=str(error), reportable=False))
            raise
    aggregate(run_root, spec)
    write(run_root / "status.json", dict(status="complete", reportable=args.mode == "full", completed_seeds=spec["seeds"]))
    print("Completed:", run_root / "report.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
