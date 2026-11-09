import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import contextlib
import io
import multiseed

from multiseed import (SEEDS, METRICS, SCHEMA_SHA, recipe, validate_summary, reduce_metrics,
                       aggregate, fingerprint, sha, write, command)
from compare_models import compare, MODELS


def fixture(model="BERT"):
    spec = dict(mode="full", model=model, seeds=SEEDS, model_name=model, model_revision="pinned",
                model_type="bert", max_length=384, expected_steps=370, training=recipe("full"),
                data=dict(train_policy_ids=["train"], eval_policy_ids=["test"], schema_sha256=SCHEMA_SHA),
                pretraining=False, metric_contract="fixture", source_hashes={k: "same" for k in
                    ["multiseed.py", "evaluation.py", "opp115_hierarchy.py", "opp115_csv_loader.py", "opp115_policy_split.py"]})
    summary = dict(model_name=model, model_revision="pinned", model_type="bert", max_length=384,
                   seed=SEEDS[0], completed_optimizer_steps=370, training_configuration=recipe("full"),
                   train_policy_ids=["train"], eval_policy_ids=["test"], label_schema=dict(schema_sha256=SCHEMA_SHA),
                   segments=3792, train_segments=2352, eval_segments=1440, train_policies=65, eval_policies=50,
                   train_eval_policy_ids_disjoint=True, category_count=10, attribute_head_count=20,
                   fine_grained_label_count=138, training_examples_used=2352, evaluation_examples_used=1440,
                   training_policies_used=65, evaluation_policies_used=50, sampling_applied=False,
                   results=dict(evaluated_examples=1440, threshold=.5, primary_metric="hierarchical_macro_f1",
                                category=dict(macro_f1=.6), hierarchical_macro_f1=.5, fine_grained=dict(fine_grained_macro_f1=.4)))
    return spec, summary


def completed(folder, spec, summary):
    write(folder / "study_specification.json", spec)
    for i, seed in enumerate(spec["seeds"]):
        current = copy.deepcopy(summary)
        current["seed"] = seed
        current["results"]["category"]["macro_f1"] += i * .01
        path = folder / f"seed_{seed}"
        write(path / "summary.json", current)
        write(path / "complete.json", dict(seed=seed, specification_sha256=fingerprint(spec), summary_sha256=sha(path / "summary.json")))


class MultiSeedTests(unittest.TestCase):
    def test_sample_standard_deviation_not_population(self):
        rows = [dict(seed=s, **{k: x for k in METRICS}) for s, x in [(1001,.2),(1002,.4)]]
        values = reduce_metrics(rows)["category_macro_f1"]
        self.assertAlmostEqual(values["mean"], .3)
        self.assertAlmostEqual(values["std"], 2**.5 * .1)
        self.assertEqual(values["ddof"], 1)

    def test_duplicate_or_single_seed_rejected(self):
        row = dict(seed=1001, **{k: .5 for k in METRICS})
        for rows in [[row], [row,row]]:
            with self.assertRaises(ValueError): reduce_metrics(rows)

    def test_reject_incomplete_wrong_seed_revision_and_training(self):
        spec, summary = fixture()
        self.assertEqual(validate_summary(summary, spec, SEEDS[0])["category_macro_f1"], .6)
        for key, value in [("completed_optimizer_steps",1), ("seed",42), ("model_revision","wrong"),
                           ("sampling_applied",True), ("max_length",64)]:
            bad = copy.deepcopy(summary); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): validate_summary(bad,spec,SEEDS[0])
        bad = copy.deepcopy(summary); bad["training_configuration"]["epochs"] = 1
        with self.assertRaises(ValueError): validate_summary(bad,spec,SEEDS[0])

    def test_reject_non_finite_metric(self):
        spec, summary = fixture(); summary["results"]["hierarchical_macro_f1"] = float("nan")
        with self.assertRaises(ValueError): validate_summary(summary,spec,SEEDS[0])

    def test_missing_seed_prevents_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); spec,summary=fixture(); completed(root,spec,summary)
            (root / f"seed_{SEEDS[-1]}" / "complete.json").unlink()
            with self.assertRaises(FileNotFoundError): aggregate(root,spec)
            self.assertFalse((root / "completed_report.json").exists())

    def test_tampered_summary_prevents_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); spec,summary=fixture(); completed(root,spec,summary)
            write(root / f"seed_{SEEDS[0]}" / "summary.json", summary | {"seed":9999})
            with self.assertRaises(ValueError): aggregate(root,spec)

    def test_four_model_report_and_paired_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for model in MODELS:
                spec,summary=fixture(model); completed(root / "full" / model,spec,summary)
            result=compare(root)
            self.assertIn("sample standard deviation", result.read_text())
            output=json.loads((root / "full/comparison.json").read_text())
            self.assertEqual(output["paired_privbert_minus_legalbert"]["category_macro_f1"]["mean_difference"],0)
            self.assertAlmostEqual(output["reports"]["BERT"]["metrics"]["category_macro_f1"]["mean"],.62)

    def test_mismatched_seeds_and_smoke_cannot_compare(self):
        for mismatch in ["seeds", "mode"]:
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                for model in MODELS:
                    spec,summary=fixture(model)
                    if model=="LegalBERT": spec[mismatch] = [17012,29024] if mismatch=="seeds" else "smoke"
                    completed(root / "full" / model,spec,summary)
                with self.assertRaises(ValueError): compare(root)

    def test_commands_start_each_seed_from_same_pinned_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); write(root / "model_config.json",dict(backend="trainer"))
            spec,_=fixture()
            commands=[command(root,root / "data",spec,s,root / f"seed_{s}") for s in SEEDS]
            for seed,cmd in zip(SEEDS,commands):
                self.assertEqual(cmd[cmd.index("--seed")+1],str(seed))
                self.assertEqual(cmd[cmd.index("--model-name")+1],"BERT")
                self.assertEqual(cmd[cmd.index("--model-revision")+1],"pinned")
                self.assertNotIn("--no-save",cmd)

    def test_driver_executes_all_seeds_then_resumes_without_retraining(self):
        model=json.loads((Path(multiseed.__file__).parent / 'model_config.json').read_text())["model"]
        spec,summary=fixture(model)
        calls=[]
        def fake_training(cmd, **kwargs):
            seed=int(cmd[cmd.index('--seed')+1]); calls.append(seed)
            self.assertEqual(kwargs['env']['PYTHONHASHSEED'],str(seed))
            destination=Path(cmd[cmd.index('--summary-file')+1])
            write(destination,summary | dict(seed=seed))
            checkpoint=Path(cmd[cmd.index('--output-dir')+1])
            for name in ['classifier_config.json','hierarchy_schema.json','hierarchy_heads.safetensors','encoder/config.json','encoder/model.safetensors']:
                path=checkpoint/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text('test fixture')
        with tempfile.TemporaryDirectory() as tmp, patch.object(multiseed,'prepare',return_value=spec), \
             patch.dict('sys.modules',{'torch':SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True))}), \
             patch.object(multiseed.subprocess,'run',side_effect=fake_training), contextlib.redirect_stdout(io.StringIO()):
            multiseed.main(['--output-root',tmp])
            self.assertEqual(calls,SEEDS)
            multiseed.main(['--output-root',tmp])
            self.assertEqual(calls,SEEDS)
            report=json.loads((Path(tmp)/'full'/model/'completed_report.json').read_text())
            self.assertEqual(report['metrics']['category_macro_f1']['n'],5)

    def test_driver_rejects_shortened_training_and_does_not_mark_complete(self):
        model=json.loads((Path(multiseed.__file__).parent / 'model_config.json').read_text())["model"]
        spec,summary=fixture(model)
        def fake_training(cmd, **kwargs):
            destination=Path(cmd[cmd.index('--summary-file')+1])
            write(destination,summary | dict(completed_optimizer_steps=1))
        with tempfile.TemporaryDirectory() as tmp, patch.object(multiseed,'prepare',return_value=spec), \
             patch.dict('sys.modules',{'torch':SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True))}), \
             patch.object(multiseed.subprocess,'run',side_effect=fake_training), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):multiseed.main(['--output-root',tmp])
            folder=Path(tmp)/'full'/model
            self.assertFalse((folder/f'seed_{SEEDS[0]}'/'complete.json').exists())
            self.assertFalse((folder/'completed_report.json').exists())
            self.assertEqual(json.loads((folder/'status.json').read_text())['status'],'failed')


if __name__ == "__main__": unittest.main()
