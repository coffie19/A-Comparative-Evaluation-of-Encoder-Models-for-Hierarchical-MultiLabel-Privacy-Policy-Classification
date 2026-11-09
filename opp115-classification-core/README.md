# Matched PrivBERT, BERT, RoBERTa, and LegalBERT OPP-115 workspaces

This delivery contains four Google Colab workspaces for the same OPP-115
category and hierarchical fine-grained classification task. The supplied
PrivBERT workspace is preserved exactly. BERT, RoBERTa, and LegalBERT change
the encoder and tokenizer while keeping the data split, hierarchy, training
procedure, and evaluation calculations aligned with PrivBERT.

## Workspaces

| Folder | Full checkpoint | Colab notebook |
|---|---|---|
| `PrivBERT` | `mukund/privbert` | `PrivBERT/bert_opp115_colab.ipynb` |
| `BERT` | `google-bert/bert-base-uncased` | `BERT/bert_opp115_colab.ipynb` |
| `RoBERTa` | `FacebookAI/roberta-base` | `RoBERTa/roberta_opp115_colab.ipynb` |
| `LegalBERT` | `nlpaueb/legal-bert-base-uncased` | `LegalBERT/legal_bert_opp115_colab.ipynb` |

Each folder contains the notebook, a self-contained ZIP, its manifest and
SHA-256 sidecar, a README, and labeled smoke evidence. The LegalBERT folder
also includes its one-step pretraining smoke summary.

## Verification files

- `EVALUATION_CONTRACT.md` defines the common category and hierarchical metrics.
- `VALIDATION_REPORT.md` and `VALIDATION_REPORT.json` contain the independent
  four-model evaluator parity audit.
- `DELIVERY_AUDIT.json` verifies archive integrity, payload hashes, notebook
  consistency, checkpoint identity, evaluation settings, and smoke evidence.
- `LEGALBERT_ROUNDTRIP_REPORT.json` verifies that a saved LegalBERT hierarchy
  classifier reloads and produces 10 category logits and 138 attribute logits.
- `SOURCE_PROVENANCE.json` records the supplied source and pinned checkpoints.
- `DELIVERY_SHA256SUMS.txt` provides hashes for every delivered file except
  the checksum list itself.

## Running the experiments

Open one notebook in Google Colab, select a GPU runtime, and run all cells with
`MODE = "full"`. Full mode is the default. `RUN_PRETRAINING = False` is kept
for the matched comparison. If you enable optional domain-adaptive masked
language-model training, report that as a separate experimental condition.

The saved smoke summaries use tiny checkpoints, eight training examples, 32
evaluation examples, and one optimizer step. They verify the pipeline only and
must not be reported as thesis results. No full-model result is pre-filled.
Report metrics only after each full run processes all 2,352 training segments
and all 1,440 held-out segments from the fixed 65/50 policy split.

## Comparison rule

Compare full result JSON files only when their manifests show the same hierarchy
schema hash, policy IDs, threshold, training settings, and evaluation coverage.
The validation report confirms that all four packages implement the same
evaluator before full training.
