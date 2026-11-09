#!/usr/bin/env python
"""Optionally continue BERT masked-language-model training on OPP-115.

This is a local and Colab-runnable OPP-115 workflow. OPP-115 provides
privacy-policy text, so continued masked-language-
model (MLM) pretraining is the appropriate unsupervised objective; its CSV
annotations are not used by this stage.

The loader recursively finds directories named ``sanitized_policies`` and
reads only their direct HTML/text files. This avoids training on duplicate
representations, annotation CSVs, JavaScript, CSS, images, and archived pages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import re
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Sequence

from opp115_policy_split import (
    DEFAULT_EVAL_POLICIES,
    DEFAULT_TRAIN_POLICIES,
    split_policy_ids,
)


LOGGER = logging.getLogger("bert_opp115_pretraining")

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OPP115_DIR = PROJECT_ROOT / "data" / "opp115"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "bert-opp115-dapt"
DEFAULT_MODEL_NAME = "google-bert/bert-base-uncased"
EXPECTED_MODEL_TYPE = "bert"
SUPPORTED_POLICY_SUFFIXES = {".html", ".htm", ".txt"}


@dataclass(frozen=True)
class PolicyDocument:
    """One clean policy document and its source metadata."""

    path: Path
    text: str
    source_characters: int
    segments: int


class _PolicyHTMLTextExtractor(HTMLParser):
    """Dependency-free HTML-to-text extractor for sanitized policies."""

    _BLOCK_TAGS = {
        "br",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "p",
        "section",
        "tr",
    }
    _SKIPPED_TAGS = {"script", "style"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        del attrs
        tag = tag.casefold()
        if tag in self._SKIPPED_TAGS:
            self._skip_depth += 1
        elif self._skip_depth == 0 and tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag in self._SKIPPED_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif self._skip_depth == 0 and tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self.parts.append(data)


def clean_opp115_policy(raw_text: str) -> str:
    """Convert OPP-115's minimal HTML and ``|||`` separators to plain text."""

    extractor = _PolicyHTMLTextExtractor()
    extractor.feed(
        raw_text.replace("\ufeff", "").replace("\x00", "").replace("|||", "\n\n")
    )
    extractor.close()

    text = "".join(extractor.parts).replace("\xa0", " ")
    paragraphs: list[str] = []
    for paragraph in re.split(r"\n+", text):
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if paragraph:
            paragraphs.append(paragraph)
    return "\n\n".join(paragraphs)


def _find_sanitized_policy_dirs(data_dir: Path) -> list[Path]:
    """Find one or more nested OPP-115 ``sanitized_policies`` directories."""

    if data_dir.name.casefold() == "sanitized_policies":
        return [data_dir]

    directories = [
        path
        for path in data_dir.rglob("*")
        if path.is_dir() and path.name.casefold() == "sanitized_policies"
    ]
    canonical_directories = {
        str(path.resolve()).casefold(): path.resolve() for path in directories
    }
    return sorted(canonical_directories.values(), key=lambda path: str(path).casefold())


def discover_policy_files(
    data_dir: Path, *, allow_generic_text_tree: bool = False
) -> list[Path]:
    """Discover clean policy files without ingesting the noisy full OPP tree.

    Multiple nested datasets are supported: every directory named
    ``sanitized_policies`` below ``data_dir`` is included.  A generic recursive
    text/HTML fallback must be enabled explicitly.
    """

    data_dir = data_dir.expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"OPP-115 data path does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"OPP-115 data path is not a directory: {data_dir}")

    sanitized_dirs = _find_sanitized_policy_dirs(data_dir)
    if sanitized_dirs:
        files = [
            path
            for directory in sanitized_dirs
            for path in directory.iterdir()
            if path.is_file() and path.suffix.casefold() in SUPPORTED_POLICY_SUFFIXES
        ]
    elif allow_generic_text_tree:
        LOGGER.warning(
            "No sanitized_policies directory found; recursively reading the "
            "explicitly allowed generic text tree at %s",
            data_dir,
        )
        files = [
            path
            for path in data_dir.rglob("*")
            if path.is_file() and path.suffix.casefold() in SUPPORTED_POLICY_SUFFIXES
        ]
    else:
        raise FileNotFoundError(
            "No directory named 'sanitized_policies' was found below "
            f"{data_dir}. Point --data-dir at OPP-115 (or its "
            "sanitized_policies folder). For another already-clean text tree, "
            "pass --allow-generic-text-tree."
        )

    canonical_files = {
        str(path.resolve()).casefold(): path.resolve() for path in files
    }
    files = sorted(canonical_files.values(), key=lambda path: str(path).casefold())
    if not files:
        raise FileNotFoundError(f"No supported policy text files found below {data_dir}")
    return files


def load_policy_documents(
    data_dir: Path,
    *,
    allow_generic_text_tree: bool = False,
    max_documents: int = 0,
    seed: int = 17011,
) -> list[PolicyDocument]:
    """Read and clean policy documents using strict UTF-8 decoding."""

    paths = discover_policy_files(
        data_dir, allow_generic_text_tree=allow_generic_text_tree
    )
    if max_documents < 0:
        raise ValueError("max_documents must be zero (all documents) or positive")

    documents: list[PolicyDocument] = []
    seen_content: dict[str, Path] = {}
    for path in paths:
        raw_text = path.read_text(encoding="utf-8", errors="strict")
        text = clean_opp115_policy(raw_text)
        if not text:
            LOGGER.warning("Skipping empty policy after cleaning: %s", path)
            continue
        content_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if content_digest in seen_content:
            LOGGER.warning(
                "Skipping duplicate policy content: %s (same as %s)",
                path,
                seen_content[content_digest],
            )
            continue
        seen_content[content_digest] = path
        documents.append(
            PolicyDocument(
                path=path.resolve(),
                text=text,
                source_characters=len(raw_text),
                segments=raw_text.count("|||") + 1,
            )
        )

    if max_documents and max_documents < len(documents):
        documents = sorted(
            random.Random(seed).sample(documents, max_documents),
            key=lambda document: str(document.path).casefold(),
        )

    if len(documents) < 2:
        raise ValueError(
            "At least two non-empty policy documents are required; found "
            f"{len(documents)}"
        )
    return documents


def split_documents(
    documents: Sequence[PolicyDocument],
    *,
    train_count: int = DEFAULT_TRAIN_POLICIES,
    eval_count: int = DEFAULT_EVAL_POLICIES,
) -> tuple[list[PolicyDocument], list[PolicyDocument]]:
    """Make the shared exact 65/50 whole-policy split."""

    by_policy_id: dict[str, PolicyDocument] = {}
    for document in documents:
        policy_id = document.path.stem
        if policy_id in by_policy_id:
            raise ValueError(f"Duplicate policy identifier: {policy_id}")
        by_policy_id[policy_id] = document

    train_ids, eval_ids = split_policy_ids(
        by_policy_id, train_count=train_count, eval_count=eval_count
    )
    train_documents = [by_policy_id[policy_id] for policy_id in train_ids]
    eval_documents = [by_policy_id[policy_id] for policy_id in eval_ids]

    train_paths = {document.path for document in train_documents}
    eval_paths = {document.path for document in eval_documents}
    if train_paths & eval_paths:
        raise AssertionError("A policy appeared in both train and evaluation splits")
    return train_documents, eval_documents


class TokenChunkDataset:
    """Minimal indexable dataset of fixed-size token blocks."""

    def __init__(self, examples: list[dict[str, list[int]]]) -> None:
        if not examples:
            raise ValueError("Tokenization produced no examples")
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


class GuaranteedMaskingCollator:
    """Retry random MLM masking and guarantee at least one supervised token."""

    def __init__(
        self, base_collator: Any, tokenizer: Any, max_attempts: int = 10
    ) -> None:
        self.base_collator = base_collator
        self.tokenizer = tokenizer
        self.max_attempts = max_attempts

    def __call__(
        self, examples: list[dict[str, list[int]]]
    ) -> dict[str, Any]:
        for _ in range(self.max_attempts):
            batch = self.base_collator(examples)
            if bool(batch["labels"].ne(-100).any()):
                return batch

        for example_index, example in enumerate(examples):
            for token_index, (is_attended, is_special) in enumerate(
                zip(example["attention_mask"], example["special_tokens_mask"])
            ):
                if is_attended and not is_special:
                    if self.tokenizer.mask_token_id is None:
                        raise ValueError("The selected tokenizer has no mask token")
                    batch["labels"][example_index, token_index] = example["input_ids"][
                        token_index
                    ]
                    batch["input_ids"][example_index, token_index] = (
                        self.tokenizer.mask_token_id
                    )
                    return batch
        raise RuntimeError("No non-special token is available for MLM masking")


def build_token_chunks(
    documents: Iterable[PolicyDocument], tokenizer: Any, block_size: int
) -> TokenChunkDataset:
    """Tokenize each policy separately and split it into padded MLM blocks."""

    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    content_block_size = block_size - special_tokens
    if content_block_size < 8:
        raise ValueError(
            f"block_size={block_size} leaves too little room after special tokens"
        )

    examples: list[dict[str, list[int]]] = []
    for document in documents:
        token_ids = tokenizer(
            document.text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            verbose=False,
        )["input_ids"]
        for start in range(0, len(token_ids), content_block_size):
            content_ids = token_ids[start : start + content_block_size]
            if not content_ids:
                continue
            input_ids = tokenizer.build_inputs_with_special_tokens(content_ids)
            if len(input_ids) > block_size:
                raise AssertionError("Tokenizer added more special tokens than reported")
            special_tokens_mask = tokenizer.get_special_tokens_mask(
                input_ids, already_has_special_tokens=True
            )
            attention_mask = [1] * len(input_ids)
            padding_length = block_size - len(input_ids)
            if padding_length:
                if tokenizer.pad_token_id is None:
                    raise ValueError("The selected tokenizer has no padding token")
                input_ids.extend([tokenizer.pad_token_id] * padding_length)
                attention_mask.extend([0] * padding_length)
                special_tokens_mask.extend([1] * padding_length)
            examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "special_tokens_mask": special_tokens_mask,
                }
            )
    return TokenChunkDataset(examples)


def corpus_summary(
    documents: Sequence[PolicyDocument],
    train_documents: Sequence[PolicyDocument],
    eval_documents: Sequence[PolicyDocument],
) -> dict[str, Any]:
    """Return JSON-serializable corpus validation information."""

    return {
        "corpus": "OPP-115",
        "documents": len(documents),
        "train_documents": len(train_documents),
        "eval_documents": len(eval_documents),
        "source_characters": sum(document.source_characters for document in documents),
        "clean_characters": sum(len(document.text) for document in documents),
        "segments": sum(document.segments for document in documents),
        "train_eval_paths_disjoint": not (
            {document.path for document in train_documents}
            & {document.path for document in eval_documents}
        ),
        "source_directories": sorted(
            {str(document.path.parent) for document in documents}
        ),
        "train_policy_ids": [document.path.stem for document in train_documents],
        "eval_policy_ids": [document.path.stem for document in eval_documents],
    }


def write_json_summary(summary: dict[str, Any], summary_file: Path) -> None:
    """Write a durable run/data summary without requiring a model checkpoint save."""

    summary_file = summary_file.expanduser().resolve()
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    LOGGER.info("Saved run summary to %s", summary_file)


def _choose_device(torch_module: Any, requested: str) -> Any:
    if requested == "auto":
        requested = "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA GPU")
    return torch_module.device(requested)


def _evaluate(
    model: Any,
    data_loader: Any,
    device: Any,
    torch_module: Any,
    max_batches: int,
) -> float:
    model.eval()
    weighted_loss_sum = 0.0
    masked_token_count = 0
    with torch_module.no_grad():
        for batch_index, batch in enumerate(data_loader):
            if max_batches and batch_index >= max_batches:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            batch_masked_tokens = int(batch["labels"].ne(-100).sum().item())
            if batch_masked_tokens == 0:
                raise RuntimeError("Evaluation batch contains no masked tokens")
            loss = model(**batch).loss
            if not torch_module.isfinite(loss):
                raise RuntimeError("Evaluation produced a non-finite MLM loss")
            weighted_loss_sum += float(loss.detach().cpu()) * batch_masked_tokens
            masked_token_count += batch_masked_tokens
    if not masked_token_count:
        raise RuntimeError("Evaluation loader produced no batches")
    return weighted_loss_sum / masked_token_count


def validate_bert_model_type(model: Any) -> None:
    """Reject checkpoints that are not backed by BERT's BERT encoder."""

    actual_model_type = getattr(model.config, "model_type", None)
    if actual_model_type != EXPECTED_MODEL_TYPE:
        raise ValueError(
            "Expected BERT with model_type='bert', but the loaded "
            f"checkpoint reports model_type={actual_model_type!r}."
        )


def run_pretraining(
    args: argparse.Namespace,
    documents: Sequence[PolicyDocument],
    train_documents: Sequence[PolicyDocument],
    eval_documents: Sequence[PolicyDocument],
) -> dict[str, Any]:
    """Load BERT and perform optional continued MLM pretraining."""

    if not args.no_save:
        output_dir = args.output_dir.expanduser().resolve()
        if (
            output_dir.exists()
            and any(output_dir.iterdir())
            and not args.overwrite_output_dir
        ):
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Choose another "
                "--output-dir or pass --overwrite-output-dir."
            )

    try:
        import torch
        import transformers
        from torch.utils.data import DataLoader
        from transformers import (
            AutoModelForMaskedLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            get_linear_schedule_with_warmup,
        )
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "BERT dependencies could not be loaded. Create a compatible Python "
            "environment and install requirements_bert_opp115.txt."
        ) from exc

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    LOGGER.info("Loading pretrained checkpoint: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, revision=args.model_revision, use_fast=True
    )
    model = AutoModelForMaskedLM.from_pretrained(
        args.model_name, revision=args.model_revision
    )
    validate_bert_model_type(model)

    tokenizer_limit = getattr(tokenizer, "model_max_length", args.block_size)
    if tokenizer_limit < 1_000_000 and args.block_size > tokenizer_limit:
        raise ValueError(
            f"block_size={args.block_size} exceeds tokenizer limit {tokenizer_limit}"
        )

    train_dataset = build_token_chunks(train_documents, tokenizer, args.block_size)
    eval_dataset = (
        None
        if args.skip_evaluation
        else build_token_chunks(eval_documents, tokenizer, args.block_size)
    )
    LOGGER.info(
        "Prepared %d train blocks and %d evaluation blocks (length %d)",
        len(train_dataset),
        len(eval_dataset) if eval_dataset is not None else 0,
        args.block_size,
    )

    collator = GuaranteedMaskingCollator(
        DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=args.mlm_probability,
        ),
        tokenizer,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    fixed_eval_batches: list[dict[str, Any]] = []
    if eval_dataset is not None:
        random_state = torch.random.get_rng_state()
        torch.manual_seed(args.seed + 1)
        try:
            for start in range(0, len(eval_dataset), args.batch_size):
                if (
                    args.max_eval_batches
                    and len(fixed_eval_batches) >= args.max_eval_batches
                ):
                    break
                examples = [
                    eval_dataset[index]
                    for index in range(
                        start,
                        min(start + args.batch_size, len(eval_dataset)),
                    )
                ]
                fixed_eval_batches.append(collator(examples))
        finally:
            torch.random.set_rng_state(random_state)

    sample_count = min(args.batch_size, len(train_dataset))
    masked_batch = collator(
        [train_dataset[index] for index in range(sample_count)]
    )
    masked_positions = masked_batch["labels"].ne(-100)
    if not bool(masked_positions.any()):
        raise RuntimeError("The MLM data collator did not mask any tokens")
    if bool(masked_positions[masked_batch["attention_mask"].eq(0)].any()):
        raise RuntimeError("Padding tokens were selected as MLM prediction targets")

    device = _choose_device(torch, args.device)
    model.to(device)
    LOGGER.info("Training device: %s", device)

    decay_parameters: list[Any] = []
    no_decay_parameters: list[Any] = []
    for parameter_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter_name.endswith("bias") or parameter_name.endswith(
            ("LayerNorm.weight", "layer_norm.weight")
        ):
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_parameters, "weight_decay": args.weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
    )
    updates_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )
    total_steps = args.max_steps or (args.epochs * updates_per_epoch)
    epochs_to_run = (
        math.ceil(total_steps / updates_per_epoch) if args.max_steps else args.epochs
    )
    warmup_steps = round(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    initial_eval_loss: float | None = None
    if fixed_eval_batches:
        initial_eval_loss = _evaluate(
            model, fixed_eval_batches, device, torch, max_batches=0
        )
        LOGGER.info("Initial evaluation MLM loss: %.6f", initial_eval_loss)
    else:
        LOGGER.info("Held-out MLM evaluation skipped; training uses 65 policies only")

    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    observed_train_losses: list[float] = []
    for epoch_index in range(epochs_to_run):
        model.train()
        accumulated_batches = 0
        accumulated_loss = 0.0
        for batch_index, batch in enumerate(train_loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Training produced a non-finite MLM loss")
            loss.backward()
            accumulated_batches += 1
            accumulated_loss += float(loss.detach().cpu())

            update_now = (
                accumulated_batches == args.gradient_accumulation_steps
                or batch_index == len(train_loader) - 1
            )
            if not update_now:
                continue

            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(accumulated_batches)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            step_loss = accumulated_loss / accumulated_batches
            observed_train_losses.append(step_loss)
            accumulated_batches = 0
            accumulated_loss = 0.0

            if global_step == 1 or global_step % args.log_every_steps == 0:
                LOGGER.info(
                    "epoch=%d step=%d/%d train_mlm_loss=%.6f lr=%.3g",
                    epoch_index + 1,
                    global_step,
                    total_steps,
                    step_loss,
                    scheduler.get_last_lr()[0],
                )
            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    final_eval_loss: float | None = None
    final_perplexity: float | None = None
    if fixed_eval_batches:
        final_eval_loss = _evaluate(
            model, fixed_eval_batches, device, torch, max_batches=0
        )
        final_perplexity = math.exp(final_eval_loss)
        LOGGER.info(
            "Final evaluation MLM loss: %.6f (perplexity %.3f)",
            final_eval_loss,
            final_perplexity,
        )

    summary: dict[str, Any] = {
        **corpus_summary(documents, train_documents, eval_documents),
        "model_name": args.model_name,
        "model_revision": args.model_revision,
        "model_type": EXPECTED_MODEL_TYPE,
        "device": str(device),
        "seed": args.seed,
        "library_versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "training_configuration": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "mlm_probability": args.mlm_probability,
            "skip_evaluation": args.skip_evaluation,
        },
        "block_size": args.block_size,
        "train_blocks": len(train_dataset),
        "eval_blocks": len(eval_dataset) if eval_dataset is not None else 0,
        "evaluated_eval_batches": len(fixed_eval_batches),
        "evaluated_eval_blocks": sum(
            int(batch["input_ids"].shape[0]) for batch in fixed_eval_batches
        ),
        "evaluated_masked_tokens": sum(
            int(batch["labels"].ne(-100).sum().item())
            for batch in fixed_eval_batches
        ),
        "masked_tokens_in_training_check_batch": int(masked_positions.sum().item()),
        "completed_optimizer_steps": global_step,
        "initial_eval_loss": initial_eval_loss,
        "mean_train_loss": sum(observed_train_losses) / len(observed_train_losses),
        "final_eval_loss": final_eval_loss,
        "final_perplexity": final_perplexity,
    }

    if not args.no_save:
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        (output_dir / "bert_opp115_pretraining_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        LOGGER.info("Saved adapted model and tokenizer to %s", output_dir)

    return summary


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Optionally continue BERT masked-language-model training on local OPP-115 "
            "policies."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_OPP115_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--summary-file",
        type=Path,
        help="Optional JSON summary path, including when --no-save is used.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--model-revision",
        help="Optional Hugging Face commit/tag revision; recorded in the summary.",
    )
    parser.add_argument("--block-size", type=positive_int, default=384)
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument("--epochs", type=positive_int, default=2)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-documents", type=int, default=0)
    parser.add_argument(
        "--train-policies", type=positive_int, default=DEFAULT_TRAIN_POLICIES
    )
    parser.add_argument(
        "--eval-policies", type=positive_int, default=DEFAULT_EVAL_POLICIES
    )
    parser.add_argument("--mlm-probability", type=probability, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation-steps", type=positive_int, default=2)
    parser.add_argument("--seed", type=int, default=17011)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--log-every-steps", type=positive_int, default=10)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Do not inspect the 50 held-out policies during MLM pretraining.",
    )
    parser.add_argument(
        "--allow-generic-text-tree",
        action="store_true",
        help="Allow recursive HTML/TXT loading when no sanitized_policies folder exists.",
    )
    parser.add_argument(
        "--validate-data-only",
        action="store_true",
        help="Validate loading and the document split without importing PyTorch.",
    )
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--overwrite-output-dir", action="store_true")
    return parser


def validate_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.max_steps < 0:
        parser.error("--max-steps must be zero (epoch-based training) or positive")
    if args.max_documents < 0:
        parser.error("--max-documents must be zero (all documents) or positive")
    if args.max_eval_batches < 0:
        parser.error("--max-eval-batches must be zero (all batches) or positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.weight_decay < 0:
        parser.error("--weight-decay cannot be negative")
    if not 0.0 <= args.warmup_ratio < 1.0:
        parser.error("--warmup-ratio must be in [0, 1)")
    if args.max_grad_norm <= 0:
        parser.error("--max-grad-norm must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    validate_arguments(parser, args)

    documents = load_policy_documents(
        args.data_dir,
        allow_generic_text_tree=args.allow_generic_text_tree,
        max_documents=args.max_documents,
        seed=args.seed,
    )
    train_documents, eval_documents = split_documents(
        documents,
        train_count=args.train_policies,
        eval_count=args.eval_policies,
    )
    summary = corpus_summary(documents, train_documents, eval_documents)
    LOGGER.info(
        "Loaded OPP-115: %d policies, %d segments, %d train / %d eval policies",
        summary["documents"],
        summary["segments"],
        summary["train_documents"],
        summary["eval_documents"],
    )

    if args.validate_data_only:
        if args.summary_file:
            write_json_summary(summary, args.summary_file)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    training_summary = run_pretraining(
        args, documents, train_documents, eval_documents
    )
    if args.summary_file:
        write_json_summary(training_summary, args.summary_file)
    print(json.dumps(training_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
