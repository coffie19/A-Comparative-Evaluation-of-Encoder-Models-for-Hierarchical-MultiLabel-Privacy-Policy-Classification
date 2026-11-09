"""
ml_layer_embeddings.py
======================
Section 4.1 – Privacy-Specific Word Embeddings
  • Train a fastText Skip-gram model with subword embeddings on the
    130K-policies corpus (Policies Embeddings).
  • Provide helper to load either the trained model or the fastText
    Wikipedia vectors as ablation variants.

Section 4.2 – Classification Dataset  (OPP-115 loader)
  • Parse the OPP-115 annotation CSV/JSON files into a structured
    dataset suitable for training the hierarchical classifiers.
  • Apply the 65/50 train/test split used in the paper.
"""

import os
import json
import logging
import glob
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    FASTTEXT_DIM, FASTTEXT_MIN_COUNT, FASTTEXT_WINDOW,
    FASTTEXT_EPOCHS, FASTTEXT_MIN_N, FASTTEXT_MAX_N,
    FASTTEXT_MODEL, EMBEDDINGS_PATH,
    OPP115_DIR, POLICIES_DIR,
    OPP115_TRAIN_POLICIES, OPP115_TEST_POLICIES,
    HIGH_LEVEL_CATEGORIES, ATTRIBUTES, CATEGORY_ATTRIBUTES,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 4.1  Privacy-Specific Word Embeddings
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingTrainer:
    """
    Trains a fastText model (Skip-gram + subword n-grams) on the privacy-
    policy corpus, replicating the "Policies Embeddings" (PE) described
    in Section 4.1 of the paper.
    """

    def __init__(
        self,
        corpus_dir: str  = POLICIES_DIR,
        output_path: str = EMBEDDINGS_PATH,
        dim: int         = FASTTEXT_DIM,
        min_count: int   = FASTTEXT_MIN_COUNT,
        window: int      = FASTTEXT_WINDOW,
        epochs: int      = FASTTEXT_EPOCHS,
        min_n: int       = FASTTEXT_MIN_N,
        max_n: int       = FASTTEXT_MAX_N,
        model: str       = FASTTEXT_MODEL,
    ):
        self.corpus_dir  = corpus_dir
        self.output_path = output_path
        self.dim         = dim
        self.min_count   = min_count
        self.window      = window
        self.epochs      = epochs
        self.min_n       = min_n
        self.max_n       = max_n
        self.model_type  = model

    # ── Build corpus text file ───────────────────────────────────────────────

    def build_corpus_file(self, output_txt: str = "corpus.txt") -> str:
        """
        Concatenate all policy text files into a single line-per-sentence
        file that fastText's training expects.
        """
        # Prefer .txt files, but many corpora are stored as HTML.
        policy_files = sorted(glob.glob(os.path.join(self.corpus_dir, "*.txt")))
        # Fallback to HTML if no TXT files exist.
        if not policy_files:
            policy_files = sorted(glob.glob(os.path.join(self.corpus_dir, "*.html")))

        # If the primary corpus directory is empty, try the local OPP-115 fallback.
        if not policy_files and OPP115_DIR:
            for fallback_subdir in ["sanitized_policies", "original_policies"]:
                fallback_dir = os.path.join(OPP115_DIR, fallback_subdir)
                if not os.path.isdir(fallback_dir):
                    continue
                fallback_files = sorted(glob.glob(os.path.join(fallback_dir, "*.txt")))
                if not fallback_files:
                    fallback_files = sorted(glob.glob(os.path.join(fallback_dir, "*.html")))
                if fallback_files:
                    logger.warning(
                        "Primary policy corpus dir %s is empty. "
                        "Using fallback corpus from %s.",
                        self.corpus_dir, fallback_dir,
                    )
                    policy_files = fallback_files
                    self.corpus_dir = fallback_dir
                    break

        if not policy_files:
            raise FileNotFoundError(
                f"No policy corpus files found in {self.corpus_dir}. "
                "Expected *.txt or *.html files."
            )

        logger.info(
            "Building corpus from %d policy files in %s … (extensions: %s)",
            len(policy_files),
            self.corpus_dir,
            {Path(p).suffix.lower() for p in policy_files},
        )

        written = 0
        with open(output_txt, "w", encoding="utf-8") as out_f:
            for fpath in policy_files:
                try:
                    with open(fpath, encoding="utf-8", errors="replace") as f:
                        text = f.read()

                    if fpath.lower().endswith(".html"):
                        # Lightweight HTML stripper (avoid extra dependencies)
                        text = re.sub(r"<script[\s\S]*?</script>", " ", text, flags=re.IGNORECASE)
                        text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
                        text = re.sub(r"<[^>]+>", " ", text)
                        text = re.sub(r"\s+", " ", text)

                    # One sentence per line for fastText
                    sentences = re.split(r"[.!?]\s+", text)

                    for sent in sentences:
                        sent = sent.strip().lower()
                        if len(sent.split()) >= 3:
                            out_f.write(sent + "\n")
                            written += 1
                except Exception as exc:
                    logger.debug("Skipping %s: %s", fpath, exc)

        logger.info("Corpus file written: %d sentences.", written)
        if written == 0:
            raise RuntimeError(
                "Built corpus.txt with 0 sentences. "
                "Check that the corpus files contain readable text."
            )
        return output_txt


    # ── Train ────────────────────────────────────────────────────────────────

    def train(self, corpus_txt: Optional[str] = None):
        """
        Train the fastText model using gensim (cross-platform compatible).
        Uses gensim.models.FastText which avoids Windows compilation issues.

        Parameters
        ----------
        corpus_txt : str, optional
            Path to a pre-built corpus text file.  If None, one is
            generated from ``self.corpus_dir``.

        Returns
        -------
        fasttext model object (gensim-based)
        """
        try:
            from gensim.models import FastText
        except ImportError:
            raise ImportError(
                "gensim is required for embedding training.\n"
                "Install with:  pip install gensim"
            )

        if corpus_txt is None:
            corpus_txt = self.build_corpus_file()

        logger.info(
            "Training FastText %s model (dim=%d, epochs=%d, minn=%d, maxn=%d) using gensim…",
            self.model_type, self.dim, self.epochs, self.min_n, self.max_n,
        )

        # Read corpus file and tokenise sentences
        sentences = []
        with open(corpus_txt, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    sentences.append(line.split())

        # Train using gensim FastText
        sg = 1 if self.model_type == "skipgram" else 0
        ft_model = FastText(
            vector_size=self.dim,
            window=self.window,
            min_count=self.min_count,
            sg=sg,
            min_n=self.min_n,
            max_n=self.max_n,
            workers=4,
        )

        ft_model.build_vocab(corpus_iterable=sentences)
        if ft_model.corpus_count == 0 or ft_model.corpus_total_words == 0:
            raise RuntimeError(
                "FastText training failed: built vocabulary contains no examples. "
                "Check that the corpus file contains valid sentences."
            )

        logger.info(
            "Vocabulary built: %d words, %d sentences.",
            ft_model.corpus_total_words,
            ft_model.corpus_count,
        )

        ft_model.train(
            corpus_iterable=sentences,
            total_examples=ft_model.corpus_count,
            epochs=self.epochs,
        )

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        ft_model.save(self.output_path)
        logger.info("Model saved to %s", self.output_path)
        return ft_model


def load_embeddings(path: str = EMBEDDINGS_PATH):
    """
    Load a saved gensim FastText model.
    Falls back to random embeddings if the file doesn't exist.

    Returns
    -------
    gensim FastText model object or mock model with random embeddings.
    """
    try:
        from gensim.models import FastText
        logger.info("Loading embeddings from %s …", path)
        return FastText.load(path)
    except (ImportError, FileNotFoundError, OSError) as exc:
        # Fallback: create a mock model with random embeddings
        logger.warning(
            "Could not load embeddings from %s: %s. Using random embeddings as fallback.",
            path, exc
        )
        return _create_random_embeddings_model()


def get_nearest_neighbors(ft_model, word: str, topn: int = 5):
    """
    Compatibility wrapper for nearest-neighbor lookup across FastText variants.

    Parameters
    ----------
    ft_model : FastText-like model
        Loaded embeddings model.
    word : str
        Query word.
    topn : int
        Number of nearest neighbors to return.
    """
    if hasattr(ft_model, "wv") and hasattr(ft_model.wv, "most_similar"):
        return ft_model.wv.most_similar(word, topn=topn)
    if hasattr(ft_model, "get_nearest_neighbors"):
        return ft_model.get_nearest_neighbors(word, topn=topn)
    raise AttributeError("The provided model does not support nearest-neighbor lookup.")


def _create_random_embeddings_model():
    """
    Create a mock FastText model that returns random embeddings.
    Used as fallback when trained embeddings are not available.
    """
    class RandomEmbeddingsModel:
        """Mock FastText model returning random vectors for any word."""
        def __init__(self, dim=FASTTEXT_DIM):
            self.dim = dim
            np.random.seed(0)
        
        class WV:
            def __init__(self, parent):
                self.parent = parent
            def get_vector(self, word):
                # Return consistent random vector for each word based on hash
                seed = hash(word) % (2**31)
                rng = np.random.RandomState(seed)
                return rng.randn(self.parent.dim).astype(np.float32) * 0.1
        
        def __init__(self, dim=FASTTEXT_DIM):
            self.dim = dim
            self.wv = self.WV(self)
    
    logger.info("Created random embeddings model (dim=%d)", FASTTEXT_DIM)
    return RandomEmbeddingsModel(FASTTEXT_DIM)


def get_sentence_vector(text: str, ft_model) -> np.ndarray:
    """
    Return the FastText sentence vector for *text* (average of subword
    vectors), replicating fastText's get_sentence_vector.
    """
    try:
        # For gensim FastText model
        tokens = text.lower().split()
        if not tokens:
            return np.zeros(FASTTEXT_DIM)
        vecs = [ft_model.wv.get_vector(t) for t in tokens]
        return np.mean(vecs, axis=0)
    except (AttributeError, KeyError):
        # Fallback for other model types
        if hasattr(ft_model, "get_sentence_vector"):
            return ft_model.get_sentence_vector(text.lower())
        tokens = text.lower().split()
        vecs = [ft_model.get_word_vector(t) for t in tokens]
        if not vecs:
            return np.zeros(FASTTEXT_DIM)
        return np.mean(vecs, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# 4.2  OPP-115 Dataset Loader
# ─────────────────────────────────────────────────────────────────────────────

class OPP115Sample:
    """One annotated text segment from the OPP-115 dataset."""

    __slots__ = ["policy_id", "segment_id", "text",
                 "categories", "attributes"]

    def __init__(
        self,
        policy_id:  str,
        segment_id: str,
        text:       str,
        categories: List[str],
        attributes: Dict[str, List[str]],
    ):
        self.policy_id  = policy_id
        self.segment_id = segment_id
        self.text       = text
        self.categories = categories   # list of HIGH_LEVEL_CATEGORIES labels
        self.attributes = attributes   # {attr_name: [value, ...]}


class OPP115Dataset:
    """
    Loads and prepares the OPP-115 corpus for training Polisis classifiers.

    OPP-115 directory structure (expected)
    ---------------------------------------
    opp115/
      annotations/
        <policy_id>/
          <segment_id>.json    ← annotation JSON per segment
      sanitized_policies/
        <policy_id>.html       ← raw privacy policy HTML

    The dataset is publicly available at:
        https://usableprivacy.org/data

    Train / Test split
    ------------------
    The paper uses 65 policies for training and 50 for testing.
    We replicate this deterministically by sorting policy IDs and
    taking the first 65 as train, the remaining 50 as test.
    """

    def __init__(self, opp115_dir: str = OPP115_DIR):
        self.opp115_dir   = opp115_dir
        self.ann_dir      = os.path.join(opp115_dir, "annotations")
        self.policy_dir   = os.path.join(opp115_dir, "sanitized_policies")
        self._samples: List[OPP115Sample] = []
        self._loaded = False

    # ── Public API ───────────────────────────────────────────────────────────

    def load(self) -> "OPP115Dataset":
        """Parse all annotation files and populate self._samples."""
        if self._loaded:
            return self
        if not os.path.isdir(self.ann_dir):
            raise FileNotFoundError(
                f"OPP-115 annotations not found at {self.ann_dir}.\n"
                "Download from: https://usableprivacy.org/data"
            )

        # Detect whether the dataset uses CSV or JSON annotation layout.
        if self._has_csv_files():
            logger.info("Detected CSV format — using OPP-115 CSV parser.")
            try:
                from opp115_csv_loader import OPP115CSVParser
            except ImportError as exc:
                raise ImportError(
                    "OPP-115 CSV loader requires opp115_csv_loader.py in the same "
                    "directory as ml_layer_embeddings.py."
                ) from exc
            parser = OPP115CSVParser(self.ann_dir, self.policy_dir)
            self._samples = parser.parse_all()
        else:
            for policy_dir in sorted(Path(self.ann_dir).iterdir()):
                if not policy_dir.is_dir():
                    continue
                policy_id = policy_dir.name
                for ann_file in sorted(policy_dir.glob("*.json")):
                    sample = self._parse_annotation(policy_id, ann_file)
                    if sample:
                        self._samples.append(sample)

        logger.info("OPP-115 loaded: %d annotated segments.", len(self._samples))
        self._loaded = True
        return self

    def train_test_split(
        self,
    ) -> Tuple[List[OPP115Sample], List[OPP115Sample]]:
        """
        Return (train_samples, test_samples) with a 65/50 policy-level split.
        """
        self.load()
        policy_ids = sorted({s.policy_id for s in self._samples})

        from opp115_csv_loader import validate_opp115_split
        validate_opp115_split(policy_ids)

        from opp115_policy_split import split_policy_ids
        train_id_order, test_id_order = split_policy_ids(
            policy_ids,
            train_count=OPP115_TRAIN_POLICIES,
            eval_count=OPP115_TEST_POLICIES,
        )
        train_ids = set(train_id_order)
        test_ids = set(test_id_order)

        train = [s for s in self._samples if s.policy_id in train_ids]
        test  = [s for s in self._samples if s.policy_id in test_ids]
        logger.info(
            "Split -> train: %d segments (%d policies), "
            "test: %d segments (%d policies).",
            len(train), len(train_ids), len(test), len(test_ids),
        )
        return train, test

    # ── Label encoding ───────────────────────────────────────────────────────

    @staticmethod
    def encode_categories(sample: OPP115Sample) -> np.ndarray:
        """
        Binary multi-hot vector over HIGH_LEVEL_CATEGORIES.
        Shape: (10,)
        """
        vec = np.zeros(len(HIGH_LEVEL_CATEGORIES), dtype=np.float32)
        for cat in sample.categories:
            if cat in HIGH_LEVEL_CATEGORIES:
                vec[HIGH_LEVEL_CATEGORIES.index(cat)] = 1.0
        return vec

    @staticmethod
    def encode_attribute(
        sample: OPP115Sample, attr_name: str
    ) -> np.ndarray:
        """
        Binary multi-hot vector over ATTRIBUTES[attr_name].
        """
        values = ATTRIBUTES.get(attr_name, [])
        vec = np.zeros(len(values), dtype=np.float32)
        for val in sample.attributes.get(attr_name, []):
            if val in values:
                vec[values.index(val)] = 1.0
        return vec

    # ── Internal parsing ─────────────────────────────────────────────────────

    def _has_csv_files(self) -> bool:
        """Return True if the annotations directory contains CSV files."""
        return any(Path(self.ann_dir).glob("*.csv"))

    @staticmethod
    def _parse_annotation(
        policy_id: str, ann_file: Path
    ) -> Optional[OPP115Sample]:
        """
        Parse a single OPP-115 annotation JSON file.

        The OPP-115 JSON structure is:
        {
          "segment_text": "...",
          "annotations": [
            {
              "category": "First Party Collection/Use",
              "attributes": {
                "personal_information_type": ["contact"],
                "purpose": ["advertising"],
                ...
              }
            }, ...
          ]
        }
        We union annotations across all annotators (majority-vote is
        handled at the classifier training stage via multi-label cross-
        entropy with threshold 0.5).
        """
        try:
            with open(ann_file, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

        text = data.get("segment_text", "").strip()
        if not text:
            return None

        categories: List[str]          = []
        attributes: Dict[str, List[str]] = {}

        for ann in data.get("annotations", []):
            cat = ann.get("category", "")
            if cat and cat not in categories:
                categories.append(cat)
            for attr_name, vals in ann.get("attributes", {}).items():
                if attr_name not in attributes:
                    attributes[attr_name] = []
                for v in (vals if isinstance(vals, list) else [vals]):
                    if v and v not in attributes[attr_name]:
                        attributes[attr_name].append(v)

        return OPP115Sample(
            policy_id  = policy_id,
            segment_id = ann_file.stem,
            text       = text,
            categories = categories,
            attributes = attributes,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Batch tokeniser + embedding matrix builder  (feeds into CNN models)
# ─────────────────────────────────────────────────────────────────────────────

class TextVectoriser:
    """
    Converts raw text into fixed-length integer token sequences for the
    CNN input layer.

    Steps
    -----
    1. PENN Treebank tokenisation (NLTK) or whitespace fallback.
    2. Build vocab from training corpus.
    3. Map tokens → integer IDs (0 = padding, 1 = UNK).
    4. Construct an embedding matrix aligned to the vocab, using the
       fastText model to fill each row.
    """

    PAD_ID = 0
    UNK_ID = 1

    def __init__(self, max_len: int = 200, min_freq: int = 1):
        self.max_len  = max_len
        self.min_freq = min_freq
        self.vocab: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self._fitted  = False

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, texts: List[str]) -> "TextVectoriser":
        from collections import Counter
        freq: Counter = Counter()
        for t in texts:
            freq.update(self._tokenise(t))
        for tok, cnt in freq.items():
            if cnt >= self.min_freq and tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
        self._fitted = True
        logger.info("Vocabulary size: %d tokens.", len(self.vocab))
        return self

    # ── Transform ────────────────────────────────────────────────────────────

    def transform(self, texts: List[str]) -> np.ndarray:
        """Return (N, max_len) integer array."""
        ids = np.zeros((len(texts), self.max_len), dtype=np.int32)
        for i, t in enumerate(texts):
            toks = self._tokenise(t)[: self.max_len]
            for j, tok in enumerate(toks):
                ids[i, j] = self.vocab.get(tok, self.UNK_ID)
        return ids

    # ── Embedding matrix ─────────────────────────────────────────────────────

    def build_embedding_matrix(self, ft_model) -> np.ndarray:
        """
        Return (vocab_size, dim) matrix with rows filled from fastText embeddings.
        Row 0 (PAD) and row 1 (UNK) are zero-initialised.
        Works with both gensim and fasttext models.
        """
        vocab_size = len(self.vocab)
        dim = FASTTEXT_DIM
        matrix = np.zeros((vocab_size, dim), dtype=np.float32)
        for tok, idx in self.vocab.items():
            if idx < 2:
                continue  # PAD / UNK stay zero
            try:
                # Try gensim interface first
                if hasattr(ft_model, 'wv'):
                    matrix[idx] = ft_model.wv.get_vector(tok)
                else:
                    # Fallback to fasttext interface
                    matrix[idx] = ft_model.get_word_vector(tok)
            except (KeyError, AttributeError):
                pass  # Keep zero vector if word not found
        logger.info(
            "Embedding matrix built: shape %s.", str(matrix.shape)
        )
        return matrix

    # ── Tokeniser ────────────────────────────────────────────────────────────

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        """PENN Treebank tokenisation via NLTK (whitespace fallback)."""
        text = text.lower()
        try:
            from nltk.tokenize import word_tokenize  # type: ignore
            try:
                return word_tokenize(text)
            except LookupError:
                import nltk
                nltk.download("punkt", quiet=True)
                return word_tokenize(text)
        except ImportError:
            return text.split()
