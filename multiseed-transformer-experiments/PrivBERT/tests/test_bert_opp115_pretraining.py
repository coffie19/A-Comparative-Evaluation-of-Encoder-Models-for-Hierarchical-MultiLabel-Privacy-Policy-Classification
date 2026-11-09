import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from bert_opp115_pretraining import (
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIR,
    PolicyDocument,
    build_argument_parser,
    build_token_chunks,
    clean_opp115_policy,
    discover_policy_files,
    load_policy_documents,
    split_documents,
    validate_roberta_model_type,
)


class OPP115PretrainingDataTests(unittest.TestCase):
    def test_defaults_select_privbert_without_reusing_previous_outputs(self):
        args = build_argument_parser().parse_args([])
        self.assertEqual(DEFAULT_MODEL_NAME, "mukund/privbert")
        self.assertEqual(args.model_name, DEFAULT_MODEL_NAME)
        self.assertEqual(DEFAULT_OUTPUT_DIR.name, "privbert-opp115-dapt")

    def test_model_type_guard_accepts_only_roberta(self):
        validate_roberta_model_type(
            SimpleNamespace(config=SimpleNamespace(model_type="roberta"))
        )
        with self.assertRaisesRegex(ValueError, "model_type='roberta'"):
            validate_roberta_model_type(
                SimpleNamespace(config=SimpleNamespace(model_type="bert"))
            )

    def test_clean_opp115_policy(self):
        raw = (
            "<p>Privacy&nbsp;Policy</p>|||First <b>party</b><br>collection"
            "<script>ignore me</script>"
        )
        self.assertEqual(
            clean_opp115_policy(raw),
            "Privacy Policy\n\nFirst party\n\ncollection",
        )

    def test_nested_discovery_reads_only_sanitized_policies(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sanitized = root / "dataset-a" / "sanitized_policies"
            noisy = root / "dataset-a" / "original_policies"
            sanitized.mkdir(parents=True)
            noisy.mkdir(parents=True)
            wanted = sanitized / "policy.html"
            wanted.write_text("Policy text", encoding="utf-8")
            (noisy / "archived.html").write_text("Noise", encoding="utf-8")

            self.assertEqual(discover_policy_files(root), [wanted.resolve()])

    def test_loading_and_split_are_document_level_and_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sanitized = root / "sanitized_policies"
            sanitized.mkdir()
            for index in range(10):
                (sanitized / f"{index}.html").write_text(
                    f"Policy {index}|||Second segment", encoding="utf-8"
                )

            documents = load_policy_documents(root)
            train_a, eval_a = split_documents(
                documents, train_count=8, eval_count=2
            )
            train_b, eval_b = split_documents(
                list(reversed(documents)), train_count=8, eval_count=2
            )

            self.assertEqual(len(documents), 10)
            self.assertEqual(sum(document.segments for document in documents), 20)
            self.assertEqual([doc.path for doc in train_a], [doc.path for doc in train_b])
            self.assertEqual([doc.path for doc in eval_a], [doc.path for doc in eval_b])
            self.assertFalse(
                {doc.path for doc in train_a} & {doc.path for doc in eval_a}
            )

    def test_identical_policy_content_is_loaded_only_once(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sanitized_a = root / "dataset-a" / "sanitized_policies"
            sanitized_b = root / "dataset-b" / "sanitized_policies"
            sanitized_a.mkdir(parents=True)
            sanitized_b.mkdir(parents=True)
            (sanitized_a / "one.html").write_text(
                "Same privacy policy", encoding="utf-8"
            )
            (sanitized_b / "duplicate.html").write_text(
                "Same privacy policy", encoding="utf-8"
            )
            (sanitized_b / "unique.html").write_text(
                "A different privacy policy", encoding="utf-8"
            )

            documents = load_policy_documents(root)

            self.assertEqual(len(documents), 2)
            self.assertEqual(len({document.text for document in documents}), 2)

    def test_split_rejects_wrong_policy_counts(self):
        documents = [
            PolicyDocument(Path("a"), "a", 1, 1),
            PolicyDocument(Path("b"), "b", 1, 1),
        ]
        with self.assertRaises(ValueError):
            split_documents(documents, train_count=1, eval_count=2)

    def test_token_chunks_are_fixed_size_and_mark_padding_special(self):
        class StubTokenizer:
            pad_token_id = 0

            def num_special_tokens_to_add(self, pair=False):
                self.assert_false(pair)
                return 2

            @staticmethod
            def assert_false(value):
                if value:
                    raise AssertionError("pair tokenization was not expected")

            def __call__(self, text, **kwargs):
                del text, kwargs
                return {"input_ids": list(range(1000, 1010))}

            def build_inputs_with_special_tokens(self, token_ids):
                return [101, *token_ids, 102]

            def get_special_tokens_mask(self, token_ids, already_has_special_tokens):
                self.assertTrue(already_has_special_tokens)
                return [1, *([0] * (len(token_ids) - 2)), 1]

            @staticmethod
            def assertTrue(value):
                if not value:
                    raise AssertionError("tokens must already include special tokens")

        document = PolicyDocument(Path("policy.html"), "policy", 6, 1)
        dataset = build_token_chunks([document], StubTokenizer(), block_size=10)

        self.assertEqual(len(dataset), 2)
        self.assertEqual(len(dataset[0]["input_ids"]), 10)
        self.assertEqual(dataset[0]["input_ids"][:1], [101])
        self.assertEqual(dataset[0]["input_ids"][-1:], [102])
        self.assertEqual(
            dataset[1]["attention_mask"], [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        )
        self.assertEqual(dataset[1]["input_ids"][-6:], [0] * 6)
        self.assertEqual(dataset[1]["special_tokens_mask"][-6:], [1] * 6)


if __name__ == "__main__":
    unittest.main()
