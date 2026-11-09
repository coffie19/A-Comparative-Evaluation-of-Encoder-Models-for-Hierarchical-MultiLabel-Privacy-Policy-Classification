import unittest

from opp115_policy_split import split_policy_ids


class OPP115PolicySplitTests(unittest.TestCase):
    def test_exact_65_50_split_is_stable_and_disjoint(self):
        policy_ids = [f"policy-{index:03d}" for index in range(115)]
        train_a, eval_a = split_policy_ids(policy_ids)
        train_b, eval_b = split_policy_ids(reversed(policy_ids))

        self.assertEqual(len(train_a), 65)
        self.assertEqual(len(eval_a), 50)
        self.assertEqual(train_a, train_b)
        self.assertEqual(eval_a, eval_b)
        self.assertFalse(set(train_a) & set(eval_a))
        self.assertEqual(set(train_a) | set(eval_a), set(policy_ids))

    def test_wrong_policy_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires 115"):
            split_policy_ids([f"policy-{index}" for index in range(114)])

    def test_duplicate_policy_ids_are_rejected(self):
        policy_ids = [f"policy-{index}" for index in range(114)] + ["policy-0"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            split_policy_ids(policy_ids)


if __name__ == "__main__":
    unittest.main()
