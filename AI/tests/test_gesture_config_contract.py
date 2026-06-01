import json
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DOUBLE_CONSONANT_IDS = {1, 4, 8, 10, 13}
COMPOUND_VOWEL_IDS = {28, 29, 33, 34}
EXCLUDED_COMPOSED_IDS = DOUBLE_CONSONANT_IDS | COMPOUND_VOWEL_IDS


def load_config(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


class GestureConfigContractTests(unittest.TestCase):
    def assert_config_contract(self, config):
        active_ids = {int(label_id) for label_id in config["active_label_ids"]}
        collection_ids = {int(label_id) for label_id in config["collection_label_ids"]}
        labels_by_id = {int(label["id"]): label for label in config["labels"]}

        self.assertEqual(len(active_ids), 32)
        self.assertEqual(len(collection_ids), 34)
        self.assertEqual(config["two_hand_label_ids"], [])
        self.assertFalse(active_ids & EXCLUDED_COMPOSED_IDS)
        self.assertFalse(collection_ids & EXCLUDED_COMPOSED_IDS)
        self.assertTrue(COMPOUND_VOWEL_IDS <= labels_by_id.keys())
        self.assertTrue({40, 41, 42} <= collection_ids)
        self.assertIn(42, active_ids)

        for label_id in COMPOUND_VOWEL_IDS:
            label = labels_by_id[label_id]
            self.assertNotEqual(label.get("collection_mode"), "both_hands")
            self.assertIn("조합 전용", label["description"])

    def test_root_config_keeps_composed_labels_out_of_training_and_collection(self):
        self.assert_config_contract(load_config(ROOT_DIR / "gesture_config.json"))

    def test_collection_bundle_config_matches_current_contract(self):
        self.assert_config_contract(
            load_config(ROOT_DIR / "team_share/data_collection_bundle/gesture_config.json")
        )


if __name__ == "__main__":
    unittest.main()
