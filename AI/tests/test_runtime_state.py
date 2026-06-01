import json
import unittest
from pathlib import Path

from src.sign_language.runtime_state import IDLE, INPUT, RESULT, RecognitionSession, StableGestureDetector


ROOT_DIR = Path(__file__).resolve().parent.parent


def config():
    with (ROOT_DIR / "gesture_config.json").open("r", encoding="utf-8") as file:
        return json.load(file)


class RecognitionSessionTests(unittest.TestCase):
    def setUp(self):
        app_config = config()
        self.session = RecognitionSession(app_config["labels"], app_config["control_labels"])

    def test_start_input_delete_end_flow(self):
        self.assertEqual(self.session.state, IDLE)
        self.session.handle(40)
        self.assertEqual(self.session.state, INPUT)
        for label_id in [0, 19, 2, 19]:
            self.session.handle(label_id)
        self.assertEqual(self.session.text, "가나")
        self.session.handle(42)
        self.assertEqual(self.session.text, "간")
        self.session.handle(41)
        self.assertEqual(self.session.state, RESULT)
        self.assertEqual(self.session.final_text, "간")

    def test_repeated_base_consonants_compose_without_double_labels(self):
        self.session.handle(40)
        for label_id in [9, 9, 19]:
            self.session.handle(label_id)
        self.assertEqual(self.session.text, "싸")

    def test_base_vowel_labels_compose_compound_vowels(self):
        self.session.handle(40)
        for label_id in [11, 27, 19]:
            self.session.handle(label_id)
        self.assertEqual(self.session.text, "와")

        self.session.handle(42)
        self.session.handle(42)
        self.session.handle(42)
        for label_id in [11, 32, 24]:
            self.session.handle(label_id)
        self.assertEqual(self.session.text, "웨")

    def test_unknown_label_does_not_start_input(self):
        self.session.handle(999)
        self.assertEqual(self.session.state, IDLE)


class StableGestureDetectorTests(unittest.TestCase):
    def test_emits_once_after_full_stable_window_and_resets_after_no_hand(self):
        detector = StableGestureDetector(window=3, min_ratio=1.0, min_confidence=0.75)
        self.assertIsNone(detector.observe(40, 0.9)["event_label"])
        self.assertIsNone(detector.observe(40, 0.9)["event_label"])
        self.assertEqual(detector.observe(40, 0.9)["event_label"], 40)
        self.assertIsNone(detector.observe(40, 0.9)["event_label"])
        detector.observe_no_hand()
        detector.observe(40, 0.9)
        detector.observe(40, 0.9)
        self.assertEqual(detector.observe(40, 0.9)["event_label"], 40)


if __name__ == "__main__":
    unittest.main()
