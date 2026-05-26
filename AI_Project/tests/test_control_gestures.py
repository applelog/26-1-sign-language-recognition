import unittest
from types import SimpleNamespace

from control_gestures import detect_bilateral_control, extended_finger_count


def hand(opened):
    points = [SimpleNamespace(x=0.0, y=0.0, z=0.0) for _ in range(21)]
    for pip_index, tip_index in ((6, 8), (10, 12), (14, 16), (18, 20)):
        points[pip_index] = SimpleNamespace(x=0.0, y=1.0, z=0.0)
        distance = 2.0 if opened else 0.7
        points[tip_index] = SimpleNamespace(x=0.0, y=distance, z=0.0)
    return SimpleNamespace(landmark=points)


class BilateralControlTests(unittest.TestCase):
    def test_two_open_hands_detect_start(self):
        self.assertEqual(detect_bilateral_control([hand(True), hand(True)], 40, 41), 40)

    def test_two_fists_detect_end(self):
        self.assertEqual(detect_bilateral_control([hand(False), hand(False)], 40, 41), 41)

    def test_one_hand_is_not_a_control_event(self):
        self.assertIsNone(detect_bilateral_control([hand(True)], 40, 41))
        self.assertEqual(extended_finger_count(hand(True)), 4)


if __name__ == "__main__":
    unittest.main()
