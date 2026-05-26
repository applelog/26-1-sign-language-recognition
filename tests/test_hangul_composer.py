import unittest

from hangul_composer import HangulComposer, compose_syllable


class HangulComposerTests(unittest.TestCase):
    def test_composes_supported_initial_and_vowel(self):
        self.assertEqual(compose_syllable("ㄱ", "ㅏ"), "가")
        self.assertEqual(compose_syllable("ㄴ", "ㅗ"), "노")

    def test_composes_multiple_syllables_and_shows_pending_consonant(self):
        composer = HangulComposer()
        for token in ["ㄱ", "ㅏ", "ㄴ", "ㅏ", "ㅁ"]:
            accepted, _ = composer.add(token)
            self.assertTrue(accepted)
        self.assertEqual(composer.text, "가나ㅁ")

    def test_delete_removes_one_jamo_like_backspace(self):
        composer = HangulComposer()
        for token in ["ㄱ", "ㅏ", "ㄴ", "ㅏ"]:
            composer.add(token)
        composer.delete()
        self.assertEqual(composer.text, "가ㄴ")
        composer.delete()
        self.assertEqual(composer.text, "가")

    def test_rejects_out_of_order_jamo(self):
        composer = HangulComposer()
        accepted, _ = composer.add("ㅏ")
        self.assertFalse(accepted)
        composer.add("ㄱ")
        accepted, _ = composer.add("ㄴ")
        self.assertFalse(accepted)
        self.assertEqual(composer.text, "ㄱ")


if __name__ == "__main__":
    unittest.main()
