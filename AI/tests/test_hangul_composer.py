import unittest

from src.sign_language.hangul_composer import HangulComposer, compose_syllable


class HangulComposerTests(unittest.TestCase):
    def test_composes_supported_initial_and_vowel(self):
        self.assertEqual(compose_syllable("ㄱ", "ㅏ"), "가")
        self.assertEqual(compose_syllable("ㄴ", "ㅗ"), "노")
        self.assertEqual(compose_syllable("ㄱ", "ㅏ", "ㅂ"), "갑")

    def test_composes_multiple_syllables_and_final_consonant(self):
        composer = HangulComposer()
        for token in ["ㄱ", "ㅏ", "ㄴ", "ㅏ", "ㅁ"]:
            accepted, _ = composer.add(token)
            self.assertTrue(accepted)
        self.assertEqual(composer.text, "가남")

    def test_delete_removes_one_jamo_like_backspace(self):
        composer = HangulComposer()
        for token in ["ㄱ", "ㅏ", "ㄴ", "ㅏ"]:
            composer.add(token)
        composer.delete()
        self.assertEqual(composer.text, "간")
        composer.delete()
        self.assertEqual(composer.text, "가")

    def test_repeated_consonants_make_double_initials(self):
        for tokens, expected in [
            (["ㅅ", "ㅅ", "ㅏ"], "싸"),
            (["ㅂ", "ㅂ", "ㅏ"], "빠"),
            (["ㅈ", "ㅈ", "ㅏ"], "짜"),
            (["ㄱ", "ㄱ", "ㅏ"], "까"),
            (["ㄷ", "ㄷ", "ㅏ"], "따"),
        ]:
            composer = HangulComposer()
            for token in tokens:
                accepted, _ = composer.add(token)
                self.assertTrue(accepted)
            self.assertEqual(composer.text, expected)

    def test_base_vowels_make_compound_vowels(self):
        for tokens, expected in [
            (["ㅗ", "ㅏ"], "ㅘ"),
            (["ㅗ", "ㅐ"], "ㅙ"),
            (["ㅜ", "ㅓ"], "ㅝ"),
            (["ㅜ", "ㅔ"], "ㅞ"),
            (["ㅇ", "ㅗ", "ㅏ"], "와"),
            (["ㅇ", "ㅗ", "ㅐ"], "왜"),
            (["ㅇ", "ㅜ", "ㅓ"], "워"),
            (["ㅇ", "ㅜ", "ㅔ"], "웨"),
            (["ㄱ", "ㅗ", "ㅏ", "ㄴ"], "관"),
        ]:
            composer = HangulComposer()
            for token in tokens:
                accepted, _ = composer.add(token)
                self.assertTrue(accepted)
            self.assertEqual(composer.text, expected)

    def test_delete_after_compound_vowel_removes_last_raw_vowel(self):
        composer = HangulComposer()
        for token in ["ㅇ", "ㅜ", "ㅔ"]:
            composer.add(token)
        self.assertEqual(composer.text, "웨")
        composer.delete()
        self.assertEqual(composer.text, "우")

    def test_composes_final_and_compound_final_before_next_vowel(self):
        composer = HangulComposer()
        for token in ["ㅈ", "ㅏ", "ㄴ", "ㅎ", "ㅏ"]:
            composer.add(token)
        self.assertEqual(composer.text, "잔하")

        composer.clear()
        for token in ["ㅈ", "ㅏ", "ㄴ", "ㅎ", "ㅎ", "ㅏ"]:
            composer.add(token)
        self.assertEqual(composer.text, "잖하")

    def test_keeps_last_consonant_as_next_initial_when_vowel_follows(self):
        composer = HangulComposer()
        for token in ["ㄱ", "ㅏ", "ㅂ", "ㅅ", "ㅏ"]:
            composer.add(token)
        self.assertEqual(composer.text, "갑사")

        composer.clear()
        for token in ["ㄱ", "ㅏ", "ㅂ", "ㅅ", "ㅇ", "ㅏ"]:
            composer.add(token)
        self.assertEqual(composer.text, "값아")

    def test_vowel_without_initial_is_shown_as_jamo(self):
        composer = HangulComposer()
        accepted, _ = composer.add("ㅏ")
        self.assertTrue(accepted)
        self.assertEqual(composer.text, "ㅏ")


if __name__ == "__main__":
    unittest.main()
