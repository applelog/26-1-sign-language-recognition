CHOSEONG = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ",
    "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]
JUNGSEONG = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ",
    "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
]
JONGSEONG = [
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
    "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ",
    "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]

_CHOSEONG_INDEX = {value: index for index, value in enumerate(CHOSEONG)}
_JUNGSEONG_INDEX = {value: index for index, value in enumerate(JUNGSEONG)}
_JONGSEONG_INDEX = {value: index for index, value in enumerate(JONGSEONG)}
_DOUBLE_INITIALS = {
    ("ㄱ", "ㄱ"): "ㄲ",
    ("ㄷ", "ㄷ"): "ㄸ",
    ("ㅂ", "ㅂ"): "ㅃ",
    ("ㅅ", "ㅅ"): "ㅆ",
    ("ㅈ", "ㅈ"): "ㅉ",
}
_COMPOUND_FINALS = {
    ("ㄱ", "ㅅ"): "ㄳ",
    ("ㄴ", "ㅈ"): "ㄵ",
    ("ㄴ", "ㅎ"): "ㄶ",
    ("ㄹ", "ㄱ"): "ㄺ",
    ("ㄹ", "ㅁ"): "ㄻ",
    ("ㄹ", "ㅂ"): "ㄼ",
    ("ㄹ", "ㅅ"): "ㄽ",
    ("ㄹ", "ㅌ"): "ㄾ",
    ("ㄹ", "ㅍ"): "ㄿ",
    ("ㄹ", "ㅎ"): "ㅀ",
    ("ㅂ", "ㅅ"): "ㅄ",
}
_COMPOUND_VOWELS = {
    ("ㅗ", "ㅏ"): "ㅘ",
    ("ㅗ", "ㅐ"): "ㅙ",
    ("ㅜ", "ㅓ"): "ㅝ",
    ("ㅜ", "ㅔ"): "ㅞ",
}


def compose_syllable(consonant, vowel, final=""):
    if consonant not in _CHOSEONG_INDEX or vowel not in _JUNGSEONG_INDEX:
        raise ValueError(f"조합할 수 없는 자모입니다: {consonant}, {vowel}")
    if final not in _JONGSEONG_INDEX:
        raise ValueError(f"조합할 수 없는 받침입니다: {final}")
    codepoint = (
        0xAC00
        + (_CHOSEONG_INDEX[consonant] * 21 + _JUNGSEONG_INDEX[vowel]) * 28
        + _JONGSEONG_INDEX[final]
    )
    return chr(codepoint)


def _is_consonant(jamo):
    return jamo in _CHOSEONG_INDEX or jamo in _JONGSEONG_INDEX


def _is_vowel(jamo):
    return jamo in _JUNGSEONG_INDEX


def _consume_initial(tokens, index):
    if index + 1 < len(tokens) and _DOUBLE_INITIALS.get((tokens[index], tokens[index + 1])):
        return _DOUBLE_INITIALS[(tokens[index], tokens[index + 1])], index + 2
    return tokens[index], index + 1


def _compose_final(consonants):
    if not consonants:
        return "", 0
    if len(consonants) >= 2 and (consonants[0], consonants[1]) in _COMPOUND_FINALS:
        return _COMPOUND_FINALS[(consonants[0], consonants[1])], 2
    if len(consonants) >= 2 and (consonants[0], consonants[1]) in _DOUBLE_INITIALS:
        doubled = _DOUBLE_INITIALS[(consonants[0], consonants[1])]
        if doubled in _JONGSEONG_INDEX:
            return doubled, 2
    if consonants[0] in _JONGSEONG_INDEX:
        return consonants[0], 1
    return "", 0


def _consume_vowel(tokens, index):
    if index + 1 < len(tokens) and _COMPOUND_VOWELS.get((tokens[index], tokens[index + 1])):
        return _COMPOUND_VOWELS[(tokens[index], tokens[index + 1])], index + 2
    return tokens[index], index + 1


class HangulComposer:
    """입력한 원시 자모 토큰을 보존하고 표시 문자열을 계산합니다."""

    def __init__(self):
        self.tokens = []

    def clear(self):
        self.tokens.clear()

    def add(self, jamo):
        if _is_consonant(jamo):
            self.tokens.append(jamo)
            return True, f"자음 입력: {jamo}"
        if _is_vowel(jamo):
            self.tokens.append(jamo)
            return True, f"모음 입력: {jamo}"
        return False, f"지원하지 않는 자모입니다: {jamo}"

    def delete(self):
        if not self.tokens:
            return False, "삭제할 입력이 없습니다"
        deleted = self.tokens.pop()
        return True, f"삭제: {deleted}"

    @property
    def text(self):
        output = []
        index = 0
        while index < len(self.tokens):
            token = self.tokens[index]
            if _is_vowel(token):
                vowel, index = _consume_vowel(self.tokens, index)
                output.append(vowel)
                continue

            if not _is_consonant(token):
                index += 1
                continue

            initial, next_index = _consume_initial(self.tokens, index)
            if next_index >= len(self.tokens) or not _is_vowel(self.tokens[next_index]):
                output.append(initial)
                index = next_index
                continue

            vowel, after_vowel = _consume_vowel(self.tokens, next_index)
            consonants = []
            scan = after_vowel
            while scan < len(self.tokens) and _is_consonant(self.tokens[scan]):
                consonants.append(self.tokens[scan])
                scan += 1

            if scan < len(self.tokens) and _is_vowel(self.tokens[scan]) and consonants:
                final_candidates = consonants[:-1]
                carry_count = 1
            else:
                final_candidates = consonants
                carry_count = 0

            final, consumed = _compose_final(final_candidates)
            output.append(compose_syllable(initial, vowel, final))

            leftovers = final_candidates[consumed:]
            if leftovers:
                output.extend(leftovers)
            if carry_count:
                index = scan - carry_count
            else:
                index = after_vowel + len(consonants)
        return "".join(output)
