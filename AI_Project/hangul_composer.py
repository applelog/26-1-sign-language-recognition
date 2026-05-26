CHOSEONG = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ",
    "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]
JUNGSEONG = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ",
    "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
]

_CHOSEONG_INDEX = {value: index for index, value in enumerate(CHOSEONG)}
_JUNGSEONG_INDEX = {value: index for index, value in enumerate(JUNGSEONG)}


def compose_syllable(consonant, vowel):
    if consonant not in _CHOSEONG_INDEX or vowel not in _JUNGSEONG_INDEX:
        raise ValueError(f"조합할 수 없는 자모입니다: {consonant}, {vowel}")
    codepoint = 0xAC00 + (_CHOSEONG_INDEX[consonant] * 21 + _JUNGSEONG_INDEX[vowel]) * 28
    return chr(codepoint)


class HangulComposer:
    """초성-중성 입력 토큰을 보존하고 완성형 표시 문자열을 생성합니다."""

    def __init__(self):
        self.tokens = []

    def clear(self):
        self.tokens.clear()

    def add(self, jamo):
        expects_consonant = len(self.tokens) % 2 == 0
        if expects_consonant and jamo in _CHOSEONG_INDEX:
            self.tokens.append(jamo)
            return True, f"초성 입력: {jamo}"
        if not expects_consonant and jamo in _JUNGSEONG_INDEX:
            self.tokens.append(jamo)
            return True, f"중성 입력: {jamo}"
        expected = "자음" if expects_consonant else "모음"
        return False, f"{expected}을(를) 입력하세요"

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
            consonant = self.tokens[index]
            if index + 1 < len(self.tokens):
                output.append(compose_syllable(consonant, self.tokens[index + 1]))
                index += 2
            else:
                output.append(consonant)
                index += 1
        return "".join(output)
