from collections import Counter, deque

import numpy as np

from .hangul_composer import HangulComposer


IDLE = "IDLE"
INPUT = "INPUT"
RESULT = "RESULT"


class StableGestureDetector:
    def __init__(self, window=7, min_ratio=0.7, min_confidence=0.75):
        self.history = deque(maxlen=window)
        self.min_ratio = min_ratio
        self.min_confidence = min_confidence
        self.latched_label = None

    def reset(self):
        self.history.clear()
        self.latched_label = None

    def observe_no_hand(self):
        self.reset()

    def observe(self, label_id, confidence):
        self.history.append((int(label_id), float(confidence)))
        counts = Counter(label for label, _ in self.history)
        stable_label, stable_count = counts.most_common(1)[0]
        stable_values = [value for label, value in self.history if label == stable_label]
        ratio = stable_count / len(self.history)
        mean_confidence = float(np.mean(stable_values))
        is_stable = (
            len(self.history) == self.history.maxlen
            and ratio >= self.min_ratio
            and mean_confidence >= self.min_confidence
        )

        event_label = None
        if is_stable and stable_label != self.latched_label:
            event_label = stable_label
            self.latched_label = stable_label
        return {
            "label_id": stable_label,
            "ratio": ratio,
            "confidence": mean_confidence,
            "stable": is_stable,
            "event_label": event_label,
        }


class RecognitionSession:
    def __init__(self, labels, control_labels):
        self.labels = {int(label["id"]): label for label in labels}
        self.start_id = int(control_labels["start"])
        self.end_id = int(control_labels["end"])
        self.delete_id = int(control_labels["delete"])
        self.composer = HangulComposer()
        self.state = IDLE
        self.message = "양손 펼친손을 보여 입력을 시작하세요"
        self.final_text = ""

    @property
    def text(self):
        return self.composer.text

    def clear(self):
        self.composer.clear()
        self.final_text = ""
        self.state = IDLE
        self.message = "입력이 초기화되었습니다. 양손 펼친손으로 시작하세요"

    def handle(self, label_id):
        label_id = int(label_id)
        if self.state in (IDLE, RESULT):
            if label_id == self.start_id:
                self.composer.clear()
                self.final_text = ""
                self.state = INPUT
                self.message = "입력 모드: 자음을 입력하세요"
                return True
            return False

        if label_id == self.end_id:
            self.final_text = self.text
            self.state = RESULT
            self.message = "입력 종료. 양손 펼친손으로 새 입력을 시작하세요"
            return True
        if label_id == self.delete_id:
            _, self.message = self.composer.delete()
            return True
        if label_id == self.start_id:
            return False

        label = self.labels.get(label_id)
        if not label or label.get("group") not in {"자음", "모음"}:
            return False
        _, self.message = self.composer.add(label["code"])
        return True
