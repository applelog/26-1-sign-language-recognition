import math


FINGER_PIP_TIP = ((6, 8), (10, 12), (14, 16), (18, 20))


def _distance(first, second):
    return math.sqrt(
        (first.x - second.x) ** 2
        + (first.y - second.y) ** 2
        + (first.z - second.z) ** 2
    )


def extended_finger_count(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    count = 0
    for pip_index, tip_index in FINGER_PIP_TIP:
        pip_distance = _distance(wrist, hand_landmarks.landmark[pip_index])
        tip_distance = _distance(wrist, hand_landmarks.landmark[tip_index])
        if tip_distance > pip_distance * 1.15:
            count += 1
    return count


def detect_bilateral_control(hand_landmarks, start_id, end_id):
    if len(hand_landmarks) != 2:
        return None
    finger_counts = [extended_finger_count(hand) for hand in hand_landmarks]
    if all(count >= 4 for count in finger_counts):
        return int(start_id)
    if all(count == 0 for count in finger_counts):
        return int(end_id)
    return None
