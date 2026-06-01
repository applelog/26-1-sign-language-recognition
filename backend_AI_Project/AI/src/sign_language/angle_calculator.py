import numpy as np

# 관절 인덱스 연결 고리 (예: 0번-1번-2번 관절이 이루는 각도)
joint_list = [[0,1,2], [1,2,3], [2,3,4], [0,5,6], [5,6,7], [6,7,8], 
              [0,9,10], [9,10,11], [10,11,12], [0,13,14], [13,14,15], [14,15,16], 
              [0,17,18], [17,18,19], [18,19,20]]

FINGERTIP_INDICES = [4, 8, 12, 16, 20]
PALM_BASE_INDICES = [5, 9, 13, 17]


def _landmarks_to_joint_array(landmarks):
    joint = np.zeros((21, 3))
    for j, lm in enumerate(landmarks.landmark):
        joint[j] = [lm.x, lm.y, lm.z]
    return joint

def compute_angles(landmarks):
    """
    MediaPipe landmark 객체를 받아 15개의 관절 각도(Degree) 배열을 반환합니다.
    """
    joint = _landmarks_to_joint_array(landmarks)

    v1 = joint[[joint_list[i][0] for i in range(len(joint_list))]]
    v2 = joint[[joint_list[i][1] for i in range(len(joint_list))]]
    v3 = joint[[joint_list[i][2] for i in range(len(joint_list))]]
    
    vec1 = v1 - v2
    vec2 = v3 - v2

    # 벡터 정규화
    norm1 = np.linalg.norm(vec1, axis=1)[:, np.newaxis]
    norm2 = np.linalg.norm(vec2, axis=1)[:, np.newaxis]
    
    # 0으로 나누기 방지
    norm1[norm1 == 0] = 1e-6
    norm2[norm2 == 0] = 1e-6
    
    vec1 = vec1 / norm1
    vec2 = vec2 / norm2

    # 내적 및 아크코사인으로 각도 계산 (도 단위 변환)
    dot_product = np.einsum('nt,nt->n', vec1, vec2)
    # 수치적 오류로 인해 -1.0 ~ 1.0 범위를 벗어나는 것 방지
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    angle = np.arccos(dot_product)
    angle = np.degrees(angle)
    
    return angle


def compute_features(landmarks):
    """
    각도 + 정규화된 상대 좌표 + 손끝 거리 특징을 합쳐 반환합니다.
    단순 각도 15개보다 손 모양/방향 정보를 더 많이 담습니다.
    """
    joint = _landmarks_to_joint_array(landmarks)

    wrist = joint[0]
    relative_joint = joint - wrist

    palm_scale = np.mean(np.linalg.norm(joint[PALM_BASE_INDICES] - wrist, axis=1))
    if palm_scale == 0:
        palm_scale = 1e-6

    normalized_joint = relative_joint / palm_scale
    fingertip_distances = np.linalg.norm(joint[FINGERTIP_INDICES] - wrist, axis=1) / palm_scale
    angles = compute_angles(landmarks) / 180.0

    return np.concatenate([
        angles,
        normalized_joint.flatten(),
        fingertip_distances,
    ])
