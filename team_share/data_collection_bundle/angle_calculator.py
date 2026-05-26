import numpy as np


JOINT_LIST = [
    [0, 1, 2], [1, 2, 3], [2, 3, 4], [0, 5, 6], [5, 6, 7], [6, 7, 8],
    [0, 9, 10], [9, 10, 11], [10, 11, 12], [0, 13, 14], [13, 14, 15],
    [14, 15, 16], [0, 17, 18], [17, 18, 19], [18, 19, 20],
]
FINGERTIP_INDICES = [4, 8, 12, 16, 20]
PALM_BASE_INDICES = [5, 9, 13, 17]


def _landmarks_to_joint_array(landmarks):
    joint = np.zeros((21, 3))
    for index, landmark in enumerate(landmarks.landmark):
        joint[index] = [landmark.x, landmark.y, landmark.z]
    return joint


def compute_angles(landmarks):
    joint = _landmarks_to_joint_array(landmarks)
    first = joint[[indices[0] for indices in JOINT_LIST]]
    middle = joint[[indices[1] for indices in JOINT_LIST]]
    last = joint[[indices[2] for indices in JOINT_LIST]]
    vector_one = first - middle
    vector_two = last - middle
    norm_one = np.linalg.norm(vector_one, axis=1)[:, np.newaxis]
    norm_two = np.linalg.norm(vector_two, axis=1)[:, np.newaxis]
    norm_one[norm_one == 0] = 1e-6
    norm_two[norm_two == 0] = 1e-6
    vector_one = vector_one / norm_one
    vector_two = vector_two / norm_two
    dot_product = np.clip(np.einsum("nt,nt->n", vector_one, vector_two), -1.0, 1.0)
    return np.degrees(np.arccos(dot_product))


def compute_features(landmarks):
    joint = _landmarks_to_joint_array(landmarks)
    wrist = joint[0]
    relative_joint = joint - wrist
    palm_scale = np.mean(np.linalg.norm(joint[PALM_BASE_INDICES] - wrist, axis=1))
    if palm_scale == 0:
        palm_scale = 1e-6
    normalized_joint = relative_joint / palm_scale
    distances = np.linalg.norm(joint[FINGERTIP_INDICES] - wrist, axis=1) / palm_scale
    angles = compute_angles(landmarks) / 180.0
    return np.concatenate([angles, normalized_joint.flatten(), distances])
