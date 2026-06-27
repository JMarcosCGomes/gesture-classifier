import numpy as np
import pandas as pd
from pathlib import Path


def center_on_wrist(row_coords):
    xs = row_coords[0::2]
    ys = row_coords[1::2]
    xs = xs - xs[0]
    ys = ys - ys[0]

    new_row_coords = np.stack([xs, ys], axis=1).flatten()
    return new_row_coords


def normalize_coordinates(row_coords):
    xs = row_coords[0::2]
    ys = row_coords[1::2]
    scale = np.max(np.abs(np.concatenate([xs, ys])))
    if scale == 0:
        raise ValueError("Scale is zero, landmark data may be corrupted.")
    xs = xs / scale
    ys = ys / scale

    new_row_coords = np.stack([xs, ys], axis=1).flatten()
    return new_row_coords


# ---------------------------------------------------------------------

"""
(ref: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker )
WRIST (0)
├── THUMB/POLEGAR:  CMC(1) → MCP(2) → IP(3)  → TIP(4)
├── INDEX/INDICADOR:  MCP(5) → PIP(6) → DIP(7) → TIP(8)
├── MIDDLE/MEDIO: MCP(9) → PIP(10)→ DIP(11)→ TIP(12)
├── RING/ANELAR:   MCP(13)→ PIP(14)→ DIP(15)→ TIP(16)
└── PINKY/MÍNIMO:  MCP(17)→ PIP(18)→ DIP(19)→ TIP(20)
"""

TRIPLETS_JOINT = {
    # Polegar
    "thumb_cmc": (0, 1, 2), #1
    "thumb_mcp":  (1,  2,  3), #2
    "thumb_ip":   (2,  3,  4), #3
    # Indicador
    "index_mcp":  (5,  6,  7), #6
    "index_pip":  (6,  7,  8), #7
    # Médio
    "middle_mcp": (9,  10, 11), #10
    "middle_pip": (10, 11, 12), #11
    # Anelar
    "ring_mcp":   (13, 14, 15), #14
    "ring_pip":   (14, 15, 16), #15
    # Mínimo
    "pinky_mcp":  (17, 18, 19), #18
    "pinky_pip":  (18, 19, 20), #19
}


TRIPLETS_ABDUCTION = {
    # every vertice is 0
    "thumb_index_mcp": (2, 0, 5), #
    "index_middle_mcp":  (5, 0, 9), #
    "middle_ring_mcp": (9, 0, 13), #
    "ring_pinky_mcp": (13, 0, 17), #
}


TRIPLETS_PINCH = {
    "thumb_index_tip": (4, 0, 8), #
    "thumb_middle_tip": (4, 0, 12), #
    "thumb_ring_tip": (4, 0, 16), #
    "thumb_pinky_tip": (4, 0, 20), #
}

#honestamente eu não acho que precisa de tudo isso, o abduction pega o bastante pros que nn são do thumb_algo
TRIPLETS_PINCH_ALL = {
    "thumb_index_tip": (4, 0, 8), #
    "thumb_middle_tip": (4, 0, 12), #
    "thumb_ring_tip": (4, 0, 16), #
    "thumb_pinky_tip": (4, 0, 20), #
    "index_middle_tip": (8, 0, 12), #
    "index_ring_tip": (8, 0, 16), #
    "index_pinky_tip": (8, 0, 20), #
    "middle_ring_tip": (12, 0, 16), #
    "middle_pinky_tip": (12, 0, 20), #
    "ring_pinky_tip": (16, 0, 20), #
}

 
JOINT_ANGLE_COLUMNS = [f"joint_angle_{name}" for name in TRIPLETS_JOINT]
ABDUCTION_ANGLE_COLUMNS = [f"abduction_angle_{name}" for name in TRIPLETS_ABDUCTION]
PINCH_ANGLE_COLUMNS = [f"pinch_angle_{name}" for name in TRIPLETS_PINCH]
ANGLES_COLUMNS = JOINT_ANGLE_COLUMNS + ABDUCTION_ANGLE_COLUMNS + PINCH_ANGLE_COLUMNS


def get_point(row_coords, idx):
    point = np.array([row_coords[idx * 2], row_coords[idx * 2 + 1]])
    return point

def angle_between(row_coords, a, b, c):
    pa, pb, pc = get_point(row_coords, a), get_point(row_coords, b), get_point(row_coords, c)
    v1 = pa - pb
    v2 = pc - pb
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    angle = float(np.degrees(np.arccos(cos_angle)))
    return angle


def extract_joint_angles(row_coords):
    angles = []
    for a, b, c in TRIPLETS_JOINT.values():
        angles.append(angle_between(row_coords, a, b, c))
    return angles


def extract_abduction_angles(row_coords):
    angles = []
    for a, b, c in TRIPLETS_ABDUCTION.values():
        angles.append(angle_between(row_coords, a, b, c))
    return angles


def extract_pinch_angles(row_coords):
    angles = []
    for a, b, c in TRIPLETS_PINCH.values():
        angles.append(angle_between(row_coords, a, b, c))
    return angles


def normalize_angles(angles):
    normalized_angles = np.array(angles) / 180.0
    return normalized_angles


def process_features(row_coords):
    new_row_coords = center_on_wrist(row_coords)
    new_row_coords = normalize_coordinates(new_row_coords)

    joint_angles = extract_joint_angles(new_row_coords)
    abduction_angles = extract_abduction_angles(new_row_coords)
    pinch_angles = extract_pinch_angles(new_row_coords)
    angles = joint_angles + abduction_angles + pinch_angles
    normalized_angles = normalize_angles(angles)

    return normalized_angles


def main():
    SRC_PATH = Path(__file__).resolve().parent
    PROJECT_ROOT = SRC_PATH.parent
    INPUT_PATH = PROJECT_ROOT / 'data' / 'raw' / 'raw_landmarks.csv'
    OUTPUT_PATH = PROJECT_ROOT / 'data' / 'processed' / 'featured_landmarks.csv'

    df = pd.read_csv(INPUT_PATH)
    coords = df.drop(columns='label').values
    processed = np.array([process_features(row) for row in coords])
    
    feature_columns = ANGLES_COLUMNS
    processed_df = pd.DataFrame(processed, columns=feature_columns)
    processed_df.insert(0, 'label', df['label'].values)
    processed_df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()