import numpy as np
import pandas as pd
from pathlib import Path

from src.process_raw import center_on_wrist, normalize_coordinates

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
#INDICES_TO_KEEP = [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20]

TRIPLETS_CURL = {
    "thumb": (0, 1, 4), #
    "index": (0, 5, 8), #
    "middle": (0, 9, 12), #
    "ring": (0, 13, 16), #
    "pinky": (0, 17, 20), #
}


TRIPLETS_PINCH = {
    "thumb_index_tip": (4, 0, 8), #
    "thumb_middle_tip": (4, 0, 12), #
    "thumb_ring_tip": (4, 0, 16), #
    "thumb_pinky_tip": (4, 0, 20), #
}


TRIPLETS_MCP_ABDUCTION = {
    # every vertice is 0
    "thumb_index_mcp": (2, 0, 5), #
    "index_middle_mcp":  (5, 0, 9), #
    "middle_ring_mcp": (9, 0, 13), #
    "ring_pinky_mcp": (13, 0, 17), #
}

TRIPLETS_TIP_ABDUCTION = {
    "thumb_index_tip": (4, 0, 8), #
    "index_middle_tip": (8, 0, 12), #
    "middle_ring_tip": (12, 0, 16), #
    "ring_pinky_tip": (16, 0, 20), #
}

 
CURL_ANGLE_COLUMNS = [f"angle_curl_{name}" for name in TRIPLETS_CURL]
PINCH_ANGLE_COLUMNS = [f"angle_pinch_{name}" for name in TRIPLETS_PINCH]
ABDUCTION_MCP_ANGLE_COLUMNS = [f"angle_abduction_mcp_{name}" for name in TRIPLETS_MCP_ABDUCTION]
ABDUCTION_TIP_ANGLE_COLUMNS = [f"angle_abduction_tip_{name}" for name in TRIPLETS_TIP_ABDUCTION]

#ANGLES_REDUCED_COLUMNS = CURL_ANGLE_COLUMNS + PINCH_ANGLE_COLUMNS + ABDUCTION_MCP_ANGLE_COLUMNS + ABDUCTION_TIP_ANGLE_COLUMNS
ANGLES_REDUCED_COLUMNS = CURL_ANGLE_COLUMNS + PINCH_ANGLE_COLUMNS

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


def extract_curl_angles(row_coords):
    angles = []
    for a, b, c in TRIPLETS_CURL.values():
        angles.append(angle_between(row_coords, a, b, c))
    return angles


def extract_pinch_angles(row_coords):
    angles = []
    for a, b, c in TRIPLETS_PINCH.values():
        angles.append(angle_between(row_coords, a, b, c))
    return angles


def extract_abduction_mcp_angles(row_coords):
    angles = []
    for a, b, c in TRIPLETS_MCP_ABDUCTION.values():
        angles.append(angle_between(row_coords, a, b, c))
    return angles


def extract_abduction_tip_angles(row_coords):
    angles = []
    for a, b, c in TRIPLETS_TIP_ABDUCTION.values():
        angles.append(angle_between(row_coords, a, b, c))
    return angles


def normalize_angles(angles):
    normalized_angles = np.array(angles) / 180.0
    return normalized_angles


def process_features(row_coords):
    new_row_coords = center_on_wrist(row_coords)
    new_row_coords = normalize_coordinates(new_row_coords)

    curl_angles = extract_curl_angles(new_row_coords)
    pinch_angles = extract_pinch_angles(new_row_coords)
    
    #abduction_mcp_angles = extract_abduction_mcp_angles(new_row_coords)
    #abduction_tip_angles = extract_abduction_tip_angles(new_row_coords)
    #angles = curl_angles + pinch_angles + abduction_mcp_angles + abduction_tip_angles
    
    angles = curl_angles + pinch_angles
    normalized_angles_reduced = normalize_angles(angles)

    return normalized_angles_reduced


def main():
    SRC_PATH = Path(__file__).resolve().parent
    PROJECT_ROOT = SRC_PATH.parent
    INPUT_PATH = PROJECT_ROOT / 'data' / 'raw' / 'raw_landmarks.csv'
    OUTPUT_PATH = PROJECT_ROOT / 'data' / 'processed' / 'featured_angles_reduced.csv'

    df = pd.read_csv(INPUT_PATH)
    coords = df.drop(columns='label').values
    processed = np.array([process_features(row) for row in coords])
    
    feature_columns = ANGLES_REDUCED_COLUMNS
    processed_df = pd.DataFrame(processed, columns=feature_columns)
    processed_df.insert(0, 'label', df['label'].values)
    processed_df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()