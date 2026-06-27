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


PAIRS_TIP_TO_WRIST = {
    "thumb_tip_wrist": (4, 0), #
    "index_tip_wrist": (8, 0), #
    "middle_tip_wrist": (12, 0), #
    "ring_tip_wrist": (16, 0), #
    "pinky_tip_wrist": (20, 0), #
}

TIP_TO_WRIST_COLUMNS = [f"dist_tip_wrist_{name}" for name in PAIRS_TIP_TO_WRIST]

DISTANCE_COLUMNS = PAIRS_TIP_TO_WRIST

# ---------------------------------------------------------------------

def get_point(row_coords, idx):
    point = np.array([row_coords[idx * 2], row_coords[idx * 2 + 1]])
    return point

def euclidean_distance(row_coords, a, b):
    pa, pb = get_point(row_coords, a), get_point(row_coords, b)
    distance = float(np.linalg.norm(pa - pb))
    return distance

def extract_tip_to_wrist(row_coords):
    distances = []
    for a, b in PAIRS_TIP_TO_WRIST.values():
        distances.append(euclidean_distance(row_coords, a, b))
    return distances


def process_features(row_coords):
    new_row_coords = center_on_wrist(row_coords)
    new_row_coords = normalize_coordinates(new_row_coords)
    
    ttw_distances = extract_tip_to_wrist(row_coords)

    distances = ttw_distances

    return distances


def main():
    SRC_PATH = Path(__file__).resolve().parent
    PROJECT_ROOT = SRC_PATH.parent
    INPUT_PATH = PROJECT_ROOT / 'data' / 'raw' / 'raw_landmarks.csv'
    OUTPUT_PATH = PROJECT_ROOT / 'data' / 'processed' / 'featured_distances.csv'

    df = pd.read_csv(INPUT_PATH)
    coords = df.drop(columns='label').values
    processed = np.array([process_features(row) for row in coords])
    
    feature_columns = DISTANCE_COLUMNS
    processed_df = pd.DataFrame(processed, columns=feature_columns)
    processed_df.insert(0, 'label', df['label'].values)
    processed_df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()