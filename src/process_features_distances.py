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


PAIRS_TIP_TO_WRIST = {
    "thumb": (4, 0), #
    "index": (8, 0), #
    "middle": (12, 0), #
    "ring": (16, 0), #
    "pinky": (20, 0), #
}

PAIRS_TIP_TO_THUMB = {
    "index": (8, 4), #
    "middle": (12, 4), #
    "ring": (16, 4), #
    "pinky": (20, 4), #
}

#
PAIRS_TIP_TO_TIP = {
    "index_middle": (8, 12), #
    "index_ring": (8, 16), #
    "index_pinky": (8, 20), #
    "middle_ring": (12, 16), #
    "middle_pinky": (12, 20), #
    "ring_pinky": (16, 20), #
}


PAIRS_PIP_TO_WRIST = {
    "thumb_mcp": (2, 0), #thumb eh diferente
    "index": (6, 0), #
    "middle": (10, 0), #
    "ring": (14, 0), #
    "pinky": (18, 0), #
}

PAIRS_MCP_TO_WRIST = {
    "thumb_cmc": (1, 0), #thumb eh diferente
    "index": (5, 0), #
    "middle": (9, 0), #
    "ring": (13, 0), #
    "pinky": (17, 0), #
}


#esses nomes tão feios, depois avalia uma forma melhor de nomear
TIP_TO_WRIST_COLUMNS = [f"dist_tip_wrist_{name}" for name in PAIRS_TIP_TO_WRIST]
TIP_TO_THUMB_COLUMNS = [f"dist_tip_thumb_{name}" for name in PAIRS_TIP_TO_THUMB]
TIP_TO_TIP_COLUMNS = [f"dist_tip_tip_{name}" for name in PAIRS_TIP_TO_TIP]
PIP_TO_WRIST_COLUMNS = [f"dist_pip_wrist_{name}" for name in PAIRS_PIP_TO_WRIST]
MCP_TO_WRIST_COLUMNS = [f"dist_mcp_wrist_{name}" for name in PAIRS_MCP_TO_WRIST]

DISTANCE_COLUMNS = TIP_TO_WRIST_COLUMNS + TIP_TO_THUMB_COLUMNS + TIP_TO_TIP_COLUMNS + PIP_TO_WRIST_COLUMNS + MCP_TO_WRIST_COLUMNS

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


def extract_tip_to_thumb(row_coords):
    distances = []
    for a, b in PAIRS_TIP_TO_THUMB.values():
        distances.append(euclidean_distance(row_coords, a, b))
    return distances


def extract_tip_to_tip(row_coords):
    distances = []
    for a, b in PAIRS_TIP_TO_TIP.values():
        distances.append(euclidean_distance(row_coords, a, b))
    return distances


def extract_pip_to_wrist(row_coords):
    distances = []
    for a, b in PAIRS_PIP_TO_WRIST.values():
        distances.append(euclidean_distance(row_coords, a, b))
    return distances


def extract_mcp_to_wrist(row_coords):
    distances = []
    for a, b in PAIRS_MCP_TO_WRIST.values():
        distances.append(euclidean_distance(row_coords, a, b))
    return distances

def process_features(row_coords):
    new_row_coords = center_on_wrist(row_coords)
    new_row_coords = normalize_coordinates(new_row_coords)

    ttw_distances = extract_tip_to_wrist(new_row_coords)
    ttthumb_distances = extract_tip_to_thumb(new_row_coords)
    tttip_distances = extract_tip_to_tip(new_row_coords)
    ptw_distances = extract_pip_to_wrist(new_row_coords)
    mtw_distances = extract_mcp_to_wrist(new_row_coords)

    distances = ttw_distances + ttthumb_distances + tttip_distances + ptw_distances + mtw_distances

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