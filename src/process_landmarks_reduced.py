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

#quero manter o 0, mcps e tips
INDICES_TO_KEEP = [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20]
reduced_columns = []
for idx in INDICES_TO_KEEP:
    reduced_columns.extend([f"x{idx}", f"x{idx}"])


def extract_reduced_landmarks(row_coords):
    reduced_coords = []
    for idx in INDICES_TO_KEEP:
        reduced_coords.append(row_coords[idx*2])
        reduced_coords.append(row_coords[idx*2 + 1])
    return reduced_coords


def process_landmarks_reduced(row_coords):
    new_row_coords = center_on_wrist(row_coords)
    new_row_coords = normalize_coordinates(new_row_coords)
    reduced_landmarks = extract_reduced_landmarks(new_row_coords)

    return reduced_landmarks


def main():
    SRC_PATH = Path(__file__).resolve().parent
    PROJECT_ROOT = SRC_PATH.parent
    INPUT_PATH = PROJECT_ROOT / 'data' / 'raw' / 'raw_landmarks.csv'
    OUTPUT_PATH = PROJECT_ROOT / 'data' / 'processed' / 'landmarks_reduced.csv'

    df = pd.read_csv(INPUT_PATH)
    coords = df.drop(columns='label').values
    processed = np.array([process_landmarks_reduced(row) for row in coords])
    
    processed_df = pd.DataFrame(processed, columns=reduced_columns)
    processed_df.insert(0, 'label', df['label'].values)
    processed_df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()