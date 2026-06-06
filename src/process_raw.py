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


def normalize(row_coords):
    xs = row_coords[0::2]
    ys = row_coords[1::2]
    scale = np.max(np.abs(np.concatenate([xs, ys])))
    if scale == 0:
        raise ValueError("Scale is zero, landmark data may be corrupted.")
    xs = xs / scale
    ys = ys / scale

    new_row_coords = np.stack([xs, ys], axis=1).flatten()
    return new_row_coords


def process_raw(row_coords):
    new_row_coords = center_on_wrist(row_coords)
    new_row_coords = normalize(new_row_coords)
    return new_row_coords


def main():
    SRC_PATH = Path(__file__).resolve().parent
    PROJECT_ROOT = SRC_PATH.parent
    INPUT_PATH = PROJECT_ROOT / 'data' / 'raw' / 'raw_landmarks.csv'
    OUTPUT_PATH = PROJECT_ROOT / 'data' / 'processed' / 'processed_landmarks.csv'

    df = pd.read_csv(INPUT_PATH)
    coords = df.drop(columns='label').values
    processed = np.array([process_raw(row) for row in coords])
    processed_df = pd.DataFrame(processed, columns=df.columns[1:])
    processed_df.insert(0, 'label', df['label'].values)
    processed_df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()