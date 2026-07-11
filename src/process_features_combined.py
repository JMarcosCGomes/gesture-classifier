import numpy as np
import pandas as pd
from pathlib import Path

from src.process_raw import center_on_wrist, normalize_coordinates
from src.process_features_angles import extract_joint_angles, extract_pinch_angles, normalize_angles
from src.process_features_angles import JOINT_ANGLE_COLUMNS, PINCH_ANGLE_COLUMNS

COMBINED_COLUMNS = JOINT_ANGLE_COLUMNS + PINCH_ANGLE_COLUMNS

def process_features(row_coords):
    new_row_coords = center_on_wrist(row_coords)
    new_row_coords = normalize_coordinates(new_row_coords)
    
    joint_angles = extract_joint_angles(new_row_coords)
    pinch_angles = extract_pinch_angles(new_row_coords)
    angles = joint_angles + pinch_angles
    normalized_angles = normalize_angles(angles)

    #features = np.concatenate([normalized_angles])
    features = normalized_angles

    return features


def main():
    SRC_PATH = Path(__file__).resolve().parent
    PROJECT_ROOT = SRC_PATH.parent
    INPUT_PATH = PROJECT_ROOT / 'data' / 'raw' / 'raw_landmarks.csv'
    OUTPUT_PATH = PROJECT_ROOT / 'data' / 'processed' / 'featured_combined.csv'

    df = pd.read_csv(INPUT_PATH)
    coords = df.drop(columns='label').values
    processed = np.array([process_features(row) for row in coords])
    
    feature_columns = COMBINED_COLUMNS
    processed_df = pd.DataFrame(processed, columns=feature_columns)
    processed_df.insert(0, 'label', df['label'].values)
    processed_df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()