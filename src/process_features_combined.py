import numpy as np
import pandas as pd
from pathlib import Path

from src.process_raw import center_on_wrist, normalize_coordinates
from src.process_features_angles import normalize_angles
from src.process_features_angles import extract_joint_angles, extract_pinch_angles, extract_curl_angles, extract_abduction_tip_angles, extract_abduction_mcp_angles
from src.process_features_angles import JOINT_ANGLE_COLUMNS, PINCH_ANGLE_COLUMNS, CURL_ANGLE_COLUMNS, ABDUCTION_TIP_ANGLE_COLUMNS, ABDUCTION_MCP_ANGLE_COLUMNS

from src.process_features_distances import extract_tip_to_wrist, extract_tip_to_thumb, extract_tip_to_tip, extract_pip_to_wrist, extract_mcp_to_wrist
from src.process_features_distances import TIP_TO_WRIST_COLUMNS, TIP_TO_THUMB_COLUMNS, TIP_TO_TIP_COLUMNS, PIP_TO_WRIST_COLUMNS, MCP_TO_WRIST_COLUMNS

def process_features_joint_pinch(new_row_coords):
    joint_angles = extract_joint_angles(new_row_coords)
    pinch_angles = extract_pinch_angles(new_row_coords)
    angles = joint_angles + pinch_angles

    normalized_angles = normalize_angles(angles)
    features = normalized_angles
    return features


def process_features_curl_pinch(new_row_coords):
    curl_angles = extract_curl_angles(new_row_coords)
    pinch_angles = extract_pinch_angles(new_row_coords)
    angles = curl_angles + pinch_angles

    normalized_angles = normalize_angles(angles)
    features = normalized_angles
    return features


def process_features_curl_abductionmcp(new_row_coords):
    curl_angles = extract_curl_angles(new_row_coords)
    abduction_mcp_angles = extract_abduction_mcp_angles(new_row_coords)
    angles = curl_angles + abduction_mcp_angles

    normalized_angles = normalize_angles(angles)
    features = normalized_angles
    return features


def process_features_curl_abductionmcp_tipttip(new_row_coords):
    curl_angles = extract_curl_angles(new_row_coords)
    abduction_mcp_angles = extract_abduction_mcp_angles(new_row_coords)
    distance_tttip = extract_tip_to_tip(new_row_coords)
    angles = curl_angles + abduction_mcp_angles

    normalized_angles = normalize_angles(angles)
    features = np.concatenate([normalized_angles, distance_tttip])
    return features


EXPERIMENTS = {
    "joint_pinch": (process_features_joint_pinch, JOINT_ANGLE_COLUMNS + PINCH_ANGLE_COLUMNS),
    "curl_pinch": (process_features_curl_pinch, CURL_ANGLE_COLUMNS + PINCH_ANGLE_COLUMNS),
    "curl_abductionmcp": (process_features_curl_abductionmcp, CURL_ANGLE_COLUMNS + ABDUCTION_MCP_ANGLE_COLUMNS),
    "curl_abductionmcp_tipttip": (process_features_curl_abductionmcp_tipttip, CURL_ANGLE_COLUMNS + ABDUCTION_MCP_ANGLE_COLUMNS + TIP_TO_TIP_COLUMNS),    
}


def process_landmarks(row_coords):
    new_row_coords = center_on_wrist(row_coords)
    new_row_coords = normalize_coordinates(new_row_coords)
    return new_row_coords


def main():
    SRC_PATH = Path(__file__).resolve().parent
    PROJECT_ROOT = SRC_PATH.parent
    INPUT_PATH = PROJECT_ROOT / 'data' / 'raw' / 'raw_landmarks.csv'
    OUTPUT_DIR = PROJECT_ROOT / 'data' / 'processed'

    df = pd.read_csv(INPUT_PATH)
    coords = df.drop(columns='label').values
    labels = df['label'].values
    processed_landmarks = [process_landmarks(row) for row in coords]
    
    for exp_name, (process_func, feature_columns) in EXPERIMENTS.items():
        output_file = OUTPUT_DIR / f'combined_{exp_name}.csv'
        processed_features = np.array([process_func(row) for row in processed_landmarks])

        processed_df = pd.DataFrame(processed_features, columns=feature_columns)
        processed_df.insert(0, 'label', labels)
        processed_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()