import csv
import mediapipe as mp
from pathlib import Path
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


def main():
    SRC_PATH = Path(__file__).resolve().parent
    PROJECT_ROOT = SRC_PATH.parent

    RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
    MODELS_DIR = PROJECT_ROOT / 'models'

    DATASET_DIR = RAW_DIR / 'hagrid-sample-30k-384p' / 'hagrid_30k'

    GESTURES_CONFIG = [
        {"label": "fist", "dir": DATASET_DIR / 'train_val_fist'},
        {"label": "palm", "dir": DATASET_DIR / 'train_val_palm'},
        {"label": "like", "dir": DATASET_DIR / 'train_val_like'},
        {"label": "peace", "dir": DATASET_DIR / 'train_val_peace'},
        {"label": "call", "dir": DATASET_DIR / 'train_val_call'},
        {"label": "ok", "dir": DATASET_DIR / 'train_val_ok'},
        {"label": "rock", "dir": DATASET_DIR / 'train_val_rock'},
    ]

    OUTPUT_DATA_DIR = RAW_DIR / 'raw_landmarks.csv'
    MEDIAPIPE_MODEL_PATH = MODELS_DIR / 'gesture_recognizer.task'

    base_options = mp_python.BaseOptions(model_asset_path=str(MEDIAPIPE_MODEL_PATH))
    running_mode = mp_vision.RunningMode.IMAGE
    options = mp_vision.GestureRecognizerOptions(
        base_options=base_options, 
        num_hands=1, 
        min_hand_detection_confidence=0.5, 
        running_mode=running_mode
    )
    mediapipe_recognizer = mp_vision.GestureRecognizer.create_from_options(options)

    valid_extensions = {'.jpg', '.jpeg', '.png'}
    images_to_extract = []
    extracted_data = []


    for config in GESTURES_CONFIG:
        label = config["label"]
        gesture_dir = config["dir"]
        if not gesture_dir.exists():
            print(f"[WARN] Couldn't find target dir: {gesture_dir}")
            continue

        images_to_extract = [p for p in gesture_dir.iterdir() if p.suffix.lower() in valid_extensions]
        if not images_to_extract:
            print("couldn't find any image")
            continue

        detected = 0
        missed = 0
        for img_path in images_to_extract:

            mp_image = mp.Image.create_from_file(str(img_path))
            result = mediapipe_recognizer.recognize(mp_image)

            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0] #these landmarks are already normalized
                row = [label]
                for landmark in hand_landmarks:
                    row.extend([landmark.x, landmark.y])
                extracted_data.append(row)
                detected += 1
            else:
                missed += 1

        print(f"[{label}] detected: {detected} | no hand: {missed}")
            

    if extracted_data:
        with open(OUTPUT_DATA_DIR, mode='w', newline='') as f:
            writer = csv.writer(f)
            header = ['label'] + [f'{axis}{i}' for i in range(21) for axis in ('x', 'y')]
            writer.writerow(header)
            writer.writerows(extracted_data)

    mediapipe_recognizer.close()


if __name__ == "__main__":
    main()