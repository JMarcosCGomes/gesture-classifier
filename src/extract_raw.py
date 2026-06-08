import csv
import mediapipe as mp
from pathlib import Path
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


def main():
    SRC_PATH = Path(__file__).resolve().parent
    PROJECT_ROOT = SRC_PATH.parent

    RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
    PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
    MODELS_DIR = PROJECT_ROOT / 'models'

    DATASET_DIR = RAW_DIR / 'hagrid-sample-30k-384p' / 'hagrid_30k'

    GESTURES_DIRS = [
        DATASET_DIR / 'train_val_like',
        DATASET_DIR / 'train_val_palm',
        DATASET_DIR / 'train_val_peace',
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

    for gesture_dir in GESTURES_DIRS:
        if not gesture_dir.exists():
            print(f"[WARN] Couldn't find target dir: {gesture_dir}")
            continue

        images_to_extract = [p for p in gesture_dir.iterdir() if p.suffix.lower() in valid_extensions]

        if not images_to_extract:
            print("couldn't find any image")
        else:
            extracted_data = []
            
            for img_path in images_to_extract:
                print(f"Processing: {img_path}")

                mp_image = mp.Image.create_from_file(str(img_path))
                result = mediapipe_recognizer.recognize(mp_image)

                if result.hand_landmarks:
                    hand_landmarks = result.hand_landmarks[0] #these landmarks are already normalized
                    if result.gestures and result.gestures[0][0].category_name not in (None, 'None', ''):
                        hand_gesture = result.gestures[0][0].category_name
                    else:
                        hand_gesture = "Unknown"

                    row = [hand_gesture]
                    for landmark in hand_landmarks:
                        row.extend([landmark.x, landmark.y])

                    extracted_data.append(row)
                    print(f" Could detect: {hand_gesture}")
                else:
                    print(" Couldn't detect a hand with mediapipe.")
            

    if extracted_data:
        with open(OUTPUT_DATA_DIR, mode='w', newline='') as f:
            writer = csv.writer(f)
            header = ['label'] + [f'{axis}{i}' for i in range(21) for axis in ('x', 'y')]
            writer.writerow(header)
            writer.writerows(extracted_data)

    mediapipe_recognizer.close()


if __name__ == "__main__":
    main()