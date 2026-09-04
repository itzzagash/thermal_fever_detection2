"""
MANUAL THERMAL IMAGE LABELING TOOL
------------------------------------
Since the downloaded Kaggle dataset has no fever/normal labels, this tool
lets you quickly sort images by eye. It shows one image at a time; you
press a key to file it into the correct folder.

CONTROLS:
    N  -> Normal temperature      (saves to dataset/train/normal or val/normal)
    F  -> Fever / elevated temperature (saves to dataset/train/fever or val/fever)
    S  -> Skip this image (don't use it)
    Q  -> Quit and save progress

Roughly 1 out of every 5 labeled images automatically goes to the
validation set instead of training, to give you a val split for free.

HOW TO RUN:
    1. Edit SOURCE_DIR below to point to the folder containing the
       downloaded thermal images (e.g. the "Single Person" folder you
       extracted from the Kaggle zip).
    2. From the thermal_fever_detection project folder, run:
           python label_tool.py
    3. Label as many images as you reasonably can (100-300 is a good start).
    4. Press Q any time to stop -- progress is saved as you go.
"""

import os
import shutil
import random
import cv2

# ----------------------------------------------------------------------
# EDIT THIS: point to the folder with your extracted thermal images
# ----------------------------------------------------------------------
SOURCE_DIR = r"C:\Users\gunaa\Downloads"  # <-- CHANGE THIS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
VAL_SPLIT_RATIO = 0.2  # ~20% of labeled images go to validation

TRAIN_NORMAL = os.path.join(DATASET_DIR, "train", "normal")
TRAIN_FEVER = os.path.join(DATASET_DIR, "train", "fever")
VAL_NORMAL = os.path.join(DATASET_DIR, "val", "normal")
VAL_FEVER = os.path.join(DATASET_DIR, "val", "fever")

for d in [TRAIN_NORMAL, TRAIN_FEVER, VAL_NORMAL, VAL_FEVER]:
    os.makedirs(d, exist_ok=True)

VALID_EXT = (".jpg", ".jpeg", ".png")


def already_labeled_files():
    """Returns the set of filenames already sorted, so re-running the tool
    skips images you've already labeled."""
    done = set()
    for d in [TRAIN_NORMAL, TRAIN_FEVER, VAL_NORMAL, VAL_FEVER]:
        for f in os.listdir(d):
            done.add(f)
    return done


def gather_source_images(source_dir):
    images = []
    for root, _, files in os.walk(source_dir):
        for f in files:
            if f.lower().endswith(VALID_EXT):
                images.append(os.path.join(root, f))
    return images


def main():
    if not os.path.isdir(SOURCE_DIR):
        print(f"SOURCE_DIR does not exist: {SOURCE_DIR}")
        print("Edit the SOURCE_DIR variable at the top of label_tool.py first.")
        return

    all_images = gather_source_images(SOURCE_DIR)
    if not all_images:
        print(f"No .jpg/.jpeg/.png images found under: {SOURCE_DIR}")
        return

    labeled_names = already_labeled_files()
    remaining = [p for p in all_images if os.path.basename(p) not in labeled_names]
    random.shuffle(remaining)

    print(f"Found {len(all_images)} total images.")
    print(f"{len(labeled_names)} already labeled previously.")
    print(f"{len(remaining)} remaining to label.")
    print()
    print("Controls: [N] Normal   [F] Fever   [S] Skip   [Q] Quit")
    print()

    window = "Label thermal image - N=Normal  F=Fever  S=Skip  Q=Quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 500, 500)

    labeled_count = 0

    for path in remaining:
        img = cv2.imread(path)
        if img is None:
            continue

        display = cv2.resize(img, (500, 500))
        cv2.imshow(window, display)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), ord("Q")):
            print("Quitting. Progress saved.")
            break

        elif key in (ord("s"), ord("S")):
            continue

        elif key in (ord("n"), ord("N"), ord("f"), ord("F")):
            is_val = random.random() < VAL_SPLIT_RATIO
            if key in (ord("n"), ord("N")):
                dest_dir = VAL_NORMAL if is_val else TRAIN_NORMAL
                label = "normal"
            else:
                dest_dir = VAL_FEVER if is_val else TRAIN_FEVER
                label = "fever"

            dest_path = os.path.join(dest_dir, os.path.basename(path))
            shutil.copy2(path, dest_path)
            labeled_count += 1
            print(f"[{labeled_count}] {os.path.basename(path)} -> {label} "
                  f"({'val' if is_val else 'train'})")

        else:
            print("Unrecognized key. Use N, F, S, or Q.")
            continue

    cv2.destroyAllWindows()

    print()
    print(f"Session complete. Labeled {labeled_count} new images.")
    print(f"Train normal: {len(os.listdir(TRAIN_NORMAL))}")
    print(f"Train fever:  {len(os.listdir(TRAIN_FEVER))}")
    print(f"Val normal:   {len(os.listdir(VAL_NORMAL))}")
    print(f"Val fever:    {len(os.listdir(VAL_FEVER))}")
    print()
    print("Run this script again any time to keep labeling more images.")
    print("Once you have a reasonable amount, run: python model/train_model.py")


if __name__ == "__main__":
    main()
