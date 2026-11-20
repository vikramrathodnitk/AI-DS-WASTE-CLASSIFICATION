
"""
Dataset Preparation and Validation
Project: Waste Classification Using CNN

Purpose:
1. Check whether the dataset folders exist.
2. Check all six waste categories.
3. Count images in training and validation folders.
4. Display a dataset summary before model training.

Expected dataset structure:

data/
└── public/
    ├── train/
    │   ├── cardboard/
    │   ├── glass/
    │   ├── metal/
    │   ├── paper/
    │   ├── plastic/
    │   └── trash/
    │
    └── val/
        ├── cardboard/
        ├── glass/
        ├── metal/
        ├── paper/
        ├── plastic/
        └── trash/
"""

from pathlib import Path


# --------------------------------------------------
# 1. PROJECT AND DATASET PATHS
# --------------------------------------------------

# Get the folder in which this Python file is located.
PROJECT_DIR = Path(__file__).resolve().parent

# Main dataset folder.
DATASET_DIR = PROJECT_DIR / "data" / "public"

# Expected waste categories.
EXPECTED_CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

# Supported image formats.
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

# Expected image counts based on our dataset.
EXPECTED_TRAIN_IMAGES = 400
EXPECTED_VAL_IMAGES = 100


# --------------------------------------------------
# 2. COUNT IMAGES
# --------------------------------------------------

def count_images(folder):
    """
    Count image files inside a folder and its subfolders.

    Parameter:
        folder: Path of the folder to examine.

    Returns:
        int: Total number of image files.
    """

    return sum(
        1
        for file in folder.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    )


# --------------------------------------------------
# 3. CHECK DATASET FOLDER STRUCTURE
# --------------------------------------------------

def check_dataset_structure():
    """
    Check whether train, val and all six class folders exist.

    Returns:
        bool: True if the folder structure is correct,
              otherwise False.
    """

    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "val"

    # Check training and validation folders.
    if not train_dir.is_dir():
        print("ERROR: Training folder is missing.")
        print(f"Expected location: {train_dir}")
        return False

    if not val_dir.is_dir():
        print("ERROR: Validation folder is missing.")
        print(f"Expected location: {val_dir}")
        return False

    # Check the six class folders in both datasets.
    for split_name, split_dir in [
        ("Training", train_dir),
        ("Validation", val_dir),
    ]:

        actual_classes = sorted(
            folder.name
            for folder in split_dir.iterdir()
            if folder.is_dir()
        )

        if actual_classes != EXPECTED_CLASSES:
            print(f"ERROR: Incorrect class folders in {split_name} data.")
            print(f"Expected: {EXPECTED_CLASSES}")
            print(f"Found: {actual_classes}")
            return False

    return True


# --------------------------------------------------
# 4. DISPLAY DATASET SUMMARY
# --------------------------------------------------

def display_dataset_summary():
    """
    Display image counts for every waste category.
    """

    print("\n" + "=" * 55)
    print("WASTE CLASSIFICATION - DATASET SUMMARY")
    print("=" * 55)

    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "val"

    total_train = 0
    total_val = 0

    print("\nTRAINING DATA")
    print("-" * 55)

    for class_name in EXPECTED_CLASSES:
        class_dir = train_dir / class_name
        image_count = count_images(class_dir)

        total_train += image_count

        print(f"{class_name.capitalize():15} : {image_count} images")

        if image_count != EXPECTED_TRAIN_IMAGES:
            print(
                f"  WARNING: Expected {EXPECTED_TRAIN_IMAGES} "
                f"images in this category."
            )

    print(f"\nTotal training images: {total_train}")

    print("\nVALIDATION DATA")
    print("-" * 55)

    for class_name in EXPECTED_CLASSES:
        class_dir = val_dir / class_name
        image_count = count_images(class_dir)

        total_val += image_count

        print(f"{class_name.capitalize():15} : {image_count} images")

        if image_count != EXPECTED_VAL_IMAGES:
            print(
                f"  WARNING: Expected {EXPECTED_VAL_IMAGES} "
                f"images in this category."
            )

    print(f"\nTotal validation images: {total_val}")

    print("\n" + "-" * 55)
    print(f"Total dataset images: {total_train + total_val}")
    print("-" * 55)

    print("\nDataset class order:")

    for index, class_name in enumerate(EXPECTED_CLASSES):
        print(f"{index} -> {class_name}")

    print("\nDataset folder structure is correct.")
    print("Dataset checking completed.")

    print("\nNEXT STEP:")
    print("Run the CNN training script:")
    print("python train_template.py --data_dir data/public")

    print("=" * 55)


# --------------------------------------------------
# 5. MAIN FUNCTION
# --------------------------------------------------

def main():
    """
    Main function to validate and summarize the dataset.

    Returns:
        int: 0 if successful, 1 if dataset validation fails.
    """

    print("\nWaste Classification Using CNN")
    print("Dataset Preparation and Validation")

    print(f"\nProject folder: {PROJECT_DIR}")
    print(f"Dataset folder: {DATASET_DIR}")

    if not DATASET_DIR.is_dir():
        print("\nERROR: Dataset folder does not exist.")
        print("Please place your dataset inside data/public/.")
        return 1

    if not check_dataset_structure():
        print("\nDataset validation failed.")
        return 1

    display_dataset_summary()

    return 0


# --------------------------------------------------
# 6. PROGRAM ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(main())
