import os
import shutil
import random
from pathlib import Path
import sys

# Add root directory to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATASET_DIR

def split_dataset(source_dir, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Splits a single raw dataset folder containing class subdirectories into
    dataset/train, dataset/validation, and dataset/test directories without data leakage.
    """
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"[ERROR] Source dataset directory does not exist: {source_path}")
        return False

    random.seed(seed)

    train_dir = DATASET_DIR / 'train'
    val_dir = DATASET_DIR / 'validation'
    test_dir = DATASET_DIR / 'test'

    for dir_path in [train_dir, val_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    class_folders = [f for f in source_path.iterdir() if f.is_dir() and not f.name.startswith('.')]
    if not class_folders:
        print(f"[ERROR] No class directories found in {source_path}")
        return False

    print(f"Found {len(class_folders)} class categories. Preparing dataset splits...")

    total_images_copied = 0
    for class_folder in class_folders:
        class_name = class_folder.name
        images = [f for f in class_folder.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]

        if not images:
            continue

        random.shuffle(images)
        total_count = len(images)
        train_end = int(total_count * train_ratio)
        val_end = train_end + int(total_count * val_ratio)

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        for img_list, target_base in [(train_images, train_dir), (val_images, val_dir), (test_images, test_dir)]:
            target_class_dir = target_base / class_name
            target_class_dir.mkdir(parents=True, exist_ok=True)
            for img in img_list:
                shutil.copy2(img, target_class_dir / img.name)
                total_images_copied += 1

        print(f"  Class '{class_name}': {len(train_images)} Train | {len(val_images)} Val | {len(test_images)} Test")

    print(f"\n[SUCCESS] Dataset split complete! Total images processed: {total_images_copied}")
    return True

if __name__ == '__main__':
    raw_data_path = input("Enter path to raw unsplit dataset folder (e.g. C:/data/PlantVillage): ").strip()
    if raw_data_path:
        split_dataset(raw_data_path)
    else:
        print("No path provided. Usage: python training/split_dataset.py")
