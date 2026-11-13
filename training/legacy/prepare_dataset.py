import os
import json
import random
import shutil
import zipfile
from datetime import datetime
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────
RAW_DIR = "/Users/hiteshprajapathi/Desktop/Naruto_Run_datacollection/Pure Naruto Hand Sign Data"
OUTPUT_DIR = "/Users/hiteshprajapathi/Desktop/Naruto_Run_datacollection/cv_model/data/prepared_dataset"
ZIP_PATH = "/Users/hiteshprajapathi/Desktop/Naruto_Run_datacollection/cv_model/data/prepared_dataset.zip"

IMAGE_SIZE = 224
RANDOM_SEED = 42
BALANCE_TARGET = 220
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CLASSES = [
    "bird", "boar", "dog", "dragon", "hare", "horse",
    "monkey", "ox", "ram", "rat", "snake", "tiger", "zero"
]

def scan_raw_dataset(raw_dir):
    merged_pool = {c: [] for c in CLASSES}
    for split in ['train', 'test']:
        split_path = os.path.join(raw_dir, split)
        if not os.path.exists(split_path):
            continue
        for c in CLASSES:
            c_path = os.path.join(split_path, c)
            if not os.path.isdir(c_path):
                continue
            for f in os.listdir(c_path):
                if f.startswith('.'):
                    continue
                if not (f.lower().endswith('.png') or f.lower().endswith('.jpg') or f.lower().endswith('.jpeg')):
                    continue
                merged_pool[c].append(os.path.join(c_path, f))
    return merged_pool

def normalize_image(img_path, size=224):
    try:
        img = Image.open(img_path)
        img.verify()
    except Exception as e:
        print(f"Error verifying {img_path}: {e}")
        return None

    try:
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        w, h = img.size
        crop_size = min(w, h)
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        img = img.crop((left, top, left + crop_size, top + crop_size))
        
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def apply_augmentation(img):
    # 1. Random Rotation (-15 to 15)
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=(0,0,0))
    
    # 2. Color Jitter
    if random.random() < 0.8:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
        img = ImageEnhance.Color(img).enhance(random.uniform(0.8, 1.2))
        
    # 3. Gaussian Blur
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.0)))
        
    # 4. Random Resized Crop (Zoom)
    scale = random.uniform(0.85, 1.0)
    w, h = img.size
    new_w = int(w * scale)
    new_h = int(h * scale)
    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)
    img = img.crop((left, top, left + new_w, top + new_h))
    img = img.resize((w, h), Image.Resampling.LANCZOS)
    
    # 5. Additive Gaussian Noise
    if random.random() < 0.2:
        img_arr = np.array(img).astype(np.float32) / 255.0
        noise = np.random.normal(0, random.uniform(0.01, 0.03), img_arr.shape)
        img_arr = np.clip(img_arr + noise, 0.0, 1.0) * 255.0
        img = Image.fromarray(img_arr.astype(np.uint8))
        
    return img

def main():
    print("═══════════════════════════════════════════════════════════")
    print("  NarutoCV Dataset Preparation — Component 1")
    print("═══════════════════════════════════════════════════════════\n")
    
    # 1. Scan
    print("[1/8] Scanning raw dataset...")
    merged_pool = scan_raw_dataset(RAW_DIR)
    total_imgs = sum(len(files) for files in merged_pool.values())
    print(f"      Found {len(CLASSES)} classes, {total_imgs} images total")
    
    # 2. Normalize
    print("\n[2/8] Normalizing images (RGBA→RGB, center-crop, resize 224×224)...")
    normalized_pool = {c: [] for c in CLASSES}
    for c in CLASSES:
        for f in tqdm(merged_pool[c], desc=f"      {c:>8}", leave=False):
            img = normalize_image(f, size=IMAGE_SIZE)
            if img:
                normalized_pool[c].append(img)
    
    # 3. Split
    print("\n[3/8] Stratified split (80/10/10, seed=42)...")
    all_imgs = []
    all_labels = []
    for c in CLASSES:
        all_imgs.extend(normalized_pool[c])
        all_labels.extend([c] * len(normalized_pool[c]))
        
    train_x, holdout_x, train_y, holdout_y = train_test_split(
        all_imgs, all_labels, test_size=(VAL_RATIO + TEST_RATIO), stratify=all_labels, random_state=RANDOM_SEED
    )
    
    val_x, test_x, val_y, test_y = train_test_split(
        holdout_x, holdout_y, test_size=(TEST_RATIO / (VAL_RATIO + TEST_RATIO)), stratify=holdout_y, random_state=RANDOM_SEED
    )
    
    train_dict = {c: [] for c in CLASSES}
    for x, y in zip(train_x, train_y): train_dict[y].append(x)
    val_dict = {c: [] for c in CLASSES}
    for x, y in zip(val_x, val_y): val_dict[y].append(x)
    test_dict = {c: [] for c in CLASSES}
    for x, y in zip(test_x, test_y): test_dict[y].append(x)
    
    print(f"      Train: {len(train_x)} | Val: {len(val_x)} | Test: {len(test_x)}")
    
    # 4/5. Augment and Balance
    print("\n[4 & 5/8] Augmenting & balancing training set (target: 220/class)...")
    aug_stats = {}
    for c in tqdm(CLASSES, desc="      Balancing"):
        originals = train_dict[c]
        deficit = BALANCE_TARGET - len(originals)
        aug_count = 0
        if deficit > 0:
            for i in range(deficit):
                src = originals[i % len(originals)]
                aug = apply_augmentation(src)
                train_dict[c].append(aug)
                aug_count += 1
        aug_stats[c] = aug_count
        
    # 6. Save Splits
    print("\n[6/8] Saving splits to cv_model/data/prepared_dataset/...")
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        
    for split_name, split_data in [('train', train_dict), ('val', val_dict), ('test', test_dict)]:
        split_dir = os.path.join(OUTPUT_DIR, split_name)
        for c in CLASSES:
            c_dir = os.path.join(split_dir, c)
            os.makedirs(c_dir, exist_ok=True)
            for i, img in enumerate(tqdm(split_data[c], desc=f"      Saving {split_name}/{c}", leave=False)):
                suffix = "_aug" if split_name == 'train' and i >= len(split_data[c]) - aug_stats[c] else ""
                filename = f"{c}_{split_name}{suffix}_{i:05d}.png"
                img.save(os.path.join(c_dir, filename), "PNG")
                
    # 7. Metadata
    print("\n[7/8] Writing metadata...")
    stats = {
        "created_at": datetime.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "image_size": [IMAGE_SIZE, IMAGE_SIZE],
        "splits": {
            "train": {
                "original_count": len(train_x),
                "augmented_count": sum(aug_stats.values()),
                "total_count": sum(len(train_dict[c]) for c in CLASSES),
                "per_class": {c: len(train_dict[c]) for c in CLASSES}
            },
            "val": {
                "total_count": len(val_x),
                "per_class": {c: len(val_dict[c]) for c in CLASSES}
            },
            "test": {
                "total_count": len(test_x),
                "per_class": {c: len(test_dict[c]) for c in CLASSES}
            }
        }
    }
    
    with open(os.path.join(OUTPUT_DIR, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
        
    class_map = {c: i for i, c in enumerate(CLASSES)}
    with open(os.path.join(OUTPUT_DIR, "class_map.json"), "w") as f:
        json.dump(class_map, f, indent=2)
        
    # 8. Zip
    print("\n[8/8] Creating ZIP...")
    with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for file in tqdm(files, desc="      Zipping", leave=False):
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(OUTPUT_DIR))
                zipf.write(file_path, arcname)
                
    print("\n═══════════════════════════════════════════════════════════")
    print("  ✅ DONE — Dataset ready for Kaggle upload")
    print("═══════════════════════════════════════════════════════════\n")

if __name__ == "__main__":
    main()
