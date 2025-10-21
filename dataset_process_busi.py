#!/usr/bin/env python3
"""
dataset_process_busi.py

Process the Breast Ultrasound Images (BUSI) dataset with Ground Truth.
The dataset has the structure: Dataset_BUSI_with_GT/{benign,malignant,normal}/

Behavior:
 - Processes images and their corresponding masks from BUSI dataset
 - Handles multiple masks per image (some images have multiple _mask_1.png, _mask_2.png files)
 - Creates train/validation/test splits
 - Produces:
     outdir/IMAGES/{train,val,test}/<imagefiles...>
     outdir/MASKS/{train,val,test}/<stem>_mask.png
     outdir/IMAGES/DEBUG_OVERLAYS/{train,val,test}/ (optional)
     outdir/<output_csv> manifest

Usage:
  python dataset_process_busi.py --root ./Dataset_BUSI_with_GT --outdir ./BUSI_PROCESSED --debug
"""
from __future__ import annotations
import os
import argparse
import shutil
import random
from typing import List, Tuple, Optional
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def list_images(dir_path: str) -> List[str]:
    """List all image files in directory, excluding mask files."""
    if not os.path.isdir(dir_path):
        return []
    all_files = [f for f in sorted(os.listdir(dir_path)) if f.lower().endswith(IMG_EXTS)]
    # Filter out mask files
    image_files = [f for f in all_files if '_mask' not in f]
    return image_files

def find_mask_files(dir_path: str, image_stem: str) -> List[str]:
    """Find all mask files for a given image stem."""
    if not os.path.isdir(dir_path):
        return []
    
    mask_files = []
    # Look for primary mask
    primary_mask = f"{image_stem}_mask.png"
    if os.path.exists(os.path.join(dir_path, primary_mask)):
        mask_files.append(primary_mask)
    
    # Look for additional masks (_mask_1.png, _mask_2.png, etc.)
    i = 1
    while True:
        additional_mask = f"{image_stem}_mask_{i}.png"
        mask_path = os.path.join(dir_path, additional_mask)
        if os.path.exists(mask_path):
            mask_files.append(additional_mask)
            i += 1
        else:
            break
    
    return mask_files

def combine_masks(mask_paths: List[str]) -> np.ndarray:
    """Combine multiple mask images into a single mask."""
    if not mask_paths:
        return None
    
    # Load first mask to get dimensions
    first_mask = cv2.imread(mask_paths[0], cv2.IMREAD_GRAYSCALE)
    if first_mask is None:
        return None
    
    # If only one mask, return it
    if len(mask_paths) == 1:
        return first_mask
    
    # Combine multiple masks using logical OR
    combined_mask = first_mask.copy()
    for mask_path in mask_paths[1:]:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    return combined_mask

def split_data(image_files: List[Tuple[str, str, str]], train_ratio: float = 0.7, 
               val_ratio: float = 0.15, random_seed: int = 42) -> Tuple[List, List, List]:
    """Split data into train/val/test sets while maintaining class balance."""
    random.seed(random_seed)
    
    # Group by class
    benign_files = [f for f in image_files if f[1] == 'benign']
    malignant_files = [f for f in image_files if f[1] == 'malignant']
    normal_files = [f for f in image_files if f[1] == 'normal']
    
    def split_class_data(class_files):
        random.shuffle(class_files)
        n = len(class_files)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        return (class_files[:train_end], 
                class_files[train_end:val_end], 
                class_files[val_end:])
    
    # Split each class
    benign_train, benign_val, benign_test = split_class_data(benign_files)
    malignant_train, malignant_val, malignant_test = split_class_data(malignant_files)
    normal_train, normal_val, normal_test = split_class_data(normal_files)
    
    # Combine splits
    train_split = benign_train + malignant_train + normal_train
    val_split = benign_val + malignant_val + normal_val
    test_split = benign_test + malignant_test + normal_test
    
    # Shuffle final splits
    random.shuffle(train_split)
    random.shuffle(val_split)
    random.shuffle(test_split)
    
    return train_split, val_split, test_split

def process_split(split_name: str, split_data: List[Tuple[str, str, str]], 
                  out_images_split: str, out_masks_split: str, debug_dir_split: str, 
                  rows: list, debug: bool):
    """Process a data split (train/val/test)."""
    ensure_dir(out_images_split)
    ensure_dir(out_masks_split)
    if debug:
        ensure_dir(debug_dir_split)
    
    for img_file, class_name, class_dir in tqdm(split_data, desc=f"Processing {split_name}"):
        img_path = os.path.join(class_dir, img_file)
        stem = os.path.splitext(img_file)[0]
        
        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARN] Failed to read image: {img_path}, skipping")
            continue
        
        # Copy image to output
        dst_img_path = os.path.join(out_images_split, img_file)
        try:
            shutil.copy2(img_path, dst_img_path)
        except Exception:
            cv2.imwrite(dst_img_path, img)
        
        # Find and process mask files
        mask_files = find_mask_files(class_dir, stem)
        mask_paths = [os.path.join(class_dir, mf) for mf in mask_files]
        
        # Combine masks if multiple exist
        if mask_paths:
            combined_mask = combine_masks(mask_paths)
        else:
            # Create empty mask if no mask files found
            h, w = img.shape[:2]
            combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        if combined_mask is None:
            h, w = img.shape[:2]
            combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Save combined mask
        mask_name = f"{stem}_mask.png"
        mask_path = os.path.join(out_masks_split, mask_name)
        cv2.imwrite(mask_path, combined_mask)
        
        # Create debug overlay if requested
        if debug:
            if img.ndim == 2:
                vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                vis = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            else:
                vis = img.copy()
            
            # Overlay mask in green
            mask_colored = cv2.applyColorMap(combined_mask, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(vis, 0.7, mask_colored, 0.3, 0)
            
            debug_path = os.path.join(debug_dir_split, f"DEBUG_{stem}.png")
            cv2.imwrite(debug_path, overlay)
        
        # Determine pathology and abnormality
        has_abnormality = int(combined_mask.sum()) > 0
        
        if class_name == 'normal':
            pathology = "Normal"
            abnormality_id = "None"
        elif class_name == 'benign':
            pathology = "Benign" if has_abnormality else "Normal"
            abnormality_id = "benign_lesion" if has_abnormality else "None"
        elif class_name == 'malignant':
            pathology = "Malignant" if has_abnormality else "Normal"
            abnormality_id = "malignant_lesion" if has_abnormality else "None"
        else:
            pathology = "Unknown"
            abnormality_id = "Unknown"
        
        # Add to manifest
        rows.append({
            "dataset": "BUSI",
            "split": split_name,
            "patient_id": stem,
            "image_file_path": os.path.abspath(dst_img_path),
            "roi_mask_file_path": os.path.abspath(mask_path),
            "pathology": pathology,
            "abnormality_id": abnormality_id,
            "class": class_name,
            "has_mask": len(mask_files) > 0,
            "num_masks": len(mask_files)
        })

def collect_all_images(root: str) -> List[Tuple[str, str, str]]:
    """Collect all image files from all class directories."""
    all_images = []
    
    for class_name in ['benign', 'malignant', 'normal']:
        class_dir = os.path.join(root, class_name)
        if not os.path.isdir(class_dir):
            print(f"[WARN] Class directory not found: {class_dir}")
            continue
        
        image_files = list_images(class_dir)
        for img_file in image_files:
            all_images.append((img_file, class_name, class_dir))
    
    return all_images

def main(root: str, outdir: str, output_csv: str, debug: bool, 
         train_ratio: float, val_ratio: float, random_seed: int):
    """Main processing function."""
    ensure_dir(outdir)
    images_out = os.path.join(outdir, "IMAGES")
    masks_out = os.path.join(outdir, "MASKS")
    debug_out = os.path.join(images_out, "DEBUG_OVERLAYS")
    
    # Collect all images
    all_images = collect_all_images(root)
    
    if not all_images:
        print("[ERROR] No images found in the dataset!")
        return
    
    print(f"[INFO] Found {len(all_images)} images total")
    
    # Count by class
    class_counts = {}
    for _, class_name, _ in all_images:
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print(f"[INFO] Class distribution: {class_counts}")
    
    # Split data
    train_split, val_split, test_split = split_data(all_images, train_ratio, val_ratio, random_seed)
    
    print(f"[INFO] Split sizes - Train: {len(train_split)}, Val: {len(val_split)}, Test: {len(test_split)}")
    
    # Process each split
    rows = []
    
    splits = [
        ("train", train_split),
        ("val", val_split),
        ("test", test_split)
    ]
    
    for split_name, split_data in splits:
        if not split_data:
            continue
        
        out_images_split = os.path.join(images_out, split_name)
        out_masks_split = os.path.join(masks_out, split_name)
        debug_dir_split = os.path.join(debug_out, split_name)
        
        process_split(split_name, split_data, out_images_split, out_masks_split, 
                     debug_dir_split, rows, debug)
    
    # Save manifest
    df = pd.DataFrame(rows)
    out_csv_path = output_csv if os.path.isabs(output_csv) else os.path.join(outdir, output_csv)
    df.to_csv(out_csv_path, index=False)
    
    print(f"[INFO] Saved manifest: {out_csv_path}")
    print(f"[INFO] Processed {len(df)} samples. Images -> {images_out}, Masks -> {masks_out}")
    
    # Print final statistics
    print(f"\n[INFO] Final dataset statistics:")
    print(df.groupby(['split', 'class']).size().unstack(fill_value=0))
    print(f"\n[INFO] Pathology distribution:")
    print(df['pathology'].value_counts())

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Process BUSI breast cancer ultrasound dataset")
    p.add_argument("--root", required=True, help="Root folder of the BUSI dataset (contains benign/malignant/normal)")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--output-csv", default="busi_dataset_manifest.csv", help="CSV filename for manifest")
    p.add_argument("--debug", action="store_true", help="Create debug overlays with masks visualized")
    p.add_argument("--train-ratio", type=float, default=0.7, help="Ratio of data for training (default: 0.7)")
    p.add_argument("--val-ratio", type=float, default=0.15, help="Ratio of data for validation (default: 0.15)")
    p.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducible splits (default: 42)")
    
    args = p.parse_args()
    
    # Validate ratios
    if args.train_ratio + args.val_ratio >= 1.0:
        print("[ERROR] train_ratio + val_ratio must be < 1.0")
        exit(1)
    
    main(args.root, args.outdir, args.output_csv, args.debug, 
         args.train_ratio, args.val_ratio, args.random_seed)