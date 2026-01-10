"""
Data Validation Script

This script checks if your data is properly formatted and ready for processing.
Run this before prepare_data.py to catch issues early.

Usage:
    python src/validate_data.py
"""

import os
import sys
import pandas as pd
from pathlib import Path
from PIL import Image


def check_results_csv(csv_path):
    """Validate the results.csv file."""
    print("\n" + "="*60)
    print("VALIDATING results.csv")
    print("="*60)

    if not os.path.exists(csv_path):
        print(f"❌ ERROR: File not found: {csv_path}")
        return False

    try:
        df = pd.read_csv(csv_path)
        print(f"✓ File loaded successfully")
        print(f"  Total rows: {len(df)}")

        # Check required columns
        required_cols = ['uid', 'best_mqe', 'topographic_error', 'inactive_neuron_ratio']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"❌ ERROR: Missing required columns: {missing_cols}")
            print(f"  Found columns: {list(df.columns)}")
            return False
        else:
            print(f"✓ All required columns present")

        # Check for missing values
        for col in required_cols:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                print(f"⚠ WARNING: Column '{col}' has {null_count} missing values")

        # Check for duplicates
        dup_count = df['uid'].duplicated().sum()
        if dup_count > 0:
            print(f"⚠ WARNING: Found {dup_count} duplicate UIDs")
        else:
            print(f"✓ No duplicate UIDs")

        # Check data types and ranges
        numeric_cols = ['best_mqe', 'topographic_error', 'inactive_neuron_ratio']
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"❌ ERROR: Column '{col}' is not numeric")
                return False

            min_val = df[col].min()
            max_val = df[col].max()
            print(f"✓ {col}: range [{min_val:.6f}, {max_val:.6f}]")

            if min_val < 0:
                print(f"⚠ WARNING: {col} has negative values")

        return True

    except Exception as e:
        print(f"❌ ERROR: Failed to read CSV: {e}")
        return False


def check_images(image_dir, results_csv):
    """Validate image files."""
    print("\n" + "="*60)
    print("VALIDATING IMAGE FILES")
    print("="*60)

    if not os.path.exists(image_dir):
        print(f"❌ ERROR: Directory not found: {image_dir}")
        return False

    # Load UIDs from CSV
    try:
        df = pd.read_csv(results_csv)
        expected_uids = set(df['uid'].values)
        print(f"✓ Expecting images for {len(expected_uids)} UIDs")
    except Exception as e:
        print(f"❌ ERROR: Could not load UIDs from CSV: {e}")
        return False

    # Check image directory
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob('*.png'))
    print(f"✓ Found {len(image_files)} PNG files")

    if len(image_files) == 0:
        print(f"❌ ERROR: No PNG images found in {image_dir}")
        return False

    # Check matching UIDs
    found_uids = {f.stem for f in image_files}
    missing_images = expected_uids - found_uids
    extra_images = found_uids - expected_uids

    if missing_images:
        print(f"⚠ WARNING: {len(missing_images)} UIDs in CSV have no corresponding image")
        if len(missing_images) <= 10:
            print(f"  Missing: {list(missing_images)[:10]}")
        else:
            print(f"  First 10 missing: {list(missing_images)[:10]}")

    if extra_images:
        print(f"⚠ WARNING: {len(extra_images)} images have no corresponding UID in CSV")

    matching = len(expected_uids & found_uids)
    print(f"✓ {matching} images match UIDs in CSV ({matching/len(expected_uids)*100:.1f}%)")

    if matching == 0:
        print(f"❌ ERROR: No matching images found!")
        return False

    # Check image quality
    print("\nChecking image files...")
    corrupt_count = 0
    valid_count = 0
    total_size = 0

    for img_file in image_files[:100]:  # Check first 100 images
        try:
            img = Image.open(img_file)
            img.verify()  # Verify it's not corrupt
            valid_count += 1
            total_size += os.path.getsize(img_file)
        except Exception as e:
            corrupt_count += 1
            if corrupt_count <= 5:
                print(f"⚠ WARNING: Corrupt image: {img_file.name}")

    if corrupt_count > 0:
        print(f"⚠ WARNING: Found {corrupt_count} corrupt images")
    else:
        print(f"✓ All checked images are valid ({valid_count} checked)")

    avg_size = total_size / valid_count if valid_count > 0 else 0
    print(f"✓ Average image size: {avg_size/1024:.1f} KB")

    # Check image dimensions
    sample_images = list(image_files)[:10]
    dimensions = []

    for img_file in sample_images:
        try:
            img = Image.open(img_file)
            dimensions.append(img.size)
        except:
            pass

    if dimensions:
        unique_dims = set(dimensions)
        if len(unique_dims) == 1:
            print(f"✓ All sample images have same dimensions: {dimensions[0]}")
        else:
            print(f"⚠ INFO: Images have varying dimensions:")
            for dim in unique_dims:
                count = dimensions.count(dim)
                print(f"  {dim}: {count} images")

    return matching >= len(expected_uids) * 0.5  # At least 50% should match


def check_disk_space():
    """Check available disk space."""
    print("\n" + "="*60)
    print("CHECKING DISK SPACE")
    print("="*60)

    try:
        import shutil
        total, used, free = shutil.disk_usage(".")

        print(f"Total: {total / (1024**3):.1f} GB")
        print(f"Used:  {used / (1024**3):.1f} GB")
        print(f"Free:  {free / (1024**3):.1f} GB")

        if free < 5 * 1024**3:  # Less than 5 GB
            print(f"⚠ WARNING: Low disk space! Models and logs require space.")
        else:
            print(f"✓ Sufficient disk space available")

        return True
    except Exception as e:
        print(f"⚠ Could not check disk space: {e}")
        return True


def main():
    """Main validation function."""
    print("="*60)
    print("SOM QUALITY ANALYZER - DATA VALIDATION")
    print("="*60)

    results_csv = "data/results.csv"
    image_dir = "data/raw_maps"

    all_passed = True

    # Validate CSV
    if not check_results_csv(results_csv):
        all_passed = False

    # Validate images
    if not check_images(image_dir, results_csv):
        all_passed = False

    # Check disk space
    check_disk_space()

    # Final summary
    print("\n" + "="*60)
    if all_passed:
        print("✓ VALIDATION PASSED")
        print("="*60)
        print("\nYour data looks good!")
        print("Next step: Run 'python src/prepare_data.py'")
        sys.exit(0)
    else:
        print("❌ VALIDATION FAILED")
        print("="*60)
        print("\nPlease fix the errors above before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
