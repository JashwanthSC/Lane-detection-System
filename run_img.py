"""
Simple Image Processing Script
Processes images from input/ folder
Saves with naming: test_module1.jpg, test_module2.jpg, etc.
"""

import os
import sys
from pathlib import Path

# Check required files
required = ['integrated_system.py', 'module1_reliability.py', 
            'module2_detection.py', 'module3_deviation.py']

missing = [f for f in required if not os.path.exists(f)]
if missing:
    print("=" * 70)
    print("MISSING FILES")
    print("=" * 70)
    for f in missing:
        print(f"  X {f}")
    print("=" * 70)
    sys.exit(1)

from integrated_system import process_image_all_modules

# ============================================
# CONFIGURATION - Edit this
# ============================================
IMAGE_NAME = "test.jpg"  # ← Your image name
# ============================================

if __name__ == "__main__":
    # Build paths
    input_path = f"input/{IMAGE_NAME}"
    
    # Check if image exists
    if not os.path.exists(input_path):
        print("=" * 70)
        print("IMAGE NOT FOUND")
        print("=" * 70)
        print(f"Looking for: {input_path}")
        print(f"Current dir: {os.getcwd()}")
        print("\nMake sure:")
        print("  1. You have 'input/' folder")
        print(f"  2. '{IMAGE_NAME}' is inside it")
        print("\nOr edit IMAGE_NAME in this script")
        print("=" * 70)
        sys.exit(1)
    
    stem = Path(IMAGE_NAME).stem
    
    print("=" * 70)
    print("PROCESSING IMAGE")
    print("=" * 70)
    print(f"Input:  {input_path}")
    print("Output: output/")
    print(f"  -> module1/{stem}_module1.jpg")
    print(f"  -> module2/{stem}_module2.jpg")
    print(f"  -> module3/{stem}_module3.jpg")
    print(f"  -> combined/{stem}_full.jpg")
    print("=" * 70)
    print()
    
    try:
        process_image_all_modules(input_path)
        print("\nALL DONE! Check output/ folder")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)