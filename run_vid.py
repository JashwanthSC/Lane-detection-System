"""
Simple Video Processing Script
Processes videos from input/ folder
Saves with naming: test_video_module1.mp4, test_video_module2.mp4, etc.
"""

import os
import sys
from pathlib import Path

# ============================================
# CHECK REQUIRED FILES
# ============================================
required = [
    'integrated_system.py',
    'module1_reliability.py',
    'module2_detection.py',
    'module3_deviation.py'
]

missing = [f for f in required if not os.path.exists(f)]
if missing:
    print("=" * 70)
    print("MISSING FILES")
    print("=" * 70)
    for f in missing:
        print(f"  X {f}")
    print("=" * 70)
    sys.exit(1)

from integrated_system import process_video_all_modules

# ============================================
# CONFIGURATION
# ============================================
VIDEO_NAME = "test_video.mp4"  # ← change if needed
# ============================================

if __name__ == "__main__":

    input_path = f"input/input_video2.mp4"

    # Check if video exists
    if not os.path.exists(input_path):
        print("=" * 70)
        print("VIDEO NOT FOUND")
        print("=" * 70)
        print(f"Looking for: {input_path}")
        print(f"Current dir: {os.getcwd()}")
        print("\nMake sure:")
        print("  1. You have 'input/' folder")
        print(f"  2. '{VIDEO_NAME}' is inside it")
        print("=" * 70)
        sys.exit(1)

    stem = Path(VIDEO_NAME).stem

    print("=" * 70)
    print("PROCESSING VIDEO")
    print("=" * 70)
    print(f"Input:  {input_path}")
    print("Output: output/")
    print(f"  -> module1/{stem}_module1.mp4")
    print(f"  -> module2/{stem}_module2.mp4")
    print(f"  -> module3/{stem}_module3.mp4")
    print(f"  -> combined/{stem}_full_system.mp4")
    print("=" * 70)
    print()

    try:
        process_video_all_modules(input_path)
        print("\nALL DONE! Check output/ folder")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)