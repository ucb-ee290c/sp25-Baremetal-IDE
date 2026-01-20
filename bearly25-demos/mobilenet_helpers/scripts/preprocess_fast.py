#!/usr/bin/env python3
"""
preprocess_static.py - Preprocess an image for MobileNetV2 inference (no torch required)

Usage: python scripts/preprocess_static.py <image_path> [output_path]
"""

import sys
import os
import numpy as np
from PIL import Image

# Hardcoded MobileNetV2 (ImageNet) normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(img):
    # Resize so shorter edge is 232
    w, h = img.size
    scale = 232.0 / min(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # Center crop to 224x224
    left = (new_w - 224) // 2
    top = (new_h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))

    # Convert to numpy, scale to [0,1]
    arr = np.asarray(img).astype(np.float32) / 255.0  # shape (224, 224, 3)
    # Normalize
    arr = (arr - MEAN) / STD
    # Change to (3, 224, 224)
    arr = arr.transpose(2, 0, 1)
    # Add batch dimension: (1, 3, 224, 224)
    arr = arr[np.newaxis, :]
    return arr

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/preprocess_static.py <image_path> [output_path]")
        print("       output_path defaults to build_artifacts/input.bin")
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "build_artifacts/input.bin"

    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found: {image_path}")
        sys.exit(1)

    img = Image.open(image_path).convert("RGB")
    print(f"Original size: {img.size}")

    tensor = preprocess(img)
    print(f"Tensor shape: {tensor.shape} (NCHW)")
    print(f"Tensor dtype: float32")
    print(f"Value range: [{tensor.min():.3f}, {tensor.max():.3f}]")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tensor.astype(np.float32).tofile(output_path)

    file_size = os.path.getsize(output_path)
    expected_size = 1 * 3 * 224 * 224 * 4  # float32 = 4 bytes

    print(f"✅ Saved preprocessed tensor to {output_path}")
    print(f"   File size: {file_size} bytes (expected: {expected_size})")
    if file_size != expected_size:
        print("⚠️  Warning: File size mismatch!")
        sys.exit(1)

if __name__ == "__main__":
    main()