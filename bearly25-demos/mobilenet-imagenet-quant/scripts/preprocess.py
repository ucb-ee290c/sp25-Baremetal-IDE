#!/usr/bin/env python3
"""
preprocess.py - Preprocess an image for MobileNetV2 inference

This script uses the official TorchVision transforms for MobileNetV2
to preprocess an input image and save it as a binary float32 tensor.

Usage: python scripts/preprocess.py <image_path> [output_path]
"""

import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/preprocess.py <image_path> [output_path]")
        print("       output_path defaults to build_artifacts/input.bin")
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "build_artifacts/input.bin"

    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    try:
        import torch
        from torchvision.models import MobileNet_V2_Weights
        from PIL import Image
        import numpy as np
    except ImportError as e:
        print(f"Error: Missing required package: {e}")
        print("   Install with: pip install torch torchvision pillow numpy")
        sys.exit(1)

    print("=" * 50)
    print("Preprocessing Image for MobileNetV2")
    print("=" * 50)
    print()

    # Get official transforms
    weights = MobileNet_V2_Weights.DEFAULT
    transforms = weights.transforms()
    
    print(f"Input image: {image_path}")
    print(f"Output file: {output_path}")
    print()
    print("Transform pipeline:")
    print(f"  - Resize to 232 (shorter edge)")
    print(f"  - Center crop to 224x224")
    print(f"  - Normalize: mean={transforms.mean}, std={transforms.std}")
    print()

    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    print(f"Original size: {img.size}")
    
    tensor = transforms(img)
    # Add batch dimension: (3, 224, 224) -> (1, 3, 224, 224)
    tensor = tensor.unsqueeze(0)
    print(f"Tensor shape: {tuple(tensor.shape)} (NCHW)")
    print(f"Tensor dtype: float32")
    print(f"Value range: [{tensor.min():.3f}, {tensor.max():.3f}]")
    
    # Save as binary float32
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tensor_np = tensor.numpy().astype(np.float32)
    tensor_np.tofile(output_path)
    
    file_size = os.path.getsize(output_path)
    expected_size = 1 * 3 * 224 * 224 * 4  # float32 = 4 bytes
    
    print()
    print(f"Saved preprocessed tensor to {output_path}")
    print(f"   File size: {file_size} bytes (expected: {expected_size})")
    
    if file_size != expected_size:
        print("   Warning: File size mismatch!")
        sys.exit(1)

if __name__ == "__main__":
    main()
