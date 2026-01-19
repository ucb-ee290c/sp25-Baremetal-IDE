#!/usr/bin/env python3
"""
Convert .bin files to C header entries and append to test_inputs.h
"""

import os
import numpy as np

# Define your new test images and their expected classes
# Class indices are 0-based (line_number - 1 from labels.txt)
NEW_IMAGES = [
    {
        "bin_file": "build_artifacts/tiger_input.bin",
        "name": "tiger",
        "expected_class": 292,  # tiger (line 293, index 292)
        "description": "Tiger photo"
    },
    {
        "bin_file": "build_artifacts/backpack_input.bin",
        "name": "backpack",
        "expected_class": 414,  # backpack (line 415, index 414)
        "description": "Backpack photo"
    },
    {
        "bin_file": "build_artifacts/park_bench_input.bin",
        "name": "park_bench",
        "expected_class": 703,  # park bench (line 704, index 703)
        "description": "Park bench photo"
    },
    {
        "bin_file": "build_artifacts/axolotl_bill_input.bin",
        "name": "axolotl",
        "expected_class": 29,   # axolotl (line 30, index 29)
        "description": "Axolotl photo"
    },
]

def bin_to_c_array(bin_path, var_name):
    """Convert a .bin file to C array string"""
    arr = np.fromfile(bin_path, dtype=np.float32)
    
    if len(arr) != 1 * 3 * 224 * 224:
        raise ValueError(f"Expected 150528 floats, got {len(arr)}")
    
    lines = []
    chunk_size = 8
    for i in range(0, len(arr), chunk_size):
        chunk = arr[i:i+chunk_size]
        line = ", ".join(f"{v:.8f}f" for v in chunk)
        if i + chunk_size < len(arr):
            lines.append(f"    {line},")
        else:
            lines.append(f"    {line}")
    
    return "\n".join(lines)

def main():
    print("=" * 60)
    print("Adding .bin files to test_inputs.h")
    print("=" * 60)
    
    # Read existing header
    header_path = "build_artifacts/test_inputs.h"
    with open(header_path, "r") as f:
        content = f.read()
    
    # Find where to insert (before the expected_classes array)
    insert_marker = "/* Expected class indices for verification */"
    if insert_marker not in content:
        print("Error: Could not find insertion point in test_inputs.h")
        return
    
    # Split content
    parts = content.split(insert_marker)
    header_part = parts[0]
    footer_part = insert_marker + parts[1]
    
    # Generate new arrays
    new_arrays = []
    all_images = []  # Track all image info for footer update
    
    for img in NEW_IMAGES:
        bin_path = img["bin_file"]
        if not os.path.exists(bin_path):
            print(f"Warning: {bin_path} not found, skipping")
            continue
        
        print(f"Converting {img['name']}...")
        
        array_data = bin_to_c_array(bin_path, img["name"])
        
        new_arrays.append(f"""/* Test image: {img['name']}
 * Source: {bin_path}
 * Description: {img['description']}
 * Expected class: {img['expected_class']}
 */
static const float test_input_{img['name']}[1][3][224][224] = {{{{
{array_data}
}}}};

""")
        all_images.append(img)
        print(f"  ✓ Added {img['name']}")
    
    # Build new footer with all images (existing + new)
    # Parse existing images from header
    existing_images = [
        {"name": "cat", "expected_class": 282, "description": "Orange tabby cat"},
        {"name": "dog", "expected_class": 208, "description": "Yellow Labrador retriever"},
    ]
    
    all_images = existing_images + all_images
    
    new_footer = """/* Expected class indices for verification */
static const int expected_classes[] = {
"""
    for img in all_images:
        new_footer += f"    {img['expected_class']},  /* {img['name']} - {img['description']} */\n"
    new_footer += "};\n\n"
    
    new_footer += """/* Image names for printing */
static const char* test_image_names[] = {
"""
    for img in all_images:
        new_footer += f'    "{img["name"]}",\n'
    new_footer += "};\n\n"
    
    new_footer += f"#define NUM_TEST_IMAGES {len(all_images)}\n\n"
    new_footer += "#endif /* TEST_INPUTS_H */\n"
    
    # Combine everything
    new_content = header_part + "".join(new_arrays) + new_footer
    
    # Write back
    with open(header_path, "w") as f:
        f.write(new_content)
    
    file_size = os.path.getsize(header_path)
    print()
    print(f"✅ Updated {header_path}")
    print(f"   Size: {file_size / 1024 / 1024:.1f} MB")
    print(f"   Total test images: {len(all_images)}")

if __name__ == "__main__":
    main()
