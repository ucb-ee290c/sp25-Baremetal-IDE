#!/usr/bin/env python3

"""
Usage: ./file_to_c_data.py <model_file> <c_file>

Output file: <c_file>

This script takes in an arbitrary binary file (though it has hardcoded variables for the
whisper models in its output) and converts it into a C file with a char array containing
that file's data to compile into a binary's memory, as well as another variable containing
the length of that array. This can be used to put arbitrary files (such as the models) in
memory instead of the filesystem.
"""

import sys

print(f"Reading model file {sys.argv[1]}")

with open(sys.argv[1], "rb") as in_file:
    file_data = in_file.read()

print(f"Writing C data file {sys.argv[2]}")

with open(sys.argv[2], "w") as out_file:
    out_file.write("#include \"whisper_model_data.h\"\n")
    out_file.write(f"size_t whisper_model_file_data_len = {len(file_data)};\n")
    out_file.write("char whisper_model_file_data[] = {")
    out_file.write(",".join(map(str, file_data)))
    out_file.write(" };\n")