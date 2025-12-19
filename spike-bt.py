#!/usr/bin/env python3

"""
Usage: `./spike-bt.py <log_file>`

Output file: <log_file>.bt

This script takes the output of `spike -l --log=<log_file> <program.riscv|program.elf>`
and attempts to generate a backtrace at the moment the program crashes. This is a rough
approximation and is likely not going to be entirely correct, but will provide some
context for which functions led up to the crash.

Spike's log shows each instruction run, as well as the start of any function. This script
creates a stack and pushes each function start as a function call, and pops whenever it
sees an instruction that it believes is a return, then outputs the final stack state.
This will make mistakes if a jump happens to the start of a function without a function
call (I believe this happens with bss_init_loop, which is explicitly filtered out) or if
a function returns without the usual return instruction.
"""

import sys
import os
import collections

file = sys.argv[1]
stripped_file = f"{file}.tmp"
out_file = f"{file}.bt"

os.system(fr"grep '>>>>\| ret' {file} > {stripped_file}")

stack = collections.deque()

with open(stripped_file, "r") as f:
    for line in f.readlines():
        if ">>>>" in line:
            func = line[16:-1]
            if func != "bss_init_loop":
                stack.append(func)
        else:
            stack.pop()

with open(out_file, "w") as f:
    for line in stack:
        f.write(line)
        f.write("\n")

os.system(f"rm {stripped_file}")