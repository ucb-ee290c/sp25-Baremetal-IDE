#!/usr/bin/env python3
"""
alloc_buffers_from_elf.py

Compute safe DRAM addresses for host-written input buffers by inspecting the firmware ELF.

Usage:
  python3 scripts/alloc_buffers_from_elf.py --elf build/.../your.elf

Output:
  - mailbox address (g_mbox if present)
  - stack top (_sp preferred)
  - safe_base = align_up(stack_top + guard)
  - input0/input1 addresses sized for 1x3x224x224 float32 (0x93000 bytes)
"""

import argparse
import re
import subprocess
import sys

INPUT_BYTES_DEFAULT = 1 * 3 * 224 * 224 * 4  # 602112 = 0x93000

def align_up(x: int, a: int) -> int:
    return (x + a - 1) & ~(a - 1)

def run_nm(nm_bin: str, elf: str) -> str:
    try:
        out = subprocess.check_output([nm_bin, "-n", elf], text=True)
        return out
    except subprocess.CalledProcessError as e:
        print(f"ERROR: nm failed: {e}", file=sys.stderr)
        sys.exit(2)

def parse_nm(nm_text: str) -> dict[str, int]:
    sym: dict[str, int] = {}
    # Example line: 0000000081d7f000 b __stack_start
    for line in nm_text.splitlines():
        m = re.match(r"^([0-9a-fA-F]+)\s+\S\s+(\S+)$", line.strip())
        if m:
            addr = int(m.group(1), 16)
            name = m.group(2)
            sym[name] = addr
    return sym

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elf", required=True, help="Firmware ELF")
    ap.add_argument("--nm", default="riscv64-unknown-elf-nm", help="nm binary")
    ap.add_argument("--guard", default="0x100000", help="Guard bytes after stack top (default 1MB)")
    ap.add_argument("--align", default="0x1000", help="Alignment (default 4KB)")
    ap.add_argument("--input-bytes", default=str(INPUT_BYTES_DEFAULT), help="Input tensor bytes")
    args = ap.parse_args()

    guard = int(args.guard, 0)
    align = int(args.align, 0)
    input_bytes = int(args.input_bytes, 0)

    nm_text = run_nm(args.nm, args.elf)
    sym = parse_nm(nm_text)

    # Prefer _sp (explicit stack pointer symbol). Fallback to __stack_end, then __stack_start+__stack_size.
    if "_sp" in sym:
        stack_top = sym["_sp"]
        stack_src = "_sp"
    elif "__stack_end" in sym:
        stack_top = sym["__stack_end"]
        stack_src = "__stack_end"
    else:
        ss = sym.get("__stack_start")
        sz = sym.get("__stack_size")
        if ss is None or sz is None:
            print("ERROR: could not find _sp or (__stack_start and __stack_size) in nm output", file=sys.stderr)
            sys.exit(3)
        stack_top = ss + sz
        stack_src = "__stack_start+__stack_size"

    safe_base = align_up(stack_top + guard, align)
    input0 = safe_base
    input1 = align_up(input0 + input_bytes, align)

    print("=== Firmware-derived addresses ===")
    if "g_mbox" in sym:
        print(f"MAILBOX (g_mbox) : 0x{sym['g_mbox']:08x}")
    elif "__mailbox_start" in sym:
        print(f"MAILBOX (__mailbox_start): 0x{sym['__mailbox_start']:08x}")
    else:
        print("MAILBOX: (not found via nm) — if you used the linker edits below, it will be ORIGIN(TCM).")

    print(f"STACK_TOP ({stack_src}) : 0x{stack_top:08x}")
    print(f"GUARD                 : 0x{guard:x}")
    print(f"SAFE_BASE             : 0x{safe_base:08x}")
    print(f"INPUT_BYTES           : {input_bytes} (0x{input_bytes:x})")
    print()
    print("=== Suggested DRAM buffers ===")
    print(f"INPUT0_ADDR           : 0x{input0:08x}")
    print(f"INPUT1_ADDR           : 0x{input1:08x}")
    print()
    print("=== Shell exports ===")
    print(f"export INPUT0_ADDR=0x{input0:08x}")
    print(f"export INPUT1_ADDR=0x{input1:08x}")

if __name__ == "__main__":
    main()
