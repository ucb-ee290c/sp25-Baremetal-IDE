#!/usr/bin/env bash
set -euo pipefail

# Get the directory where this script lives (works even if called from elsewhere)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  ./send_image.sh -e FIRMWARE.elf -t /dev/ttyUSBX -i image.jpg [options]

Required:
  -e, --elf        Path to firmware ELF (already built)
  -t, --tty        UART-TSI tty device (e.g. /dev/ttyUSB1)
  -i, --image      Input image path (jpg/png/etc.)

Options:
  -b, --baud       UART baudrate (default: 921600)
  -p, --preprocess Path to preprocess_fast.py (default: <script_dir>/preprocess_fast.py)
  -o, --outdir     Output directory (default: <script_dir>/bin)
      --guard      Guard bytes after stack top (default: 0x100000 = 1MB)
      --align      Alignment for DRAM buffers (default: 0x1000 = 4KB)
      --slot       Which ping-pong buffer to use: 0 or 1 (default: alternates by seq)
      --mailbox    Mailbox base address (hex). Default: 0x8F000000 (high DRAM)
      --no-check   Skip size check of input.bin (default checks for 0x93000)
      --wait       Poll mailbox status/result after kicking (best-effort)
      --reset-seq  Reset sequence counter to 0 (use after chip reset)

What it does:
  1) preprocess_fast.py -> input.bin (float32 NCHW 1x3x224x224, expected 0x93000 bytes)
  2) choose safe DRAM buffer address above firmware stack (from nm symbols)
  3) objcopy input.bin -> input_payload.elf placed at chosen DRAM address
  4) uart_tsi +no_hart0_msip loads input_payload.elf (writes DRAM only)
  5) uart_tsi init_write sets mailbox fields (addr/len/seq/format) and sets READY last

EOF
}

BAUD=921600
PREPROCESS="$SCRIPT_DIR/preprocess_fast.py"
OUTDIR="$SCRIPT_DIR/bin"
GUARD="0x100000"
ALIGN="0x1000"
SLOT=""
MAILBOX=""
NO_CHECK=0
WAIT=0
RESET_SEQ=0

ELF=""
TTY=""
IMAGE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -e|--elf) ELF="$2"; shift 2;;
    -t|--tty) TTY="$2"; shift 2;;
    -i|--image) IMAGE="$2"; shift 2;;
    -b|--baud) BAUD="$2"; shift 2;;
    -p|--preprocess) PREPROCESS="$2"; shift 2;;
    -o|--outdir) OUTDIR="$2"; shift 2;;
    --guard) GUARD="$2"; shift 2;;
    --align) ALIGN="$2"; shift 2;;
    --slot) SLOT="$2"; shift 2;;
    --mailbox) MAILBOX="$2"; shift 2;;
    --no-check) NO_CHECK=1; shift 1;;
    --wait) WAIT=1; shift 1;;
    --reset-seq) RESET_SEQ=1; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "$ELF" || -z "$TTY" || -z "$IMAGE" ]]; then
  usage
  exit 1
fi

if [[ ! -f "$ELF" ]]; then
  echo "ERROR: ELF not found: $ELF"
  exit 2
fi
if [[ ! -e "$TTY" ]]; then
  echo "ERROR: TTY not found: $TTY"
  exit 2
fi
if [[ ! -f "$IMAGE" ]]; then
  echo "ERROR: Image not found: $IMAGE"
  exit 2
fi
if [[ ! -f "$PREPROCESS" ]]; then
  echo "ERROR: preprocess.py not found: $PREPROCESS"
  exit 2
fi

command -v uart_tsi >/dev/null 2>&1 || { echo "ERROR: uart_tsi not in PATH"; exit 3; }
command -v riscv64-unknown-elf-nm >/dev/null 2>&1 || { echo "ERROR: riscv64-unknown-elf-nm not in PATH"; exit 3; }
command -v riscv64-unknown-elf-objcopy >/dev/null 2>&1 || { echo "ERROR: riscv64-unknown-elf-objcopy not in PATH"; exit 3; }
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not in PATH"; exit 3; }

mkdir -p "$OUTDIR"

# ---------------------------
# 1) Preprocess image -> input.bin
# ---------------------------
echo "[1/5] Preprocessing image -> input.bin"
# preprocess_fast.py expects: preprocess_fast.py <image_path> <output_file_path>
BIN="$OUTDIR/input.bin"
python3 "$PREPROCESS" "$IMAGE" "$BIN"

if [[ ! -f "$BIN" ]]; then
  echo "ERROR: Expected $BIN not found. Check preprocess.py output."
  exit 4
fi

BIN_BYTES=$(stat -c%s "$BIN")
echo "  input.bin size: $BIN_BYTES bytes"

# Default MobileNet float32 input size = 1*3*224*224*4 = 602112 = 0x93000
EXPECTED_BYTES=$((1*3*224*224*4))
if [[ $NO_CHECK -eq 0 && "$BIN_BYTES" -ne "$EXPECTED_BYTES" ]]; then
  printf "ERROR: input.bin size %d != expected %d (0x%x). Use --no-check to bypass.\n" \
    "$BIN_BYTES" "$EXPECTED_BYTES" "$EXPECTED_BYTES"
  exit 5
fi

# ---------------------------
# 2) Compute safe DRAM address from firmware ELF symbols + choose ping-pong slot
# ---------------------------
echo "[2/5] Computing safe DRAM addresses from ELF (nm symbols)"

SEQ_FILE="$OUTDIR/.mbox_seq"
if [[ $RESET_SEQ -eq 1 ]]; then
  echo "  Resetting sequence counter to 0"
  echo "0" > "$SEQ_FILE"
fi
if [[ -f "$SEQ_FILE" ]]; then
  SEQ=$(cat "$SEQ_FILE")
else
  SEQ=0
fi
SEQ=$((SEQ + 1))
echo "$SEQ" > "$SEQ_FILE"

# Decide slot if not provided: alternate by seq parity
if [[ -z "$SLOT" ]]; then
  if (( SEQ % 2 == 0 )); then SLOT=1; else SLOT=0; fi
fi
if [[ "$SLOT" != "0" && "$SLOT" != "1" ]]; then
  echo "ERROR: --slot must be 0 or 1"
  exit 6
fi

# Inline python parses nm output:
# - mailbox: g_mbox or __mailbox_start, else default 0x08010000 unless user passed --mailbox
# - stack_top: _sp preferred, else __stack_end, else __stack_start + __stack_size
# - safe_base = align_up(stack_top + guard, align)
# - input0/input1 derived from input.bin size
PY_OUT=$(
python3 - <<PY
import re, subprocess, sys

elf = ${ELF@Q}
guard = int(${GUARD@Q}, 0)
align = int(${ALIGN@Q}, 0)
bin_bytes = int(${BIN_BYTES@Q})
slot = int(${SLOT@Q})

nm = subprocess.check_output(["riscv64-unknown-elf-nm", "-n", elf], text=True)
sym = {}
for line in nm.splitlines():
    m = re.match(r"^([0-9a-fA-F]+)\\s+\\S\\s+(\\S+)$", line.strip())
    if m:
        sym[m.group(2)] = int(m.group(1), 16)

def align_up(x,a): return (x + a - 1) & ~(a - 1)

if "_sp" in sym:
    stack_top = sym["_sp"]
elif "__stack_end" in sym:
    stack_top = sym["__stack_end"]
else:
    ss = sym.get("__stack_start")
    sz = sym.get("__stack_size")
    if ss is None or sz is None:
        print("ERR no_stack_syms=1")
        sys.exit(10)
    stack_top = ss + sz

safe_base = align_up(stack_top + guard, align)
input0 = safe_base
input1 = align_up(input0 + bin_bytes, align)
img_addr = input0 if slot == 0 else input1

# Mailbox address - use hardcoded high DRAM address
# This matches MAILBOX_ADDR in main_int8.c (0x8F000000)
mbox_addr = 0x8F000000

print(f"STACK_TOP=0x{stack_top:08x}")
print(f"SAFE_BASE=0x{safe_base:08x}")
print(f"INPUT0_ADDR=0x{input0:08x}")
print(f"INPUT1_ADDR=0x{input1:08x}")
print(f"IMG_ADDR=0x{img_addr:08x}")
print(f"MAILBOX_ADDR=0x{mbox_addr:08x}")
PY
)

if echo "$PY_OUT" | grep -q "ERR no_stack_syms=1"; then
  echo "ERROR: Could not find stack symbols in ELF. Need _sp or (__stack_end) or (__stack_start + __stack_size)."
  exit 7
fi

echo "$PY_OUT"

# Parse outputs into shell vars
STACK_TOP=$(echo "$PY_OUT" | awk -F= '/^STACK_TOP=/{print $2}')
SAFE_BASE=$(echo "$PY_OUT" | awk -F= '/^SAFE_BASE=/{print $2}')
INPUT0_ADDR=$(echo "$PY_OUT" | awk -F= '/^INPUT0_ADDR=/{print $2}')
INPUT1_ADDR=$(echo "$PY_OUT" | awk -F= '/^INPUT1_ADDR=/{print $2}')
IMG_ADDR=$(echo "$PY_OUT" | awk -F= '/^IMG_ADDR=/{print $2}')
AUTO_MBOX=$(echo "$PY_OUT" | awk -F= '/^MAILBOX_ADDR=/{print $2}')

if [[ -z "$MAILBOX" ]]; then
  MAILBOX="$AUTO_MBOX"
fi

echo "  Using SEQ=$SEQ SLOT=$SLOT IMG_ADDR=$IMG_ADDR MAILBOX=$MAILBOX"

# ---------------------------
# 3) Build payload ELF from input.bin at IMG_ADDR
# ---------------------------
echo "[3/5] Building data-only ELF payload for DRAM address $IMG_ADDR"

PAYLOAD_OBJ="$OUTDIR/input_payload.o"
PAYLOAD_LD="$OUTDIR/payload.ld"
PAYLOAD_ELF="$OUTDIR/input_payload.elf"

# Step 1: Convert binary to relocatable object file
riscv64-unknown-elf-objcopy -I binary -O elf64-littleriscv -B riscv "$BIN" "$PAYLOAD_OBJ"

# Step 2: Create minimal linker script to place data at IMG_ADDR and produce executable
cat > "$PAYLOAD_LD" << EOF
OUTPUT_FORMAT("elf64-littleriscv")
OUTPUT_ARCH(riscv)
ENTRY(_start)

SECTIONS {
  . = $IMG_ADDR;
  .data : { *(.data) }
  /DISCARD/ : { *(*) }
}
EOF

# Step 3: Link into an executable ELF (uart_tsi requires ET_EXEC, not ET_REL)
riscv64-unknown-elf-ld -T "$PAYLOAD_LD" "$PAYLOAD_OBJ" -o "$PAYLOAD_ELF"

echo "  Created executable payload ELF at $PAYLOAD_ELF"

# ---------------------------
# 4) Load payload ELF into DRAM without rebooting core
# ---------------------------
echo "[4/5] Loading payload ELF into DRAM via uart_tsi (+no_hart0_msip)"
uart_tsi +tty="$TTY" +baudrate="$BAUD" +no_hart0_msip "$PAYLOAD_ELF"

# ---------------------------
# 5) Write mailbox fields, then set READY last
# Offsets assumed:
#   +0x00 magic      (0x4D424F58)
#   +0x04 status     (0 idle, 1 ready, 2 busy, 3 done, 4 err)
#   +0x08 seq
#   +0x0c img_bytes
#   +0x10 img_addr_low
#   +0x14 img_addr_high
#   +0x18 img_format (1 = float32 NCHW 1x3x224x224)
#   +0x20 result_top1 (optional clear)
#   +0x24 err_code    (optional clear)
# ---------------------------
echo "[5/5] Writing mailbox fields at $MAILBOX (TCM/scratch) and setting READY last"

# Convert IMG_ADDR to 64-bit low/high (bash arithmetic is 64-bit on most modern shells)
IMG_ADDR_DEC=$((IMG_ADDR))
ADDR_LOW=$(printf "0x%08x" $((IMG_ADDR_DEC & 0xffffffff)))
ADDR_HIGH=$(printf "0x%08x" $(((IMG_ADDR_DEC >> 32) & 0xffffffff)))

# Convert MAILBOX to decimal for arithmetic operations
MBOX_DEC=$((MAILBOX))

MBOX_MAGIC="0x4D424F58"
IMG_FMT="0x00000001"
IMG_BYTES_HEX=$(printf "0x%08x" "$BIN_BYTES")
SEQ_HEX=$(printf "0x%08x" "$SEQ")

# Helper: one 32-bit init_write
# Usage: w32 <offset_from_mailbox> <value>
w32() {
  local offset="$1"
  local val="$2"
  local addr=$(printf "0x%08x" $((MBOX_DEC + offset)))
  echo "  Writing $val to $addr (offset +0x$(printf '%02x' $offset))"
  uart_tsi +tty="$TTY" +baudrate="$BAUD" +no_hart0_msip +init_write="${addr}:${val}" /dev/null
}

# Write fields (status last to ensure atomicity)
# Field offsets from mailbox_t struct:
#   +0x00 magic
#   +0x04 status
#   +0x08 seq
#   +0x0C img_bytes
#   +0x10 img_addr (low 32 bits)
#   +0x14 img_addr (high 32 bits)
#   +0x18 img_format
#   +0x1C reserved0
#   +0x20 result_top1
#   +0x24 err_code

w32 0x00 "$MBOX_MAGIC"
w32 0x10 "$ADDR_LOW"
w32 0x14 "$ADDR_HIGH"
w32 0x0C "$IMG_BYTES_HEX"
w32 0x18 "$IMG_FMT"
w32 0x08 "$SEQ_HEX"
# Clear old results (optional but nice)
w32 0x20 "0x00000000"
w32 0x24 "0x00000000"

# READY last - this triggers firmware to process
w32 0x04 "0x00000001"

echo "Done. Firmware should now pick up SEQ=$SEQ and run inference."

if [[ $WAIT -eq 1 ]]; then
  echo "Polling status/result (best-effort)..."
  # Read helper
  r32() {
    local offset="$1"
    local addr=$(printf "0x%08x" $((MBOX_DEC + offset)))
    echo -n "  [$addr]: "
    uart_tsi +tty="$TTY" +baudrate="$BAUD" +no_hart0_msip +init_read="${addr}" /dev/null 2>/dev/null || echo "(read failed)"
  }
  
  echo "Status (0x04 - expect 3 for DONE):"
  r32 0x04
  echo "Result top1 (0x20):"
  r32 0x20
  echo "Error code (0x24 - expect 0):"
  r32 0x24
fi
