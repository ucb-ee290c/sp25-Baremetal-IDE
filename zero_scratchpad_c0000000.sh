#!/usr/bin/env bash
#
# zero_scratchpad_c0000000.sh
#
# Writes 0x00000000 across the 16 KiB C2C scratchpad that lives at 0xC000_0000
# (DSP25's local scratchpad in the bidirectional Sophia Lake C2C config).
#
# Each word is written with its own uart_tsi invocation, exactly like the
# manual command:
#
#   uart_tsi +tty=/dev/ttyUSB1 +no_hart0_msip +init_write=0xc0000000:0xf00d2f54 +baudrate=921600 none
#
# After each write we wait ${WAIT} seconds for the write to propagate over the
# TSI/serial-TL link, then send Ctrl-C (SIGINT) via `timeout -s INT` to exit
# uart_tsi before moving on to the next address.
#
set -u

# ---- knobs -------------------------------------------------------------------
TTY="${TTY:-/dev/ttyUSB1}"          # serial port the FPGA is on
BAUD="${BAUD:-921600}"
BASE="${BASE:-0xC0000000}"          # start of the scratchpad
SIZE="${SIZE:-$((16 * 1024))}"      # 16 KiB
STEP="${STEP:-4}"                   # bytes per write (uart_tsi writes 32-bit words)
WAIT="${WAIT:-1}"                   # seconds to let each write propagate before Ctrl-C
UART_TSI="${UART_TSI:-uart_tsi}"    # override with a full path if not on PATH
VALUE="${VALUE:-0x0}"               # value to write
# ------------------------------------------------------------------------------

if ! command -v "${UART_TSI%% *}" >/dev/null 2>&1; then
  echo "error: '${UART_TSI}' not found. Set UART_TSI=/path/to/uart_tsi" >&2
  exit 1
fi

nwords=$(( SIZE / STEP ))
echo "Zeroing ${SIZE} bytes (${nwords} words) at ${BASE} on ${TTY} (WAIT=${WAIT}s/write)"

i=0
for (( off=0; off<SIZE; off+=STEP )); do
  addr=$(printf "0x%08x" $(( BASE + off )))
  i=$(( i + 1 ))
  printf "[%d/%d] init_write %s:%s\n" "${i}" "${nwords}" "${addr}" "${VALUE}"

  # Run uart_tsi, then after ${WAIT}s send SIGINT (== Ctrl-C) so it exits.
  # timeout returns 124 when it fires the signal; that is the expected path.
  timeout -s INT "${WAIT}" \
    ${UART_TSI} +tty="${TTY}" +no_hart0_msip +init_write="${addr}:${VALUE}" +baudrate="${BAUD}" none
  rc=$?
  if [[ ${rc} -ne 0 && ${rc} -ne 124 && ${rc} -ne 130 ]]; then
    echo "warning: uart_tsi exited with code ${rc} at ${addr}" >&2
  fi
done

echo "Done: wrote ${VALUE} to [${BASE}, +${SIZE} bytes)."
