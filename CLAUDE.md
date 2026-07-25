# CLAUDE.md

Working notes for Claude when developing on this repo. This is the **baremetal bringup
environment** for two SP25 chips: **Bearly ML 25** and **DSP 25**. Primary current focus
is the **chip-to-chip (C2C) test suite** under `c2c-demos/`.

Read the "Known Chip Bugs & Quirks" section before writing or changing any C2C code — it is
the living record of hardware behavior we've discovered on silicon. Respect it.

### Current focus

- **FULL C2C KWS DEMO WORKS END-TO-END ON SILICON (2026-07-22).** `dsp-kws-rolling` computes MFCC on
  DSP → streams the case over the C2C link → `bearly-kws-rolling` runs TinySpeech and correctly
  predicts `yes` on the embedded `yes_test_005` sample. This is the payoff of plans 001+002.
- **Sync = the proven turn-taking pattern (plan 002 DONE).** Reused from `hello-wfi` via
  `c2c-demos/common/c2c_turnsync.h`: turn register (spad `0x20`, 0=DSP/1=BML) + CLINT-MSIP wake +
  CLINT-timer safety net + `ack_index`/`case_index`-keyed **self-heal retransmit**. Streams
  indefinitely; a dropped cross-link write self-heals (see "Reliable C2C turn-taking synchronization"
  and the bug log). This is the template for all C2C sync.
- **Two accuracy fixes were needed on top of sync** (see "KWS accuracy / TinySpeech" section):
  1. **MFCC layout transpose** — DSP now writes the case **coeff-major** (`idx = coeff*94 + frame`)
     to match the model input `{1,1,12,94}`; it was frame-major (scrambled features). FIXED.
  2. **int8 conv2 kernel is broken on silicon** (garbage activation max `M2 ≈ INT32_MAX` poisons
     calibration → wrong + non-deterministic). **Workaround: run the FLOAT pipeline**
     (`TINYSPEECH_INT8_PIPELINE=0`, CMake option `KWS_BEARLY_ROLLING_USE_FLOAT_PIPELINE=ON`). int8
     root-cause is deferred (suspect the RVV conv2 microkernel).
- **NEXT: validate other sample recordings** (plan `.claude/plans/003-kws-multi-testcase.md`) — the
  demo is proven on one `yes` clip; add more test inputs / cross-check against the 100-case
  reference labels. Then optionally: match the MFCC quant scale exactly, and fix the int8 conv2 bug.
- **The shared region / two scratchpads are non-negotiable** — they stay the transport.
- **LIVE I2S MIC AUDIO SOURCE WIRED (plan 001 P1, 2026-07-25) — built, pending on-silicon
  validation.** `dsp-kws-rolling` gained a `KWS_DSP_ROLLING_USE_MIC` CMake option: instead of
  embedded waveforms it captures ~1 s of live audio from the DSP I2S mic (proven in `dsp-i2s-test`),
  then runs the *same* MFCC → quantize → C2C-stream pipeline unchanged. Mic is clocked at **16 kHz
  directly** via `set_I2S_sample_freq(ch, target_freq, 16000, 32)` (matches the MFCC front-end — no
  resampling). Capture is **VAD-gated** (short-frame AC energy onset + pre-roll; knobs
  `KWS_DSP_ROLLING_VAD_THRESHOLD`/`_FRAME_SAMPLES`/`_PREROLL_SAMPLES`/`_ENABLE` in the config header).
  Live cases carry `expected_label = -1` → BML reports the prediction as `expected=unknown (not
  scored)`. **CONFIRMED WORKING ON SILICON (2026-07-25)** — speak at the mic, BML prints the keyword.
  Build DSP:
  `make build CHIP=dsp25 PLATFORM=CHIP TARGET=dsp-kws-rolling
  EXTRA_CMAKE_ARGS="-DBUILD_MFCC_LIB=ON -DKWS_DSP_ROLLING_USE_MIC=ON -DKWS_DSP_ROLLING_MULTI_SIGNAL=OFF"`;
  BML: `... CHIP=bearly25 TARGET=bearly-kws-rolling EXTRA_CMAKE_ARGS="-DKWS_BEARLY_ROLLING_USE_FLOAT_PIPELINE=ON -DKWS_BEARLY_ROLLING_DEBUG_INPUT_COMPARE=0"`
  (INPUT-CMP is meaningless for live audio). Two quality knobs added after first-light:
  - **VAD energy threshold** `KWS_DSP_ROLLING_VAD_THRESHOLD` (DSP), default now **5e-4** (was 1e-4) —
    rejects quiet room noise; tune from the logged per-frame `energy` lines (each capture also prints
    an `absmean`/min/max stats line).
  - **Confidence gate** `KWS_BEARLY_ROLLING_MIN_SCORE` (BML, default **3.0**) — softmax is OFF so the
    score is the top raw **logit**; when it's not `> 3.0` the RESULT line prints `pred=no word` and
    the case doesn't count toward the tally. Filters low-confidence non-speech that trips the VAD gate.
  Mutually exclusive with `KWS_DSP_ROLLING_MULTI_SIGNAL`.

---

## The two chips

Both are RISC-V SoCs built from the Chipyard ecosystem. Board bringup is done over UART.

| | **Bearly ML 25** | **DSP 25** |
|---|---|---|
| `CHIP=` (build) | `bearly25` | `dsp25` |
| Platform dir | `platform/bearly25/` | `platform/dsp25/` |
| Accelerator | `CONV2D` (2D conv, `hal_ope`) | `CONV1D`/DMA/I2S (`hal_conv`, `hal_dma`, `hal_i2s`) |
| Role in C2C demos | **receiver / consumer** (bearly-*) | **producer / transmitter** (dsp-*) |
| `SYS_CLK_FREQ` | 50 MHz (nominal) | 50 MHz (nominal) |
| Linker scripts | `bearly25.ld`, `bearly25-maxheap.ld` | `dsp25.ld`, `dsp25-flash.ld`, `dsp25-scratch.ld` |
| OpenOCD cfg | `platform/bearly25/bearly25.cfg` | `platform/dsp25/dsp25.cfg` |

`platform/c2c25/chip_config.h` is nearly identical to `dsp25` and is where the C2C link
address window notes live (`// #define C2C_BASE 0x180000000U`).

Timing note: demos set a `TARGET_FREQUENCY_HZ` (e.g. 500 MHz in
`c2c_transfer_dsp_config.h`) that differs from the nominal `SYS_CLK_FREQ` (50 MHz). All
in-demo timing is measured in **core cycles via `rdcycle`**, not wall-clock, so treat cycle
counts as the source of truth and confirm the true operating frequency before converting to
seconds.

---

## C2C link & shared-memory model

- **Two scratchpads, one adjacent to each chip:**
  - **`0xC0000000` — DSP-adjacent scratchpad.** (`*_SHARED_BASE_ADDR`, 16 KiB.)
  - **`0xD0000000` — BML-adjacent scratchpad.**
- **Golden access rule — you may READ only your own adjacent spad; you may WRITE to both.**
  - DSP: reads `0xC0000000` (local); writes `0xC0000000` **and** `0xD0000000`.
  - BML: reads `0xD0000000` (local); writes `0xD0000000` **and** `0xC0000000`.
  - **Neither chip can read across the link.** To send data you *write into the other chip's
    spad*; that chip then reads it locally from its own spad. Never issue a load to the remote
    spad — it is not readable.
- **Cross-spad (remote) writes are the unstable ones** — repeat every write into the *other*
  chip's spad several times to make it stick (see unstable-access bug). Writes to your own spad
  are local/stable. Every read of your own spad still needs a **full cache flush first**,
  because the remote wrote it behind your cache's back.
- **Reaching the peer across the link (addressing).** You cannot READ the remote spad, but to
  WRITE the peer's spad or ring its MSIP, take the peer-local address and **prepend a leading `1`
  (bit 32)**:
  - Peer spad: from BML, DSP's `0xC000_0000` is reached at **`0x1_C000_0000`**; from DSP, BML's
    `0xD000_0000` is reached at **`0x1_D000_0000`**.
  - Peer CLINT MSIP: own MSIP is `0x0200_0000` (CLINT base, hart 0); the peer's is
    **`0x1_0200_0000`**. Writing `1` there raises a machine software interrupt on the peer.
- **Cross-chip wake = CLINT MSIP + `wfi`.** A sleeping core waits in `wfi` with `mie.MSIE` set and
  `mstatus.MIE=0` (so the interrupt wakes it but is NOT taken as a trap — no handler; execution
  resumes after `wfi`). The peer wakes it by writing its MSIP across the link. CLINT layout: MSIP
  @ `+0x0000`, `mtimecmp` @ `+0x4000`, `mtime` @ `+0xBFF8`; `MTIME_FREQ = 50 kHz` (20 us/tick).
- **The cache flush must be force-eviction.** Writing `1` to the cache-controller flush register
  (`0x02010200`) does **NOT** evict on this silicon — always use the 256 KiB buffer-walk
  (`cache_evict_all` / `hwfi_cache_flush`).
- **DSP writes, Bearly reads (data direction unchanged).** DSP is the producer.
- **Handshake** (finalized): DSP writes the case **into `0xD0000000` (BML's spad)** and sets a
  `data_ready` flag there. BML polls `0xD0000000` locally, reads the case, runs inference, then
  **clears `data_ready` in `0xD0000000`** (local write) and **sets a `rx_ready` flag by writing
  into `0xC0000000` (DSP's spad)**. DSP polls `0xC0000000` locally for `rx_ready` before sending
  the next case. (Note: this is the *physically forced* mapping — a receiver can only read its
  own spad, so payload for BML must live in BML's spad `0xD0000000`.)
- **Coherence is not automatic across the link, and access is unreliable.** See the
  **cache-manipulation** and **unstable-access** entries in the bug log — those two hardware
  facts dictate the shape of all shared-memory code. In short: every access (read *or* write,
  including a poll) must be followed by a **full cache flush**, and because individual
  reads/writes are not stable, intended data may need to be **written several times**.
  Demos implement this with:
  - `*_fence_rw()` → `fence rw, rw` around every shared-memory access.
  - `cache_evict_all()` / `cache_writeback_pressure()` — walks a large (`256 KiB`) aligned
    scratch buffer, touching one byte per 64-byte line for several passes, to force the entire
    cache out. This is the current stand-in for a real full-cache-flush instruction.
- Protocol headers live in `c2c-demos/common/`:
  - `transfer_proto.h` — single 64-bit cycle word, DSP overwrites with its `rdcycle` (latency
    measurement).
  - `simpletest_proto.h` — mailbox + ring of message slots with `commit_seq` handshake.
  - `kws_proto.h` — KWS streaming mailbox + ring slots / fast case slots (`commit_seq`
    guards a partially-filled slot: 0 while filling, N when case N committed).
  - `kws_rolling_proto.h` — rolling-window KWS variant built on `kws_proto.h`.
- **Commit-sequence handshake pattern** (used across simpletest/kws): producer fills a slot,
  fences, then writes `commit_seq = N` last; consumer waits for `commit_seq != 0` / expected N
  before reading the payload. Never reorder the commit write before the payload write.

---

## Reliable C2C turn-taking synchronization (proven on silicon 2026-07-12)

The robust bidirectional sync we converged on after several failed attempts. Proven end-to-end in
`c2c-demos/hello-wfi` (single-chip `wfi`/MSIP sanity) and `c2c-demos/dsp-hello-wfi` +
`c2c-demos/bearly-hello-wfi` (two-chip interrupt ping-pong that counts a shared "baton" back and
forth forever). Shared implementation: **`c2c-demos/common/hello_wfi_link.h`**. **Reuse this
template for all future C2C sync.** It has been ported to the KWS demos (plan 002, DONE) via a
factored header **`c2c-demos/common/c2c_turnsync.h`** (same three layers, built on `c2c_shm` so it
shares one flush + write-repeat implementation). KWS uses turn-register offset **`0x20`** (not
`0x00`) because the KWS spad control block already uses `0x00`.

Why the earlier attempts failed: a one-shot MSIP edge into a core asleep in `wfi` is
**unrecoverable** if that single cross-link write drops — and cross-link writes drop
non-deterministically. No amount of "write it more times" removes the tail; you also need the
receiver to re-check independently, and a way to ignore wakes that aren't for it.

Three layers, each covering a distinct failure mode:

1. **Turn register (correctness / who-goes-next).** One word at **offset `0x00` of each spad**:
   `0 = DSP's turn`, `1 = BML's turn`. A chip reads it from its OWN spad (local, flush-first) and
   runs ONLY when the value equals its own id; any other value → back to `wfi`. Every wake is
   self-checking, so spurious / duplicate / early wakes never cause double-processing.
2. **CLINT timer (liveness / dropped-wake recovery).** Both cores arm a periodic machine-timer
   interrupt (`mie.MTIE`) alongside `MSIE`, with `mstatus.MIE=0` (wakes, never traps). So a
   sleeper re-checks its turn register at least every `HELLO_WFI_POLL_INTERVAL_TICKS` (~50 ms)
   even if the wake MSIP was dropped — a dropped MSIP costs latency, not liveness.
3. **Hardened cross-link writes (delivery).** Every cross-link store (data + turn register + MSIP)
   is repeated `HELLO_WFI_WRITE_REPEATS` times (fenced) then flushed, to fight the
   unstable-remote-write quirk.

**Spad layout:** turn register @ `0x00`, baton/data @ `0x04` (both 32-bit; spads are
32-bit-access-only).

**Handoff order (commit discipline):** write **data** into peer spad → set **turn register** in
peer spad (the commit; data is resident before the peer sees its turn) → set turn register in OWN
spad to the peer's id (so a later spurious wake of ours reads "not my turn") → raise peer **MSIP**.

**Wait path:** `wfi`; on every wake (MSIP or timer) clear own MSIP, flush, re-read own turn
register; proceed only when it is our turn.

**Boot:** each chip clears its own spad (baton + turn) and sets its own turn register to the
PEER's id, so it stays asleep until explicitly handed to (also defeats stale spad SRAM across a
chip-only reset).

**Residual / operational notes:**
- In plain `hello-wfi`, the one thing that must land is the **turn-register write** (hardened by
  repeats); the timer recovers a dropped MSIP but not a dropped turn write.
- **KWS added self-heal retransmit** (closes that residual, keyed on the monotonic counters):
  after granting case `N`, the producer waits for `ack_index >= N` and, on each idle timer tick with
  no ack, **re-grants `N`** (re-writes payload/`case_index` + turn + wake). The consumer only infers
  when `case_index > last_consumed`; a **duplicate** grant (its ack was lost) → it **re-acks without
  re-inferring**. A dropped grant, dropped ack, OR dropped wake all self-heal, and the monotonic
  guard prevents double-inference. On silicon this recovered mid-stream drops (you see occasional
  `re-grant`/`dup grant` log lines and it keeps going). **This is the recommended shape for any
  payload-carrying C2C sync.**
- **Start DSP first (or together).** boot-init sets the turn to "peer"; the boot barrier (`bml_ready`)
  ensures BML clears its spad before DSP's first write lands, removing the boot-order stale hazard.

---

## KWS accuracy / TinySpeech (findings 2026-07-22)

Getting correct predictions (not just a working link) took three separate discoveries. Once sync
worked, the model still mispredicted; debugging split cleanly into **link → MFCC front-end → inference**.

**The TinySpeech runtime lives in `bearly25-bmarks/tinyspeech-mc/`.** Classes (index order):
`0=yes, 1=no, 2=on, 3=off, 4=stop, 5=go`. Two golden references shipped in that lib:
- `include/tinyspeech_inputs.h` — 100 cases, each an **int8 MFCC map (12×94=1128 B) + expected_label**.
  This is the *exact* input Spike ran on (BML now includes it for debug/compare/calibration).
- `include/tinyspeech_reference.h` — per-case expected/predicted labels, probs, logits, and 12
  per-layer **stage sums** (great for localizing which layer diverges on-chip).

**1. MFCC layout was transposed (FIXED).** The model input is `{1,1,H=12,W=94}` → it reads
`g_case[coeff*94 + frame]` (**coefficient-major**). DSP originally wrote **frame-major**
(`g_case[frame*12 + coeff]`) → scrambled features. `compute_full_case()` in `dsp-kws-rolling` now
writes coeff-major. Confirmed via the `INPUT-CMP` diagnostic: the received coeff-0 row now ramps
like the reference.

**2. MFCC values still differ by scale/recipe (OPEN, but model is robust enough).** DSP computes
MFCC on-chip (`mfcc_driver`, FFT/mel/DCT + `quantize_mfcc` with `KWS_DSP_ROLLING_MFCC_QUANT_SCALE=4.0`,
`_ZERO=0.0`), which does NOT match the reference extractor bit-for-bit (window/mel/log/DCT + a
different int8 scale). After the transpose, `INPUT-CMP max_abs_diff ≈ 72` (coeff-0 spans ≈[-89,+40]
vs ref ≈[-127,+17]) — but the **float model still predicts `yes` correctly** on DSP's features. If a
future sample mispredicts, tune `QUANT_SCALE`/`_ZERO` to minimize `INPUT-CMP` `max_abs_diff`.

**3. The int8 pipeline is broken on silicon (WORKAROUND: float).** Calibration reduces to **three
integers** — the per-conv activation maxima `g_calib_max1/2/3` (everything else — requant
multipliers `mulN_q31`, scales `sN_fixed`, biases — is derived). On BML, `M2` (conv2) came back as
**deterministic garbage ≈ `INT32_MAX`** while `M1`/`M3` were sane, poisoning calibration → wrong +
non-deterministic predictions. It is **not** multicore-specific (identical single-core), so the
suspect is the **RVV conv2 microkernel** (a vector tail/reduction or accumulation-overflow bug the
Spike reference never exercises). **Workaround in use: the FLOAT model** (`TINYSPEECH_INT8_PIPELINE=0`),
which the reference validates and which bypasses the int8 conv2 path. Float is slower
(~16M cycles/inference vs ~7.7M int8) but correct.
- Also learned: our demo originally calibrated int8 on a **single** sample (degenerate scales); the
  standalone benchmark calibrates over all 100 cases then freezes. Added `tinyspeech_int8_calib_get_max`
  / `set_max` to `tinyspeech_int8.{h,c}` so a computed calibration (just the 3 maxima) can be **baked
  in** and loaded with no on-chip pass — relevant once the int8 conv2 bug is fixed.

**Debug harness (in `bearly-kws-rolling`, all `#ifndef`-guarded config flags):**
- `KWS_BEARLY_ROLLING_DEBUG_INPUT_COMPARE` (=1): prints `INPUT-CMP` once — received `g_case` vs
  `g_tinyspeech_test_inputs[REF_CASE_INDEX]` (mismatch count, max_abs_diff, both `[0:16]` previews).
- `KWS_BEARLY_ROLLING_USE_GOLDEN_INPUT` (=0 normally): infer on the reference bytes instead of the
  received case — isolates model/inference from the MFCC front-end + link. Infer log tags `[GOLDEN]`
  vs `[dsp]` so you always know which input was used.
- `KWS_BEARLY_ROLLING_REF_CASE_INDEX` (=5): which reference case (`yes_test_005`) to compare/use.
- `KWS_BEARLY_ROLLING_CALIBRATE_FULL` (int8 only): calibrate over all 100 ref cases at boot + print
  `CALIB_MAX M1/M2/M3`.
- CMake options in `c2c-demos/bearly-kws-rolling/CMakeLists.txt`:
  `KWS_BEARLY_ROLLING_USE_FLOAT_PIPELINE` (ON → `TINYSPEECH_INT8_PIPELINE=0`) and
  `KWS_BEARLY_ROLLING_FORCE_SINGLECORE_INT8` (int8 debug). New CMake options need a clean reconfigure.

**Current known-good config for the working demo:** float pipeline ON, golden OFF, transpose in
place, DSP pacing `KWS_DSP_ROLLING_INTER_CASE_QUIET_CYCLES=500M` (~1 s between predictions). Start
DSP first, then BML.

---

## Building & running

Build uses CMake driven by the top-level `Makefile`. Pick `CHIP` and a `TARGET`:

```bash
# DSP side (producer)
make build CHIP=dsp25    TARGET=c2c-transfer-dsp
# Bearly side (receiver)
make build CHIP=bearly25 TARGET=c2c-transfer-bearly
```

C2C demo targets (`c2c-demos/CMakeLists.txt`): `dsp-kws`, `bearly-kws`, `dsp-kws-rolling`,
`bearly-kws-rolling`, `dsp-simpletest`, `bearly-simpletest`, `c2c-measure`,
`c2c-transfer-dsp`, `c2c-transfer-bearly`, `hello-wfi` (single-chip `wfi`/MSIP sanity — build
with either `CHIP`), `dsp-hello-wfi`, `bearly-hello-wfi` (two-chip turn-taking ping-pong; the
reference sync implementation). Each `<chip>-*` target must be built with the matching `CHIP=`.
Output ELF lands in `build/c2c-demos/<target>/<target>.elf`.

Flash / run on real silicon over UART (see `Makefile`):

```bash
make tsi-run   TTY=<tty> BINARY=build/c2c-demos/<target>/<target>.elf   # load & run via uart_tsi
make checktsi  TTY=<tty>                                                # sanity poke a scratch addr
```

Simulation path (VCS) exists via `make run CONFIG=... BINARY=...` but chip bringup is the
primary use case here.

Because a C2C run needs **both** chips: build+flash the DSP producer and the Bearly consumer
onto their respective boards, then start them. The producer uses a `STARTUP_DELAY_CYCLES`
grace period so the consumer can be up and polling first.

`PLATFORM=CHIP` (default) selects on-chip linker variants; `PLATFORM=SIMS` selects sim
linkers. `LINKER` can override the linker-script variant.

---

## `c2c-demos/` structure & conventions

Each demo is a standalone `app_init()` / `app_main()` executable:
- `src/main.c` — `app_init()` sets up, `app_main()` runs the loop then `wfi`s forever.
- `include/<name>_config.h` — all tunables as `#ifndef`-guarded `#define`s (packet counts,
  intervals, cache-evict geometry, log enable). Prefer adding a new guarded `#define` over
  hardcoding.
- `include/main.h` — pulls in config + `simple_setup.h` (from `bmark-lib/`).
- Logging via `<NAME>_LOG(...)` macro (wraps `printf`, gated by `<NAME>_LOG_ENABLE`).
- `_Static_assert` the cache-evict geometry invariants (power-of-two line size, evict bytes a
  multiple of a line).

When writing new C2C code, match the existing idiom: guarded config `#define`s, `*_LOG`
macros, `fence` + `cache_evict_all` around every shared-memory touch, `commit_seq`-style
handshakes for multi-word payloads, and structured single-line log records
(`key=value` pairs) so runs can be parsed after the fact.

---

## Planning docs & skills

- Longer design/planning docs for C2C work go in `.claude/plans/` (e.g. one file per feature
  or investigation). Keep CLAUDE.md itself lean; link out to plans. Current plans:
  `001-full-c2c-kws-stream.md` (KWS streaming design), `002-kws-turn-taking-sync.md` (DONE — sync
  ported + demo works end-to-end), `003-kws-multi-testcase.md` (active next: validate more sample
  recordings, then MFCC-scale match / int8 conv2 fix).
- If a repeatable workflow emerges (e.g. "bring up a new C2C demo", "parse a transfer log"),
  capture it as a skill rather than re-explaining each time.

---

## Known Code Bugs / TODO fixes

Software defects to fix later (distinct from the hardware quirks below).

### int8 TinySpeech conv2 kernel produces garbage on silicon (use the float pipeline)
- **Where:** `bearly25-bmarks/tinyspeech-mc/src/tinyspeech_int8.c` — conv2 path (suspect the RVV
  microkernel `conv3x3_pool2x2_acc_c_rvv` / accumulation).
- **Issue:** the conv2 activation maximum `M2` comes back as **deterministic garbage ≈ `INT32_MAX`**
  (`M1`/`M3` sane), poisoning int8 calibration → wrong + non-deterministic predictions. Not
  multicore-specific (identical single-core). The Spike reference (host, no RVV) never exercises it.
- **Workaround:** run the **float pipeline** (`TINYSPEECH_INT8_PIPELINE=0`, CMake option
  `KWS_BEARLY_ROLLING_USE_FLOAT_PIPELINE=ON`). Correct, ~2× slower.
- **Action:** root-cause the RVV conv2 kernel (dump per-stage sums vs `tinyspeech_reference.h`
  `ref_stage_sums` to find the diverging layer). See "KWS accuracy / TinySpeech".
- **Status:** worked-around (float); int8 fix deferred.

### (RESOLVED) Consumer read path write-then-flush
- The turn-taking rewrite (plan 002) replaced the old `refresh_shared`/`poll_next_frame` path. BML
  now reads its own spad via `c2c_local_read_block_verify` (full flush, then read, then FNV checksum,
  bounded retries) only when the turn register grants it the turn — proven correct on silicon
  (checksum passes every case). No dummy-write-before-read was needed. Historical note kept for
  context; the "full cache flush every access" quirk below still governs all reads.

## Known Chip Bugs & Quirks

> **This is the living log.** Add an entry every time we discover something about how the
> silicon actually behaves — especially anything that constrains how C2C code must be written.
> Newest first. Keep each entry concrete: what we observed, on which chip, and what the code
> must do about it.

**Entry template:**

```
### [<CHIP>] Short title  (discovered YYYY-MM-DD)
- **Symptom:** what we observed.
- **Scope:** which chip(s), which demo/peripheral, conditions to reproduce.
- **Workaround / rule:** what C2C (or other) code must do because of it.
- **Status:** open / worked-around / fixed-in-hw / under-investigation.
```

### [C2C link] Cross-chip wake via CLINT MSIP works, but a single wake can drop and a dropped wake into a sleeping core is unrecoverable  (discovered 2026-07-12, proven on silicon)
- **Symptom:** a chip can wake a `wfi`-sleeping peer by writing the peer's CLINT MSIP across the
  link (`0x1_0200_0000` — own MSIP `0x0200_0000` with a leading 1). But a **single** MSIP write
  drops non-deterministically; when it does, the sleeping peer never wakes and the exchange
  deadlocks (nothing re-drives an edge into a sleeping core). Repeating the write helped (1 → a
  few exchanges) but never reached 100%.
- **Scope:** any cross-chip wake; both directions. Same unstable-write family as remote-spad writes.
- **Workaround / rule:** the **turn-register + timer** pattern (see "Reliable C2C turn-taking
  synchronization"). Wait in `wfi` with `mie.MSIE` + `mie.MTIE` set and `mstatus.MIE=0`; a
  periodic machine timer re-wakes the sleeper so it re-reads a persistent **turn register** in its
  own spad — recovering any dropped MSIP within one interval instead of deadlocking. Harden the
  turn-register + MSIP writes with repeats. Start DSP first (no retransmit; boot-init marks the
  turn as the peer's). CLINT: MSIP `+0x0000`, `mtimecmp` `+0x4000`, `mtime` `+0xBFF8`, MTIME_FREQ 50 kHz.
- **Status:** worked-around (proven reliable in `hello-wfi` / `*-hello-wfi`).

### [C2C link] Writing the cache-controller flush register does NOT evict — must force-evict  (discovered 2026-07-12, observed on silicon)
- **Symptom:** writing `1` to the cache-controller flush register (`0x02010200`) did not make a
  peer's cross-link write visible; the reader kept seeing stale data.
- **Scope:** every own-spad read after a remote write; both chips.
- **Workaround / rule:** keep using the **force-eviction buffer walk** (touch one byte per 64-byte
  line across a 256 KiB aligned buffer, several passes, + `fence rw,rw`) — `cache_evict_all` /
  `hwfi_cache_flush`. The register write is not a substitute.
- **Status:** worked-around (hardware behavior).

### [C2C link] A cross-link write to an absent/wedged peer hangs the writer  (discovered 2026-07-05, observed on silicon)
- **Symptom:** a chip hangs on boot (needs an FPGA reset to recover) if it writes the peer's spad
  while the peer is powered off or the link is wedged. BML hung rebooting with DSP off because its
  `app_init` did a cross-link write (the boot-clear) to `0xC` before anything confirmed DSP was up.
- **Scope:** any store to the *other* chip's spad. Local reads/writes of your own spad are safe.
- **Workaround / rule:** **do no cross-link (peer-spad) writes in `app_init`** — only local setup
  there. Defer every write to the peer's spad into `app_main`, after the core has fully booted and
  printed. (DSP's identity/epoch write and BML's ack are now the first peer-spad writes, both in
  `app_main`.) A powered, booted-but-idle peer is fine to write to; a powered-off/wedged one is not.
- **Status:** worked-around (moved all peer-spad writes out of init).

### [C2C link] A cross-link write into a chip while it is BOOTING kills that chip  (discovered 2026-07-05, observed on silicon)
- **Symptom:** whichever chip comes up **second** dies — prints nothing, and the FPGA has to be
  reset. It is **not** poll-vs-write contention and **not** about *continuous* writes: even a
  **single** write from the first chip into the second chip's spad, landing during the second's
  boot, kills it. Fully symmetric (DSP-first kills BML; BML-first kills DSP). Confirmed: DSP did
  one `publish` to `0xD` while BML was still booting (not yet polling) and BML's FPGA died.
- **Scope:** both chips; any store into the peer's spad while the peer has not finished booting.
- **Workaround / rule:** **boot barrier.** No chip writes the peer's spad until the peer signals
  it has fully booted. Implemented as: BML boots, waits a grace, then writes `bml_ready`
  (`KWS_STREAM_READY_MAGIC`) into the DSP spad; DSP polls its **local** `0xC` for `bml_ready` and
  does not touch `0xD` until it sees it. Also keep steady-state traffic **link-quiet** (write the
  peer's spad only in brief, infrequent bursts; poll your own spad in between —
  `KWS_DSP_ROLLING_ACK_POLL_BUDGET`, `KWS_BEARLY_ROLLING_REACK_EVERY`). Prefer starting **DSP
  first** (it waits indefinitely for `bml_ready`), then BML.
- **Status:** worked-around with a boot barrier. *(Open: what exactly the peer must reach before
  it can safely receive a write — is `bml_ready` after runtime prep enough, or is a longer settle
  needed?)*

### [C2C link] Scratchpads are 32-bit-access-only — byte/half accesses hang  (discovered 2026-07-05, confirmed on silicon)
- **Symptom:** a **byte-granular** store to a scratchpad (`0xC`/`0xD`) hangs the core. BML hung on
  the first byte of a `memcpy`-style block write to `0xC`; switching that path to 32-bit word
  stores fixed it and BML progressed all the way to inference.
- **Scope:** all spad accesses on both chips. Word (32-bit) accesses work; byte/half do not.
- **Workaround / rule:** only ever access the spads with **aligned 32-bit** loads/stores.
  `c2c_shm`'s block helpers assemble/disassemble bytes and move whole words; callers pass
  4-aligned addresses and 4-byte-multiple lengths. Never `memcpy` to/from a spad.
- **Status:** worked-around (hardware constraint). Likely also explains earlier "DSP published but
  BML never received" — the byte-wise payload write was mangling data sub-word.

### [C2C link] Shared-region access requires a full cache flush every time  (discovered 2026-07-05)
- **Symptom:** data written to / read from the `0xC0000000` shared region is not made visible
  across the die by an ordinary load/store. Stale data is returned otherwise.
- **Scope:** every access to the shared region on either chip — reads, writes, and polling
  loops alike. All C2C demos.
- **Workaround / rule:** any time we touch the shared region for any purpose, we must **write
  to the address and then flush the entire cache**. In code this is the `cache_evict_all()` /
  `cache_writeback_pressure()` full-cache buffer walk plus `fence rw, rw`, done after every
  access. Do not treat a plain load/store to `0xC0000000` as sufficient.
- **Status:** worked-around (hardware behavior; the flush is mandatory, not an optimization).

### [C2C link] Cross-spad (remote) writes are unstable — repeat them  (discovered 2026-07-05)
- **Symptom:** a single write into the *other* chip's scratchpad across the link is not reliable
  — it may not "take."
- **Scope:** writes into the remote spad (DSP→`0xD0000000`, BML→`0xC0000000`); all C2C demos.
  Writes to your own adjacent spad are local and stable.
- **Workaround / rule:** **repeat every remote-spad write several times** so it sticks. A value
  is not "sent" until it has been written repeatedly. Sync protocols must be idempotent
  (flags/counters that tolerate duplicate writes). Local reads of your own spad still need a full
  cache flush first (remote wrote it behind your cache).
- **Status:** open / worked-around. *(Quantify how many repeats are needed in practice.)*
