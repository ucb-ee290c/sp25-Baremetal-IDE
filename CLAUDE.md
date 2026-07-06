# CLAUDE.md

Working notes for Claude when developing on this repo. This is the **baremetal bringup
environment** for two SP25 chips: **Bearly ML 25** and **DSP 25**. Primary current focus
is the **chip-to-chip (C2C) test suite** under `c2c-demos/`.

Read the "Known Chip Bugs & Quirks" section before writing or changing any C2C code — it is
the living record of hardware behavior we've discovered on silicon. Respect it.

### Current focus

- **Actively developing the rolling KWS demo** — `c2c-demos/dsp-kws-rolling` (producer) and
  `c2c-demos/bearly-kws-rolling` (consumer). DSP streams quantized MFCC frames through the
  shared region; Bearly maintains a rolling window in TCM and runs TinySpeech inference.
- **Goal: make the C2C link bidirectional with easier synchronization.** Today each chip runs
  its own program and they talk one-way through the shared region, with only a minimal
  back-channel (Bearly writes a `done_marker` at `+0x10`). We want a cleaner two-way sync.
  - **Producer/consumer roles do NOT change** — DSP produces, Bearly consumes. Only the
    synchronization gets easier/bidirectional.
  - **The shared region at `0xC0000000` is non-negotiable** — it stays the transport. The
    internal layout/protocol within it is open to redesign during planning.

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

- The two chips communicate through a **shared memory window at `0xC0000000`**
  (`*_SHARED_BASE_ADDR`, default `0xC0000000`, `*_SHARED_BYTES` = 16 KiB). Both chips address
  the same window; the C2C link routes accesses across the die boundary.
- **DSP writes, Bearly reads.** The DSP side is the producer; Bearly polls for changes.
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
`c2c-transfer-dsp`, `c2c-transfer-bearly`. Each `<chip>-*` target must be built with the
matching `CHIP=`. Output ELF lands in `build/c2c-demos/<target>/<target>.elf`.

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
  or investigation). Keep CLAUDE.md itself lean; link out to plans.
- If a repeatable workflow emerges (e.g. "bring up a new C2C demo", "parse a transfer log"),
  capture it as a skill rather than re-explaining each time.

---

## Known Code Bugs / TODO fixes

Software defects to fix later (distinct from the hardware quirks below).

### Consumer read path may not honor the "write-then-flush" rule
- **Where:** `c2c-demos/bearly-kws-rolling/src/main.c` — `refresh_shared()` / `poll_next_frame()`
  (and the analogous read paths in other consumer demos).
- **Issue:** per the hardware rule (see "full cache flush" quirk), every shared-region access
  must **write to the address, then flush the entire cache**. The read path currently does the
  full-cache-evict buffer walk and *then reads*, without a dummy write to the target address
  first. It may be relying on the evict walk alone and not fully honoring the rule.
- **Action:** confirm on silicon whether a dummy write before the read is required; if so, fix
  all consumer read paths. Track in the bidirectional-link redesign.
- **Status:** to fix.

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

### [C2C link] Shared-region reads/writes are unstable — repeat writes  (discovered 2026-07-05)
- **Symptom:** an individual read or write across the link is not reliable; a single write may
  not "take," and a single read may return a wrong/torn value.
- **Scope:** every shared-region access on either chip; all C2C demos.
- **Workaround / rule:** **write the intended data several times** (and/or re-read and confirm)
  rather than assuming one access succeeds. Sync protocols must tolerate this — prefer
  idempotent writes and confirm-by-reread over single-shot handshakes. Keep this in mind when
  designing the bidirectional link: a value is not "sent" until it has been repeated/confirmed.
- **Status:** open / worked-around. *(Quantify how many repeats are needed in practice.)*
