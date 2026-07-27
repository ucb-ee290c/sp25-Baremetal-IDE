# Plan 004 — Dual-core KWS + Llama voice control (Bearly)

**Status:** implemented + builds clean; NOT yet validated on silicon (2026-07-25).

## Goal

On the Bearly chip, run TinyLlama (borai int8) inference on **hart 1** while the C2C **KWS receiver
runs on hart 0**. A confidently-recognized keyword controls the Llama core:

- **START** (`yes`, `go`, `on`) → begin/continue Llama generation on hart 1.
- **STOP** (`no`, `off`, `stop`) → abort generation mid-stream and park (halt) hart 1.

Continuous generation: one START keeps streaming tokens (re-arming a new story each time one
finishes) until a STOP. Only predictions above the confidence gate
(`KWS_BEARLY_ROLLING_MIN_SCORE`, top-logit > 3.0) act.

## Core assignment (decided)

Hart 0 = KWS + C2C link + UART + **controller**; hart 1 = Llama. The C2C link (CLINT MSIP/timer +
spad) and init already live on the boot hart (hart 0); moving them to hart 1 would mean re-pointing
the cross-chip wake MSIP for no behavioral gain. "Halt core 0" from the original ask = halt the
Llama core (hart 1).

## Build architecture — ONE binary, two reused source units

The combined target **`c2c-demos/bearly-kws-llama`** compiles, unmodified in behavior, the two proven
source files, each guarded by an integration macro (inert in their standalone builds):

- `c2c-demos/bearly-kws-rolling/src/main.c`  → `-DKWS_BEARLY_LLAMA`   (hart 0: KWS + controller; owns `main`/`app_init`/`app_main` + a strong `__main`)
- `bearly25-demos/borai/int8/src/main.c`     → `-DKWS_LLAMA_COMBINED` (hart 1: Llama; its `main`/`__main`/`app_main`/`target_frequency` are #ifdef'd out, `softmax`→`llama_softmax`, exposes `llama_build()`/`llama_run_forever()`/`g_llama_stop`)

plus the TinySpeech float runtime, `c2c_shm.c`, `simple_setup.c`, `vecnn`, `glossy`.

### Collisions resolved
- **`softmax`** (both TUs export it): borai's is `#define softmax llama_softmax` under the guard.
- **`main`/`__main`/`app_main`/`target_frequency`**: borai's are `#ifndef KWS_LLAMA_COMBINED`. The
  KWS TU provides `main`/`app_init`/`app_main` and a **strong `__main`** (hart-1 entry). No
  `bmark-lib/hthread.c` is compiled (its `__main` would clash).
- **Header name clashes** (`main.h`, `weights.h` exist in both trees): borai reaches its own headers
  via a uniquely-named wrapper `borai_main.h` (quoted-include resolves from its own dir), and borai's
  include dir is listed **last** so tinyspeech's `weights.h`/`main.h` win for the tinyspeech TU.

## Dual-core mechanism

- **hart-1 entry:** strong `__main()` in the KWS TU. hart 1 spin-waits on `g_llama_run`
  (boraiq-style spin, proven on this silicon — avoids relying on a wfi/MSIP wake for the worker),
  runs `llama_run_forever()` when asked, parks when done. Idle hart 1 spins (fine for a demo).
- **control flags** (plain cached DRAM + `__sync_synchronize()` fences; intra-die is coherent — NOT
  the C2C spad): `g_llama_run` (hart0→hart1 intent), `g_llama_stop` (hart0→generate abort, polled per
  token), `g_llama_active`/`g_llama_ready` (liveness/build gate).
- **build once:** hart 0 calls `llama_build()` in `app_init` (before the KWS loop / any run request),
  then sets `g_llama_ready`. No half-built model, no build/inference malloc race.
- **SMP-safe malloc:** newlib `__malloc_lock`/`__malloc_unlock` implemented as an owner-hart
  recursive spinlock (`__sync_val_compare_and_swap`), because hart 0 (TinySpeech) and hart 1 (Llama
  `generate()` prompt-token / tokenizer allocs) both touch the shared heap.
- **control point:** in the KWS loop, right after inference + the confidence gate, map
  `g_last_pred_class` → START/STOP and set the flags (idempotent).
- **abort point:** borai `generate()`'s `while (pos < steps)` polls `g_llama_stop` once per token.

## Build & run

```bash
# BML (dual-core KWS + Llama)
make build CHIP=bearly25 PLATFORM=CHIP TARGET=bearly-kws-llama EXTRA_CMAKE_ARGS="-DBUILD_VECNN=ON"
make tsi-run TTY=<bml-tty> BINARY=build/c2c-demos/bearly-kws-llama/bearly-kws-llama.elf

# DSP (unchanged mic producer)
make build CHIP=dsp25 PLATFORM=CHIP TARGET=dsp-kws-rolling \
  EXTRA_CMAKE_ARGS="-DBUILD_MFCC_LIB=ON -DKWS_DSP_ROLLING_USE_MIC=ON -DKWS_DSP_ROLLING_MULTI_SIGNAL=OFF"
```
Start DSP first, then BML. Speak `yes`/`go`/`on` → Llama starts streaming a story on hart 1;
`no`/`off`/`stop` → it aborts and halts.

## Open / verify-on-silicon
- **UART interleaving:** hart 0 (KWS logs) and hart 1 (Llama token stream) both `printf`. Output will
  interleave; if it *corrupts* (non-reentrant `_write`), add a UART spinlock mirroring the malloc lock.
- **Vector unit sharing:** if the two harts share one RVV unit, TinySpeech (hart0) + Llama (hart1)
  contend; correctness should hold via per-hart `mstatus.VS`, but confirm.
- **hart-1 spin at idle** burns a core before the first START (acceptable; could wfi later).
- **Operating frequency** is the KWS value (`KWS_BEARLY_ROLLING_TARGET_FREQUENCY_HZ`), not borai's 1 GHz.
- First START pays the model-build latency only if build were lazy — here it's at init, so START is instant.
