# 001 — Full C2C KWS streaming demo (DSP → BML, looping, gated)

**Status:** draft plan, awaiting review. No code yet.
**Targets:** evolve the existing pair **in place** — `c2c-demos/dsp-kws-rolling` (producer) and
`c2c-demos/bearly-kws-rolling` (consumer). Shared transport/cache workarounds get factored into
`c2c-demos/common/` so both sides use one implementation.

Related: bug log in `/CLAUDE.md` (full-cache-flush-per-access; unstable reads/writes; consumer
read-path TODO). All of those constrain this design.

### Decisions locked (from review)
- Evolve `*-kws-rolling` in place (not a new target).
- On trigger, transfer a **pre-roll window snapshot** (DSP always keeps the last 94 frames).
- VAD gate uses **both**: raw-audio energy to trigger, MFCC energy to confirm.
- **Single payload slot, hold-until-ack** for v1 (double-buffer deferred).
- **Two-window flag handshake** (finalized): `0xC0000000` = BML's inbox (DSP writes data +
  data-ready flag); `0xD0000000` = DSP's inbox (BML writes "ready for more" flag). See §4.

### Behavior change note
This repurposes the rolling demo: today DSP streams frame-by-frame and BML keeps the rolling
window + infers every frame. After this change, **the rolling window lives on the DSP side**,
and BML receives whole gated **cases** and infers once per case. This is a real semantic change
to a working demo — call it out in the commit and keep the old logs/flags recognizable.

---

## 1. Goal

End-to-end continuous keyword-spotting across the two chips:

- **DSP (producer):** raw audio → MFCC → maintains a rolling 94-frame feature window → a
  **threshold/VAD gate** decides if the current window is "interesting" → if so, transfers the
  **full case (94 × 12 = 1128 int8, `KWS_CASE_PAYLOAD_BYTES`)** into the shared region.
- **BML (consumer):** waits for a committed case, reads + verifies it, runs **TinySpeech
  inference**, acks.
- **Both loop forever, no reset / reload** between cases.
- Transport is the shared region at **`0xC0000000`** (non-negotiable); every access obeys the
  hardware rules (full flush per access; repeat unstable writes; integrity-checked reads).

### Phasing
1. **P0 — Static audio.** Raw audio compiled into the DSP binary (reuse the embedded
   `yes_test_005` signal already in `dsp-kws-rolling`). Prove the full pipeline.
2. **P1 — I2S audio.** Replace the static source with live **I2S** capture (`hal_i2s` + `hal_dma`)
   behind the same audio-source interface, so nothing downstream changes.

(“~96×12” from the brief = 94 frames × 12 MFCC dims = `KWS_CASE_PAYLOAD_BYTES` = 1128 B.)

---

## 2. System data flow

```
  ┌────────────────────────── DSP 25 (producer) ──────────────────────────┐
  │ audio_source ─▶ MFCC ─▶ quantize ─▶ rolling window[94][12] (pre-roll)  │
  │      ▲                                   │                             │
  │   P0: static buf                         ├─▶ raw-energy trigger        │
  │   P1: I2S+DMA                            └─▶ MFCC-energy confirm ─▶ gate│
  │                                                     │ (fire)           │
  │                                                     ▼                  │
  │        publish_case(): snapshot window, remote-write payload ×N into    │
  │        BML spad @0xD + checksum + flush, then set data_ready @0xD ×N     │
  └──────────────────────────────────────────────────┼───────────────────┘
   DSP writes ACROSS link into BML spad @0xD0000000   │  (payload + data_ready)
                                                      ▼
   BML reads its OWN spad @0xD0000000 (local, flush-first) ─▶ ... ─▶
   BML writes ACROSS link into DSP spad @0xC0000000 (rx_ready + result) ─┐
                                                      ▲                   │
   DSP reads its OWN spad @0xC0000000 (local, flush-first) for rx_ready ◀─┘
  ┌───────────────────────────────────────────────────────────────────────┐
  │ poll data_ready@0xD (own spad) ─▶ read+verify payload ─▶ TinySpeech ─▶  │
  │ clear data_ready@0xD (own spad) ─▶ remote-set rx_ready@0xC ×N (+result) │
  └────────────────────────── BML / Bearly 25 (consumer) ─────────────────┘
```

**Access rule (hardware, see `/CLAUDE.md`):** a chip may **read only its own adjacent spad**
and may **write to both**. So all data/flags a chip must *read* live in *its own* spad, placed
there by the other chip's cross-link write. Cross-link (remote) writes are the unstable ones →
repeated. Local reads still need a full flush first.

---

## 3. Shared-region layout (two windows)

New protocol header: `c2c-demos/common/kws_stream_proto.h` (built alongside the existing
`kws_proto.h` / `kws_rolling_proto.h`). **Single payload slot, hold-until-ready** — DSP does not
overwrite the slot until BML has signalled it is ready for more, so BML can re-read the payload
freely to defeat unstable transfers.

### BML-adjacent spad `0xD0000000` — DSP→BML data (BML reads locally; DSP remote-writes)

```
offset  size   field                        writer→reader          purpose
------  -----  ---------------------------  ---------------------  ---------------------------
0x0000  4      magic  'KWSD'                 DSP → BML              protocol id (sanity)
0x0004  4      version                       DSP → BML              KWS_STREAM_PROTO_VERSION
0x0008  4      payload_bytes                 DSP → BML              = KWS_CASE_PAYLOAD_BYTES (1128)
0x000C  4      payload_checksum              DSP → BML              integrity of the case in slot
0x0010  4      data_ready                    DSP set / BML clear    SET = case valid, 0 = consumed
0x0014  4      case_index                    DSP → BML              ++ per case (telemetry/debug)
0x0018  8      dsp_tx_cycle                  DSP → BML              rdcycle at commit (telemetry)
...     ...    reserved / debug counters
0x0040  1128   case_payload[94*12] int8      DSP → BML              the MFCC feature map
```
DSP's writes here are **remote/cross-link → repeated ×N**. BML's read is local (flush-first).
BML clearing `data_ready` is a **local** write into its own spad.

### DSP-adjacent spad `0xC0000000` — BML→DSP sync (DSP reads locally; BML remote-writes)

```
offset  size   field                        writer→reader          purpose
------  -----  ---------------------------  ---------------------  ---------------------------
0x0000  4      magic  'KWSC'                 BML → DSP              protocol id (sanity)
0x0004  4      rx_ready                      BML set / DSP clear    SET = BML ready for next case
0x0008  4      bml_pred_class                BML → DSP              last inference result (optional)
0x000C  4      bml_pred_score_q              BML → DSP              quantized score (optional)
0x0010  8      bml_rx_cycle                  BML → DSP              rdcycle at ack (telemetry)
...     ...    reserved / debug counters
```
BML's writes here are **remote/cross-link → repeated ×N**. DSP's read is local (flush-first).
DSP clearing `rx_ready` is a **local** write into its own spad.

`data_ready` / `rx_ready` sit on their own cache lines, separate from the payload, so flushing
one doesn't entangle the other. To harden against the unstable remote-write, a flag "set" uses a
distinctive value (`KWS_STREAM_FLAG_SET`, e.g. `0xA5A5A5A5`) rather than bare `1`, and the local
reader requires it to appear **stably across two fresh (flushed) reads** before acting.
`case_index` + `payload_checksum` let BML detect a torn/stale slot even though the interlock is
flag-based.

---

## 4. Synchronization protocol (two-window flags)

Flags carry the handshake (finalized design). Each flag lives in the spad of the chip that
**reads** it (so that chip can read locally); the other chip **remote-writes** it:
- `data_ready` @ `0xD0000000` (BML's spad) — "a valid case is in the slot." DSP remote-sets;
  BML local-reads and local-clears.
- `rx_ready` @ `0xC0000000` (DSP's spad) — "BML ready for the next case." BML remote-sets; DSP
  local-reads and local-clears.

**Invariant (hold-until-ready):** DSP writes a new case only after it has seen `rx_ready == SET`.
While `data_ready == SET` and BML hasn't cleared it, the slot is BML's to read/re-read.

### Startup
- DSP: init `0xD` spad (magic/version, `data_ready = 0`, `case_index = 0`) via remote-writes ×N.
- BML: init `0xC` spad (magic, `bml_pred_* = 0`), then **remote-set `rx_ready = SET` @0xC** ×N so
  DSP may send the first case.

### Producer (DSP) loop
1. Wait until `rx_ready` @0xC (own spad, flush+read, stable across two reads) `== SET`.
2. Clear `rx_ready` @0xC `= 0` (local write, own spad).  *(inferred necessary step — see open
   item Q7; without it DSP can't tell rounds apart.)*
3. `case_index++`; **remote-write** into `0xD`: `case_payload` (×N) → `payload_checksum` →
   `payload_bytes`, full flush.
4. **Remote-set** `data_ready` @0xD `= SET` (×N) + full flush. Written **last**; never reorder
   before the payload.
5. goto 1.

### Consumer (BML) loop
1. Wait until `data_ready` @0xD (own spad, flush+read, stable across two reads) `== SET`.
2. Read `case_payload` @0xD (own spad, flush-first); verify `payload_checksum`. Mismatch →
   re-read (DSP is holding the slot); bounded retries → log + count drop, then release the slot.
3. Run TinySpeech inference; optionally **remote-write** `bml_pred_*` @0xC (×N).
4. Clear `data_ready` @0xD `= 0` (local write, own spad).
5. **Remote-set** `rx_ready` @0xC `= SET` (×N) + full flush.
6. goto 1.

### Loop / no-reset
Both loop forever; the flag handshake is inherently repeatable, so the demo re-arms each round
with no reset/reload. A `done` path is optional; default is run-forever.

---

## 5. DSP producer design

- **Audio-source interface** (the P0/P1 seam):
  ```c
  void      audio_source_init(void);
  uint32_t  audio_source_read(float32_t *dst, uint32_t n_samples);  // next audio hop
  ```
  - P0: replays the embedded static signal (as `dsp-kws-rolling` does via `load_yes005_window` /
    `g_kws_dsp_yes005_signal`), looping.
  - P1: pulls from an I2S+DMA ring (§8). Downstream code unchanged.
- **MFCC:** reuse `mfcc_driver` (`mfcc_driver_run_sp1024x23x12_f32` path already wired), then
  `quantize_mfcc` → int8.
- **Rolling pre-roll window in TCM:** always keep the last 94 frames, so a trigger snapshots a
  case that *includes* the audio that crossed the threshold. (This is the current BML-side window
  logic moved to DSP.)
- **Gate (§7).** On fire + slot-free → snapshot window → `publish_case()`.
- **`publish_case()`:** the §4 producer commit using the shared-transfer helper (§6).

## 6. Shared-transfer helper (factored, respects the bugs)

New module `c2c-demos/common/c2c_shm.{h,c}`, added to each target's sources (like
`simple_setup.c`), so the workarounds live in exactly one place. The API splits by the hardware
access rule (remote-write / local-write / local-read):

- `c2c_full_flush()` — the 256 KiB / multi-pass buffer walk + `fence rw, rw` (unify today's
  `cache_evict_all` / `cache_writeback_pressure`). Owns the static aligned evict buffer.
- **Remote (cross-link) writes — repeated ×N** (into the *other* chip's spad):
  - `c2c_remote_write_u32(addr, val)` — write `val` **N times** with fences, then full flush.
  - `c2c_remote_write_block(dst, src, bytes)` — repeat-write payload ×N, then full flush.
- **Local writes — single** (into your *own* spad, e.g. clearing your own flag):
  - `c2c_local_write_u32(addr, val)` — write once + fence + full flush.
- **Local reads — flush-first** (from your *own* spad, which the remote wrote behind your cache):
  - `c2c_local_read_u32(addr)` — full flush, then load.
  - `c2c_local_read_u32_stable(addr)` — flush+read twice until two fresh reads agree (bounded).
  - `c2c_local_read_block_verify(dst, src, bytes, expect_checksum)` — flush, copy, checksum,
    bounded retry; returns ok/failed.
- `c2c_checksum(buf, bytes)` — cheap sum/CRC.

**Read-path bug note:** the earlier "consumer read-path" TODO assumed a dummy write to the
*target* address before reading. Under the confirmed spad model that would clobber the
remote-written data, so it's wrong for locations the remote owns. The correct local-read recipe
is **flush-then-read** (what `c2c_local_read_*` do). The helper centralizes this; confirm on
silicon that flush-then-read alone is sufficient for a poll (no target write needed).

Config knobs: `C2C_WRITE_REPEATS` (N), `C2C_READ_RETRIES`, flush geometry (`CACHE_LINE_BYTES`,
`EVICT_BYTES`, `EVICT_PASSES`). `N`'s real value needs both chips (milestone 2/3); milestone 1
lands the knob + a single-chip build/smoke check.

## 7. Threshold / VAD gate (both: energy trigger + MFCC confirm)

Only spend link bandwidth + BML inference on audio worth classifying.

- **Trigger:** short-term **raw-audio energy** (sum of squares) of the incoming hop, computed
  before/around MFCC.
- **Confirm:** **MFCC energy** (MFCC[0] / log-energy, already computed) must also exceed its
  threshold before a case is committed — rejects transients that spike raw energy but aren’t
  speech-like.
- **Debounce/refractory (configurable):**
  - `VAD_RAW_ENERGY_THRESHOLD`, `VAD_MFCC_ENERGY_THRESHOLD`
  - `VAD_MIN_ACTIVE_HOPS` — consecutive active hops required before firing.
  - `VAD_REFRACTORY_HOPS` — suppress re-fire after a transfer so one utterance ≠ many transfers.
- Fire when: raw-energy active for ≥ `MIN_ACTIVE_HOPS` **and** MFCC-energy confirms **and** window
  full **and** slot free → snapshot + `publish_case()`. Else keep ingesting; no transfer, no
  `dsp_case_seq` bump.
- Telemetry: log gate decisions (raw active? mfcc confirm? fired/suppressed) to tune thresholds.

## 8. P1 — I2S input (later phase)

- DSP `chip_config.h` exposes `I2S_BASE`; `hal_i2s` + `hal_dma` already included.
- Configure I2S RX → DMA into a double-buffered ring; `audio_source_read()` returns the newest
  hop as `float32_t` for `mfcc_driver`.
- Pin down at that phase: sample rate/bit depth, mono/stereo, DMA buffer vs MFCC hop
  (`SIGNAL_HOP_SAMPLES`), overrun handling. Interface stays identical to P0.

---

## 9. Deferred / optional
- **Double-buffer payload** (2 slots) to overlap DSP producing case N+1 while BML infers N. Do
  only if single-slot throughput is insufficient.
- **Result back-channel:** BML → DSP predicted class/score (`bml_pred_*` reserved). Open item —
  ack-only vs also react on DSP (e.g. GPIO/LED on keyword). Default v1: fields written but DSP
  doesn’t act on them.

---

## 10. Build-system integration
- Add `c2c-demos/common/kws_stream_proto.h` and `c2c_shm.{h,c}`.
- Edit existing `dsp-kws-rolling` / `bearly-kws-rolling` targets:
  - DSP: keep glossy + mfcc-lib links; add the pre-roll window + gate + new protocol.
  - BML: keep glossy + TinySpeech/tensor runtime; switch from frame-streaming to case receive.
  - Include `common/` for the new headers (already on the include path pattern).
- Build/run (per `/CLAUDE.md`): `make build CHIP=dsp25 TARGET=dsp-kws-rolling`, likewise bearly;
  flash each via `make tsi-run TTY=… BINARY=…`; start consumer first (producer has startup grace).

---

## 11. Milestones (incremental bring-up order)
1. **Shared-transfer helper** (`c2c_shm`) + poke via an existing simple test / `c2c-measure`;
   fix the consumer read-path bug here; measure a workable `C2C_WRITE_REPEATS`.
2. **Single-shot full-case transfer + inference** (no gate, no loop): one 1128 B case DSP→BML,
   checksum-verified, one TinySpeech run.
3. **Loop with seq/ack hold-until-ack:** many cases, no reset; watch drops/torn commits.
4. **VAD gate (energy + MFCC confirm):** only active audio transferred; tune knobs from logs.
5. **P1 I2S source** swapped behind `audio_source_*`.

---

## 12. Remaining open items (non-blocking)
- **Q7 — DSP clearing `rx_ready`:** the finalized description has BML set `rx_ready` @0xD but
  didn't state who clears it. Plan assumes **DSP clears it** after observing (§4 step 2), which
  is required to distinguish rounds. Confirm, or specify an alternative (e.g. BML re-clears at
  start of each inference).
- **Result back-channel:** ack-only or also react on DSP? (default: write `bml_pred_*`, don't act.)
- **Write-repeat `N` / read-retry bound:** true cross-die stability can only be measured with
  **both chips running** (milestone 2/3). Milestone 1 lands the tunable knob + logging; the
  empirical value is pinned once the paired transfer runs.
