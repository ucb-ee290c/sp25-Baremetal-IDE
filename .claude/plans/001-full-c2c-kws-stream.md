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
  │                    publish_case(): snapshot window, write payload ×N   │
  │                    + checksum + full flush, then bump dsp_case_seq      │
  └──────────────────────────────────────────────────┼───────────────────┘
                                                      │ shared region @ 0xC0000000
       dsp_case_seq / payload / checksum  ───────────▶│◀──────── bml_ack_seq / (result)
                                                      │
  ┌───────────────────────────────────────────────────┼──────────────────┐
  │ poll dsp_case_seq ─▶ read+verify payload ─▶ TinySpeech ─▶ write ack    │
  └────────────────────────── BML / Bearly 25 (consumer) ─────────────────┘
```

---

## 3. Shared-region layout (`0xC0000000`, 16 KiB)

New protocol header: `c2c-demos/common/kws_stream_proto.h` (built alongside the existing
`kws_proto.h` / `kws_rolling_proto.h`). One control block + one payload slot, **single slot with
hold-until-ack** (producer never overwrites until the consumer acks), so the consumer can re-read
freely to defeat unstable transfers.

```
offset  size   field                        owner    purpose
------  -----  ---------------------------  -------  ------------------------------------
0x0000  4      magic  'KWSS'                 DSP      protocol id
0x0004  4      version                       DSP      KWS_STREAM_PROTO_VERSION
0x0008  4      dsp_flags                     DSP      alive / streaming / done
0x000C  4      dsp_case_seq                  DSP      ++ per committed case  (READY signal)
0x0010  4      payload_bytes                 DSP      = KWS_CASE_PAYLOAD_BYTES (1128)
0x0014  4      payload_checksum              DSP      integrity of committed case
0x0018  8      dsp_tx_cycle                  DSP      rdcycle at commit (telemetry)
0x0020  4      bml_flags                     BML      ready / running / done
0x0024  4      bml_ack_seq                   BML      last case consumed (ACK signal)
0x0028  4      bml_pred_class                BML      inference result (optional back-channel)
0x002C  4      bml_pred_score_q              BML      quantized score (optional)
0x0030  8      bml_rx_cycle                  BML      rdcycle at ack (telemetry)
...     ...    reserved / debug counters
0x0040  1128   case_payload[94*12] int8      DSP      the MFCC feature map
```

Control words sit on their own cache line(s), separate from the payload, so flushing one doesn’t
entangle the other. Every control field is a single word written with the repeat+flush helper.

---

## 4. Synchronization protocol

Two 32-bit counters carry the whole handshake:
- `dsp_case_seq` — producer: "case N is committed."
- `bml_ack_seq` — consumer: "I consumed case N."

**Invariant (hold-until-ack):** producer commits case `N` only when `bml_ack_seq == N-1`. Until
the consumer acks `N`, the payload is left untouched, so the consumer may re-read it as needed.

### Producer commit (order matters)
1. Wait until `bml_ack_seq == dsp_case_seq` (slot free).
2. Write `case_payload` (repeat ×N) → `payload_checksum` → `payload_bytes`.
3. **Full flush.**
4. Write `dsp_case_seq = N` (repeat ×N). **Full flush.** `dsp_case_seq` is written *last*; never
   reorder before the payload.

### Consumer receive
1. Poll `dsp_case_seq` (read discipline §6). `dsp_case_seq != bml_ack_seq` → candidate case.
2. Read `case_payload`; checksum vs `payload_checksum`.
   - Mismatch → re-read (producer is holding the slot, so retry is safe). Bounded retries → log,
     count as drop, ack anyway to release the slot.
3. Re-read `dsp_case_seq`; if it changed mid-read, restart (torn-commit guard, same `seq0==seq1`
   idea already in the rolling consumer).
4. Run inference; optionally write `bml_pred_*`.
5. Write `bml_ack_seq = dsp_case_seq` (repeat ×N) + full flush.

### Loop / no-reset
Both loop forever. Startup handshake makes re-entry safe: consumer waits for valid `magic`/
`version` + `dsp_flags.alive`; producer waits for `bml_flags.ready`. Because sync is seq/ack
based (not one-shot), the demo repeats without reload. `done` flag path optional; default is
run-forever.

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

New module `c2c-demos/common/c2c_shm.h` (+ `.c` if not header-only), used by both sides so the
workarounds live in exactly one place:

- `c2c_full_flush()` — the 256 KiB / multi-pass buffer walk + `fence rw, rw` (unify today’s
  `cache_evict_all` / `cache_writeback_pressure`).
- `c2c_write_u32_stable(addr, val)` — write `val` **N times** with fences, then `c2c_full_flush()`.
- `c2c_write_block_stable(dst, src, bytes)` — repeat-write payload, then flush.
- `c2c_read_u32_fresh(addr)` — enforce the read rule (**fix the consumer read-path bug here**:
  perform the required write-then-flush before the load), then read.
- `c2c_read_block_verify(dst, src, bytes, expect_checksum)` — flush, read, checksum, bounded retry.
- `c2c_checksum(buf, bytes)` — cheap sum/CRC.

Config knobs: `C2C_WRITE_REPEATS` (N), `C2C_READ_RETRIES`, flush geometry (`CACHE_LINE_BYTES`,
`EVICT_BYTES`, `EVICT_PASSES`). Tune N empirically (milestone 1; bug log asks to quantify repeats).

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
- **Result back-channel:** ack-only or also react on DSP? (default: write fields, don’t act.)
- **Write-repeat `N` / read-retry bound:** measure in milestone 1 unless you already have values.
