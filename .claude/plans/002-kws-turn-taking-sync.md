# 002 — Port the KWS demos to the proven turn-taking C2C sync

**Status:** DONE — validated on silicon 2026-07-22. Sync ported via `c2c-demos/common/c2c_turnsync.h`
and the full demo (DSP MFCC → C2C → BML TinySpeech → correct `yes`) works end-to-end. Beyond the
plain hello-wfi sync, KWS added **self-heal retransmit** keyed on `ack_index`/`case_index` (producer
re-grants case N on idle timer ticks until acked; consumer re-acks duplicate grants without
re-inferring) — streams indefinitely, recovering mid-stream dropped writes. Two accuracy fixes were
also required (NOT sync): MFCC coeff-major transpose in DSP, and running the float pipeline to dodge
a broken int8 conv2 kernel. See CLAUDE.md "KWS accuracy / TinySpeech" and plan 003 for follow-ups.
**Targets:** `c2c-demos/dsp-kws-rolling` (producer) and `c2c-demos/bearly-kws-rolling` (consumer).
**Reference implementation to copy from:** `c2c-demos/common/hello_wfi_link.h`.

## Goal

Replace the KWS demos' current synchronization (turn-taking + fixed-park + `commit_seq`, which had
a first-case wake race and non-deterministic wedging) with the **reliable turn-register + CLINT
MSIP wake + timer safety net + hardened writes** pattern from `hello_wfi_link.h`. Keep the
producer/consumer roles and the payload path (DSP computes MFCC cases; BML runs TinySpeech).

## What stays the same

- DSP produces cases, BML consumes and infers. Data direction unchanged.
- The shared region / two scratchpads (`0xC0000000` DSP-adjacent, `0xD0000000` BML-adjacent) stay
  the transport.
- Golden access rule (read only your own spad, write both), 32-bit-only spad access, force-eviction
  cache flush, boot barrier / no cross-link writes in `app_init`.
- The 1128-byte MFCC case payload (`KWS_CASE_PAYLOAD_BYTES`), checksum verification.

## What changes — the sync

Adopt the three layers from `hello_wfi_link.h`:

1. **Turn register** at a fixed spad offset (e.g. `0x00`): `0 = DSP's turn`, `1 = BML's turn`.
   Each chip reads it from its OWN spad; acts only when it is its turn.
   - DSP's turn = "produce + publish the next case". BML's turn = "read + verify + infer + hand back".
2. **CLINT timer safety net:** both cores arm `MTIE`+`MSIE`, `mstatus.MIE=0`; a sleeper re-checks
   its turn register every ~50 ms even if a wake MSIP dropped.
3. **Hardened cross-link writes:** repeat every store (payload words, turn register, MSIP) N times
   + flush.

### Handoff mapping

- **DSP → BML (new case):** write case payload + checksum + `case_index` into BML's spad
  (`0x1_D000_0000`), then set turn register in BML's spad = BML, set own turn = BML, raise BML MSIP.
- **BML → DSP (ack / ready):** write result (pred class/score, cycle) into DSP's spad
  (`0x1_C000_0000`), set turn register in DSP's spad = DSP, set own turn = DSP, raise DSP MSIP.
- Commit order: payload/data first, **turn register last** (before MSIP), exactly as
  `hwfi_handoff` does.

### Boot / start order

- Each chip clears its own spad control block + turn register on boot, setting its own turn to the
  PEER (so it stays parked until handed to). Keep the existing `bml_ready` boot barrier so no chip
  writes a still-booting peer.
- **Start DSP first** (no retransmit; matches hello-wfi note). BML, being the consumer, likely
  takes the first turn after the barrier — decide who owns the very first turn during implementation
  (probably BML arms first / DSP holds the initial turn to publish case 1).

## Steps

1. Factor the `hwfi_*` helpers into a shared C2C sync module usable by KWS (either include
   `hello_wfi_link.h` directly, or lift its turn-register/timer/handoff helpers into
   `c2c-demos/common/` alongside `c2c_shm.c`). Prefer reuse over reimplementation.
2. Extend the KWS spad layout (`kws_stream_proto.h`) with a turn register at a fixed offset; keep
   the existing payload/checksum/case_index fields.
3. Replace the DSP producer loop: on DSP's turn → compute/publish case → handoff to BML.
4. Replace the BML consumer loop: on BML's turn → read+verify+infer → handoff (with result) to DSP.
5. Bring up on silicon incrementally (static payload first, like the current milestone), confirm
   many cases stream with no wedge, then re-enable per-case payload / VAD.

## Open items

- Who holds the very first turn (initial case) and how it interacts with the `bml_ready` barrier.
- Whether to keep `commit_seq`/checksum in addition to the turn register (belt-and-suspenders for a
  torn payload) — likely yes for the payload, with the turn register as the interlock.
- Tune `WRITE_REPEATS` / timer interval for the larger payload traffic.
