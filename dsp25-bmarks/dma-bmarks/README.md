# dma-bmarks

DMA-focused memcpy bandwidth benchmark suite for DSP25.

This suite is intentionally separate from `bearly25-bmarks/bandwidth-bmarks` and is
meant to provide a directly comparable DMA-only dataset.

## What it measures

- Transfer cases (no TCM):
  - `DRAM->DRAM`
  - `DRAM->Scratchpad`
  - `Scratchpad->DRAM`
- Cache states:
  - `COLD`, `WARM_SRC`, `WARM_DST`, `WARM_BOTH`, `HOT_REPEAT`
- DMA execution modes:
  - single-channel
  - multi-channel (work split across core DMA channels)

For every row it reports:
- best/avg `cycles`
- best/avg `ns`
- best/avg `MB/s`
- DMA overhead split:
  - `setup_avg` cycles (program + start)
  - `xfer_avg` cycles (wait/transfer)
- correctness `PASS`/`FAIL`

## Configuration

Edit `include/bench_config.h`.

Important toggles:
- `DMA_BENCH_ENABLE_SINGLE_CHANNEL`
- `DMA_BENCH_ENABLE_MULTI_CHANNEL`
- `DMA_BENCH_MULTI_CHANNELS`
- `DMA_BENCH_ENABLE_CASE_*`
- `DMA_BENCH_ENABLE_CACHE_*`
- `DMA_BENCH_LOGW`
- `DMA_BENCH_ENABLE_PLL_SWEEP`
- `DMA_BENCH_PLL_FREQ_LIST`

## Build target

Target name: `dma-bmarks`.
