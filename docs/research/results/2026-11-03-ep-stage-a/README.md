# Raw results, expert-parallelism Stage A (3 Nov 2026)

Pixel 7 Pro (Tensor G2, Mali-G710, Android 13), `MobileMoE-M-QAT-q4_0.gguf`. Analysis:
[../../ep-stage-a-validation-20261103.md](../../ep-stage-a-validation-20261103.md). Each folder is one
`scripts/ep/run-phone.sh` cell: `cmd.txt`, `stdout.log`, `stderr.log`, `exit_code`, `settings-*.txt` (clocks,
temperatures), `freq.csv` (100 ms clock sampler, realtime ns), `odpm-*.txt` (rail energy, `scripts/ep/odpm-diff.py`),
`trace-*.csv[.gz]` (`LLAMA_EP_TRACE`, `scripts/ep/trace-summary.py`). Large CSVs are gzipped; `simpleperf` data files
were left out (their reports are in `stdout.log`).

Pinned clocks where `settings-pinned.txt` exists: CPU 1328/1491/1426 MHz, GPU 701 MHz, MIF 2028 MHz.

## Current results (latest build)

| folder | what |
|---|---|
| `compare-matrix` | **Main comparison**: normal path vs all-cpu, all-gpu, 1+3, 2+2, 3+1 at -t 4, -t 1, and -t 4 with `LLAMA_EP_CPU_THREADS=1`; plus `check.txt` (numerical check of this build). `python3 scripts/ep/compare-table.py compare-matrix` |
| `cpu-gap-fused` | Fused CPU GEMV before the thread-pool fix: -t 1/2/4 sweep (4-thread rows show the two-pool problem), decode-path ablation, numerical check |

## Diagnosis of the slow CPU branch (earlier builds, kept as evidence)

| folder | what |
|---|---|
| `cpu-gap-stat` | simpleperf counters, normal path vs all-cpu before the fix: 129 K vs 17.4 M page faults |
| `cpu-gap-fix` | after persistent buffers: page faults 98 K, phase trace (prep/cleanup gone) |
| `cpu-gap-prof` | simpleperf flat profiles (normal: memcpy 45%; all-cpu: 74% in thread-pool loops) |
| `cpu-gap-threads` | -t 1/2/4 sweep and op profile (strided transpose+cast 19.4 ms/layer). The all-cpu rows at -t 1/2 ran with the backend thread count unset (4 planned on a 1-2 thread pool): do not quote them |

## Correctness and first timing (Stage A build, decode path on both branches)

| folder | what |
|---|---|
| `check-all` | 10 policies, every expert vs the double reference, logits (`ep-check.json`, traces) |
| `drift` | per-layer residual drift CPU vs GPU experts |
| `perf-pinned2` | per-policy timing with traces and op profiles |
| `serial-vs-conc-f0`, `-t3`, `-ff` | concurrent vs serialized branches; 3 compute threads; all 8 cores allowed |
| `smoke-nohost3` | Vulkan host buffer vs `--no-host` (7953 vs 490 ms/token) |

## Superseded or invalid

| folder | why |
|---|---|
| `vk-ops` | test-backend-ops segfaults on the phone (not investigated) |
| `check-cpu` | crashed at model load, before the Vulkan fix (`smoke` was rerun after the fix and is valid) |
| `smoke-nohost`, `smoke-nohost2` | `smoke-nohost2` ran an old binary (the build had failed); use `smoke-nohost3` |
| `perf-pinned` | trace variables were not exported (`VAR=x eval` in sh): its "opprof=1" rows were not profiled; timings valid |
