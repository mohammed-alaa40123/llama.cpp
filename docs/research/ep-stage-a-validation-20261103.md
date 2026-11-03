# Expert parallelism, Stage A: correctness on CPU, Vulkan and mixed splits (3 Nov 2026)

Branch `research/ep-vulkan-foundation`. Device: Pixel 7 Pro (Tensor G2, Mali-G710, Android 13, root). Model:
`MobileMoE-M-QAT-q4_0.gguf` (26 layers, d_model 1024, expert dim 512, 60 experts, top-4,
`expert_layout=qat_output_grouped_g32`, SHA256 `242ab04c...0c78`). MobileMoE-S/L QAT GGUFs are not available
(the Hugging Face repos are gated; no converter is in this branch), so S-QAT was not tested.

## Summary

- **Correct.** Every routed expert output, on CPU and on the Mali GPU (Vulkan), matches an independent
  double-precision reference decoded from the GGUF bytes: worst relative L2 error 4.2e-7 over 12,280 expert outputs
  and 10 policies (all-cpu, all-gpu, alternating, expert-id, rotating, 1+3, 2+2, 3+1 in both slot orders).
  `all-cpu` is bit-identical to the normal llama.cpp path (logits, every layer). Policies that use the GPU give the
  same greedy tokens on all 7 checked positions; their logits differ by at most 0.78 (RMS 0.07).
- **Four bugs fixed** (below): the executor skipped the model's BF16 weight rounding; a failed expert assignment
  aborted the process; ggml-Vulkan crashed at model load on this phone; with a GPU present the CPU's buffers were
  placed in GPU host memory, making the CPU path 16x slower.
- **OpenCL cannot run on this phone**: ggml's OpenCL backend only accepts Adreno and Intel GPUs. Vulkan is the GPU
  backend here.
- **No mixed split beats all-gpu yet.** Decode, pinned clocks: normal path 490 ms/token; executor all-gpu 632,
  1+3 752, 2+2 883, 3+1 1146, all-cpu 1439. Weight decoding/rounding still dominates (Stage B work).
- **Running the branches concurrently slows the CPU branch** (1+3: 6.7 ms alone, 15.7 ms concurrent per layer)
  while the GPU branch is unchanged. The cause is not isolated yet (see Open questions).

## Fixes

| Change | Files | Evidence |
|---|---|---|
| Executor rounds decoded expert weights to BF16 (transpose+cast to BF16, then F32), as the normal path does and as the GGUF declares (`weight_materialization = ... BF16 expert materialization`) | `src/llama-context.cpp` | Before: executor matched the F32-weight reference (2.4e-7) and missed the BF16 one by 3.5e-3; after: bit-identical to the normal path |
| Exceptions from the executor (e.g. GPU requested but none present) now fail `llama_decode` with an error instead of `std::terminate` | `src/llama-context.cpp` (`graph_compute`) | `--expert-policy all-gpu` on a CPU-only build: exit 1 with `graph_compute: invalid expert assignment or unavailable GPU` (was exit 134) |
| ggml-Vulkan uses buffer device address only on Vulkan 1.2+ devices | `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | Mali-G710 reports device API 1.1 but `bufferDeviceAddress=1`; the core `vkGetBufferDeviceAddress` call jumped to address 0 at model load (backtrace symbolized). Every Vulkan run crashed before the fix |
| `--no-host` also keeps the CPU compute buffer out of the GPU host buffer type | `src/llama-context.cpp`, `src/llama-model.{h,cpp}` | Normal path decode: 7953 ms/token with Vulkan host buffers, 490 with `--no-host` (16x). Consistent with that memory being uncached for the CPU on Mali (not checked directly) |

## Added

| What | Files |
|---|---|
| Split policies `cpu-first-N`, `cpu-last-N` (first/last N top-k slots of each token on CPU); `--expert-policy` validates names through the parser | `common/expert-policy.cpp`, `common/arg.cpp` |
| Executor trace `LLAMA_EP_TRACE=file.csv`: per layer callback, CLOCK_MONOTONIC us for policy, CPU-branch prep, GPU-branch prep, GPU submit/sync/read, CPU compute/read, join, output, cleanup; bytes uploaded per branch; per-slot expert id and backend. Header carries the realtime offset | `src/llama-context.{h,cpp}` |
| `LLAMA_EP_SERIAL=1`: GPU branch then CPU branch (control for concurrency). `LLAMA_EP_OPPROF=1`: also one op per compute call, timed (`file.csv.ops.csv`; serialized, includes per-op launch/sync) | same |
| The executor passes its output tensor to `cb_eval` hooks, so a checker can see every expert output | same |
| `llama-ep-check`: per-policy run on the same tokens (prompt as one batch, then teacher-forced decode); checks each expert output against a double reference with F32 and BF16 weights (independent Q4_0/fp16 decoder, reads GGUF bytes); logits, final hidden state and per-layer residual drift vs the first policy; JSON report and one trace per policy | `examples/ep-check/` |
| Phone tools: `run-phone.sh` (push, run, pull), `phone-cell.sh` (optional CPU/GPU/MIF pinning with restore, 100 ms clock/temperature sampler, ODPM snapshots), `trace-summary.py` (phase medians, branch overlap, per-op medians), `odpm-diff.py` (per-rail energy), `truncate-gguf.py` (N-layer test model) | `scripts/ep/` |

## Correctness results (phone, Vulkan, `--no-host`, 4 threads on cores 4-7)

`llama-ep-check --policies none,all-cpu,all-gpu,alternating,expert-id,rotating,cpu-first-1,cpu-first-2,cpu-first-3,cpu-last-1 --steps 6`
(6 prompt tokens as one batch, then 6 decode steps; 1,228 expert outputs per policy).

| policy | CPU slots | GPU slots | worst rel. err vs BF16 ref (CPU / GPU) | logits max / RMS vs normal path | greedy match |
|---|---|---|---|---|---|
| none (normal path) | - | - | 3.1e-7 | 0 / 0 | 7/7 |
| all-cpu | 1228 | 0 | 3.1e-7 / - | 0 / 0 | 7/7 |
| all-gpu | 0 | 1228 | - / 3.1e-7 | 0.78 / 0.068 | 7/7 |
| alternating | 614 | 614 | 4.1e-7 / 2.4e-7 | 0.53 / 0.070 | 7/7 |
| expert-id | 649 | 579 | 3.1e-7 / 2.4e-7 | 0.50 / 0.063 | 7/7 |
| rotating | 815 | 413 | 3.0e-7 / 2.6e-7 | 0.45 / 0.066 | 7/7 |
| cpu-first-1 (1+3) | 307 | 921 | 3.7e-7 / 2.8e-7 | 0.49 / 0.060 | 7/7 |
| cpu-first-2 (2+2) | 614 | 614 | 4.2e-7 / 3.1e-7 | 0.49 / 0.062 | 7/7 |
| cpu-first-3 (3+1) | 921 | 307 | 3.6e-7 / 2.4e-7 | 0.33 / 0.046 | 7/7 |
| cpu-last-1 | 307 | 921 | 2.9e-7 / 2.8e-7 | 0.33 / 0.042 | 7/7 |

Why logits differ although each expert is accurate: the residual stream difference to the CPU path grows through
the layers (relative RMS, all-gpu: 2.9e-7 at layer 0, 1.3e-4 at 3, 1.4e-3 at 12, 1.7e-2 at 24, 2.2e-2 at 25;
2+2 jumps from 2e-8 at layer 3 to 5e-4 at layer 6). Individual expert errors do not grow. The step-like growth
is consistent with small differences flipping roundings in the 8-bit activation quantization of later Q4_0
matmuls, but that cause was not isolated. A perplexity comparison would settle whether the drift matters; the
executor is too slow for that today.

Host checks (same tool): 4-layer truncated model on x86 CPU and on lavapipe (software Vulkan), same results
(worst 2.7e-7). Index-tensor GET_ROWS runs on CPU and Vulkan. Callback absent: normal path unchanged.

## Timing (decode, warm, pinned clocks: CPU 1328/1491/1426 MHz, GPU 701 MHz, MIF 2028 MHz)

One run per cell, 15 decode tokens; no repetitions or confidence intervals yet. Per layer callback medians from the
trace; "alone" = `LLAMA_EP_SERIAL=1`.

| policy | ms/token | layer total | CPU branch alone / concurrent | GPU branch alone / concurrent |
|---|---|---|---|---|
| normal path | 490 | - | - | - |
| all-gpu | 632 | 16.9 | - | 11.5 |
| 1+3 | 747 / 759 serial | 23.3 | 6.7 / 15.7 | 8.7 / 8.8 |
| 2+2 | 949 / 951 serial | 30.5 | 14.9 / 22.0 | 6.1 / 6.1 |
| 3+1 | 1250 / 1486 serial | 37.5 | 22.8 / 24.9 | 5.7 / 3.3 |
| all-cpu | 1439 | 40.8 | 24.9 | - |

Per-op time per layer (op-by-op mode, all-cpu / all-gpu, ms): GET_ROWS decode 13.5 / 5.3, transpose+cast to
BF16 22.5 / 7.8, cast to F32 9.3 / 4.0, GEMV 5.5 / 3.7, SWIGLU 1.2 / 0.9. The BF16 fix added the F32 cast pass
(about 9 ms CPU, 4 ms GPU per layer); a fused Q4 GEMV would apply the rounding in registers. Host prep per layer:
4.0 ms (all-cpu), 2.7 ms (all-gpu); CPU-branch cleanup (buffer frees) 6.3 ms in all-cpu.

Affinity matters: allowing all 8 cores (`CPUS=ff`) made 2+2 1521 ms/token (cores 4-7: 949), with the CPU
branch at 39 ms per layer.

## Open questions

1. **Why the CPU branch slows down while the GPU branch runs.** ggml-Vulkan spin-waits on the fence
   (`ggml_vk_wait_for_fence`), so the GPU worker thread occupies a core. Leaving one core free (3 compute threads)
   did not remove the slowdown (1+3: 8.2 -> 14.9 ms; 2+2: 15.7 -> 22.3 ms), so core sharing is not the main
   cause. Next tests: pin each compute thread and the worker to separate cores; a blocking fence wait; the
   synthetic CPU/GPU memory streams of the contention plan to measure the latency the CPU sees.
2. `test-backend-ops` segfaults on the phone right after device enumeration (also after the Vulkan fix); not
   investigated. The checker covers the executor's ops at their real shapes.
3. The NDK glslc (2022) lacks cooperative-matrix, integer-dot and BF16 shader extensions; Mali-G710 reports
   `int dot: 0` anyway.
4. Only M-QAT was tested; S and L need a converted GGUF with the same layout.

## Next milestone

Stage B: persistent GPU worker and buffers (cleanup alone is 6 ms per layer on CPU), then a fused
output-grouped Q4 GEMV with in-register BF16 rounding on CPU and Vulkan, checked with `llama-ep-check`. Then
re-run the split sweep with repetitions, and work out question 1 before any contention modelling.

## Commands

```bash
# host builds (CPU, and Vulkan with lavapipe: GGML_VK_VISIBLE_DEVICES=0)
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=OFF
cmake -S . -B build-vk -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON \
  -DVulkan_INCLUDE_DIR=$HOME/vulkan-headers/include -DVulkan_LIBRARY=/usr/lib/x86_64-linux-gnu/libvulkan.so.1 \
  -DVulkan_GLSLC_EXECUTABLE=$NDK/shader-tools/linux-x86_64/glslc -DCMAKE_PREFIX_PATH=$HOME/spirv-headers
# Vulkan-Headers v1.4.330 at ~/vulkan-headers; SPIRV-Headers vulkan-sdk-1.4.328.0 installed to ~/spirv-headers and
# linked as ~/vulkan-headers/include/spirv (ggml includes <spirv/unified1/spirv.hpp>)

# Android
cmake -S . -B build-android-vk -G Ninja -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-33 -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DGGML_OPENMP=OFF -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF \
  -DGGML_VULKAN=ON -DVulkan_INCLUDE_DIR=$HOME/vulkan-headers/include \
  -DVulkan_LIBRARY=$NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/33/libvulkan.so \
  -DVulkan_GLSLC_EXECUTABLE=$NDK/shader-tools/linux-x86_64/glslc -DCMAKE_PREFIX_PATH=$HOME/spirv-headers \
  -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH
cmake --build build-android-vk --target llama-ep-check llama-completion llama-cli test-backend-ops
# NDK 27.3.13750724; strip the binaries before pushing (170 MB -> 28 MB)

# phone: correctness
NAME=check-all CMD='$BIN/llama-ep-check -m $MODEL -t 4 --no-repack --no-host -ngl 0 --device Vulkan0 \
  --policies none,all-cpu,all-gpu,alternating,expert-id,rotating,cpu-first-1,cpu-first-2,cpu-first-3,cpu-last-1 \
  --steps 6 --out $OUT' bash scripts/ep/run-phone.sh
# phone: timing with trace (export the variables; VAR=x eval ... does not export them in sh)
NAME=serial-vs-conc PIN=1328000/1491000/1426000/701000/2028000 CMD='... export LLAMA_EP_SERIAL=$s \
  LLAMA_EP_TRACE=$OUT/trace-$p-s$s.csv; $BIN/llama-completion ... --expert-policy $p' bash scripts/ep/run-phone.sh
python3 scripts/ep/trace-summary.py RESULTS/NAME/trace-*.csv
python3 scripts/ep/odpm-diff.py RESULTS/NAME
```

Raw results: `/home/mohamed/mobilemoe/ep-results/{check-all,drift,perf-pinned2,serial-vs-conc-f0,serial-vs-conc-ff,serial-vs-conc-t3,smoke-nohost3}`.
