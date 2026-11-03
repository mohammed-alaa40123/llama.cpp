# Mobile MoE CPU–GPU Expert Parallelism: Coding Agent Implementation Plan

**Status:** Actionable research/engineering plan; not a claim that Android GPU execution or speedups have been validated.
**Repository:** `mohammed-alaa40123/llama.cpp`
**Starting point:** merged `master` at `f643507f9e68953767e465aecbf383b23a2075d1` (Maphist0/dev-ep).
**Working branch:** `research/ep-vulkan-foundation`. Do not commit directly to `master`.
**Priorities:** P0 functional and numerical correctness; P1 remove pathological overhead and enable Vulkan; P2 meaningful benchmarks; P3 memory-contention modeling and scheduling; P4 NPU/DVFS.
**Scope of first release:** CPU + one selected GPU backend (OpenCL *or* Vulkan), real resident MoE weights, top-k routed FFN computation, fixed clocks. NPU is not in the critical path.

## 0. Read the evidence and preserve the scientific question

Read these attachments supplied with the project (the local agent should receive them alongside this plan):
- `LLM_moe_mobile_experts_cpu_gpu(1).docx`: objectives, motivation, baseline comparisons, prefill/decode, energy/thermal questions, and staged CPU/GPU-first implementation.
- `related_prioer_work(1).docx`: novelty limits and strong baselines (HybriMoE, HeteroInfer, HaX-CoNN, CoDL, Apex, FusionML, ZipMoE, NPUMoE, CRISP).
- `mem-contention(1).md`: proposed synthetic memory latency surface `L(U_CPU,U_GPU)`, equivalent pressure `e`, latency sensitivity `m`, partial-overlap simulation, and caveats.
- `expert-parallelism-review-20261103(2).md`: current prototype trace and known bottlenecks.

**Research hypothesis:** when MoE routing exposes independent experts on unified-memory hardware, an appropriate CPU/GPU split can reduce end-to-end MoE-layer latency and/or energy versus **both** optimized CPU-only and GPU-only implementations. Contention-aware scheduling must beat static or solo-time-balanced splitting and justify its profiling/runtime cost. Do not count generic CPU/GPU concurrency or extra aggregate DRAM bandwidth as novelty by itself. PCCS/HaX-CoNN provide existing comparison points.

The first question is **whether efficient useful GEMVs overlap and improve performance**. Current trace proves some backend operation overlap but predominantly shows overlap of unnecessary weight transforms.

## 1. Verified code state and high-priority findings

Relevant locations in the merged tree:

| File | Existing role | Action |
|---|---|---|
| `src/llama-graph.cpp` | `build_moe_ffn` inserts `expert_parallel-<layer>` marker and reroutes to custom execution | Preserve ordering, weights and shape checks; later generalize architecture |
| `src/llama-context.cpp` | `llama_context::expert_eval` performs policy, staging, branch graphs, backend submission and join | Primary performance/correctness work |
| `src/llama-context.h` | Expert callback state | Add persistent resource state with explicit lifetime |
| `common/expert-policy.cpp` | all-cpu, all-gpu, alternating, expert-id, rotating policies | Add split enumeration and eventually predictive policy |
| `ggml/src/ggml-opencl/kernels/get_rows.cl` | Raw Q4_0 GET_ROWS path | Verify row-grid correctness and scale precision |
| `src/models/mobilemoe.cpp` | Special MobileMoE QAT weight format and tensor layout | Keep conversion/layout checks strict |
| `ggml/src/ggml-vulkan` | Existing generic Vulkan backend | Use this backend; do not duplicate GGML kernels prematurely |

**P0 defects/limitations confirmed by code and supplied trace:**
1. Weight decoder represented each entire expert matrix as a *single Q4 row*, leaving only one CPU decode partition/one OpenCL workgroup. Fixed in initial branch patch by preserving the original Q4-aligned matrix rows and gathering all row indices.
2. Each selected gate/up/down matrix is copied, fully dequantized to F32, materialized in another contiguous transposed F32 allocation, then multiplied by F32 GEMV. This must ultimately be replaced by layout-aware direct quantized GEMV; the initial patch is not the final solution.
3. CPU backend, backend buffers, graphs, staging and GPU thread are recreated per MoE-layer callback; the initial branch patch reuses the dedicated CPU backend only. Remaining resource churn is TODO.
4. Entire branch preparation and weight uploads are on the critical path before both branches run; asynchronous submission cannot hide this cost.
5. The original GPU worker had an unsafe C++ exception path: throw from worker could terminate the process, and throw on CPU before join could also terminate. Initial branch patch now captures exceptions, always signals/joins, and rejects graph-compute failure before reading outputs.
6. The policy path currently supports only special `MobileMoE` output-grouped Q4_0 gated SiLU FFN with no LoRA/merged gate-up. This is an intentional limitation, not generic MoE support.
7. GPU backend selection is based on GGML device type GPU/IGPU, rather than OpenCL-specific APIs. This is **a Vulkan-compatible dispatch foundation**, but no Vulkan phone execution has been verified.
8. Logs and timestamps need explicit mode separation: normal performance runs should not emit per-expert INFO logs or force GPU profiling synchronization. Initial branch reduces hot-path per-expert logs.
9. Quantization layout `qat_output_grouped_g32` is **not** interchangeable with normal GGML input-grouped Q4_0; raw transpose/requantization can silently change scale grouping and logits.
10. Mixed CPU/GPU output merging, duplicate routed expert IDs, batch>1, cancellation, backend failure, and multiple sessions/contexts require explicit tests.

**Current captured numbers (diagnostic, not speedups):** warm per-layer callback medians 15.687 ms concurrent rotating, 21.451 ms serialized rotating, 56.342 ms all-GPU using the *same inefficient preparation path*. CPU separate phase medians Q4 decode 3.402 ms, cont/transposition 3.775 ms, GEMV 0.202 ms; GPU 6.903, 0.264, 0.124 ms. Different expert counts and measurement clocks mean these cannot be summed or used as equal-work CPU/GPU comparisons. Only 18/160 warm callbacks established conservative simultaneous GEMV overlap. One profiled run per case, without fixed thermals, frequencies or confidence intervals.

## 2. Non-negotiable engineering rules for the agent

- Work in small, independently reviewable commits/PRs on a branch. Preserve the original upstream code path when `--expert-policy` is unset.
- Do not optimize around a known-bad Q4 layout conversion; maintain a bit-exact or numerically justified oracle for individual decoded weights, expert outputs, logits and greedy output.
- Include model SHA256, GGUF metadata, git SHA, Android build flags, backend device name and device SoC in every benchmark manifest.
- Separate host-wall, CPU operator-wall, actual CPU-on-core, GPU device event, queue latency, weight bytes uploaded, DRAM counters, and end-to-end token latency. Log formats must identify clock domains.
- Never claim no-copy/zero-copy just because CPU/GPU share physical DRAM. Validate allocations, mapping, coherence, copies and residency.
- Never claim a speedup without a warm matched baseline on the **same backend and kernel format**, controlled clocks/thermal state, multiple runs and reported uncertainty.
- Do not silently fall back to CPU when a GPU operation is unsupported. Fail with operation/type/shape/backend details during development; optionally add a separately labeled fallback mode later.
- Do not replace production quantization semantics with a different Q4 layout merely to obtain a faster kernel.
- Avoid changing custom `llama-model` architecture and generic GGML backend code in the same patch unless the dependency is unavoidable.

## 3. Stage A — correctness gate for the initial branch patch

**Changes already committed on `research/ep-vulkan-foundation`:**
- `src/llama-context.cpp`: replace one giant Q4 row with natural `[rows,cols]` packed tensor; gather all `cols` rows with I32 indices; preserve decoded transpose semantics.
- `src/llama-context.cpp`: catch backend-thread exceptions; signal submission on failure; join worker before rethrowing; check async graph status before output reads.
- `src/llama-context.h` and `src/llama-context.cpp`: persist a dedicated CPU expert backend across layer callbacks.
- `src/llama-context.cpp`: move hot-path per-expert upload/route/execute messages to DEBUG.
- `scripts/ep-vulkan-smoke.sh`: basic Vulkan build and operation/prototype smoke sequence.

**Agent must build and TEST these changes before further optimization:**

A1. Native CPU-only configure/build. Compile with `GGML_VULKAN=OFF`, then build with `GGML_VULKAN=ON` on a machine with Vulkan SDK/`glslc`; fix portability or compilation issues from the initial branch patch.

A2. For `MobileMoE-S-QAT` converted to **matching** GGUF layout, compare original `master` versus branch at fixed input and routing:
- Every `GET_ROWS` decoded FP32 value for gate/up/down on multiple experts/layers (ideally exact bitwise; investigate backend-specific rounding).
- Individual expert gate/up/down outputs and final weighted sum.
- Per-token logits (max absolute/RMS difference) and same deterministic greedy output.
- Duplicate selected expert IDs, all-CPU, all-GPU, 1+3, 2+2, 3+1, batch-1; `n_tokens>1` is currently unsupported as a validated workload and needs its own gate.
- Verify index tensor works on CPU, OpenCL and Vulkan and that decoding uses >1 CPU work partition or GPU workgroup.
- Test callback absent (normal llama.cpp execution) unaffected.
- Force graph-compute failure/unsupported backend and confirm informative error, no deadlock or terminate.

A3. Re-run exactly the diagnostic input from the attached trace, then collect a NEW per-phase breakdown. Do not infer performance from first-run initialization. Success criteria: compatibility, numerical stability, and verified row parallelism. Faster wall time is a hypothesis, not an acceptance assertion.

**Do not proceed to serious contention experiments until Stage A passes.**

## 4. Stage B — remove structural overhead and preserve direct expert weights

**B1. Persistent resources:** implement an explicit per-context `ExpertExecutionRuntime` with persistent CPU backend, one GPU backend handle, dedicated/persistent worker thread, reusable ggml graph contexts where shapes/policy allow, scratch/storage buffers with capacity bounds, and safe destruction on context release. Thread creation, graph allocation and buffer alloc/free should cease occurring for every layer callback. Confirm no use-after-free across graph reuse and back-to-back tokens.

**B2. Resident weights:** keep all selected model weights resident in DRAM for initial experiments. Disable explicit UFS streaming/prefetch and keep them out of the timed critical path. Separate model data residency from backend copies; measure both allocated memory and transfer bytes. Consider prepacked resident per-backend representations but do NOT duplicate the entire model blindly; track actual peak RSS, GPU memory and OOM.

**B3. Quantization-aware execution, end goal:** develop CPU and OpenCL/Vulkan-native GEMV that consumes the original output-grouped Q4_0 `g32` weights and FP16 scale directly. Fuse dequantization into multiply/accumulate without materializing whole F32 matrices, and avoid `ggml_cont(ggml_transpose(...))` per token. First verify exact group orientation and Q4 scale layout using Meta's original QAT checkpoint; write a **scalar reference**; then GPU shader and efficient CPU path. Keep this separate from placement policy. Native FP16/BF16 models can serve as an additional *kernel* control, but cannot be treated as numerically identical models to the QAT release.

**B4. Pipelining:** only after persistent weights exist, start whichever branch is ready; overlap GPU command preparation with CPU work when safe. Explicitly measure host dispatch, device queue wait, synchronization and useful GEMV overlap.

**Exit:** exact routed outputs within validated tolerance; no per-callback full F32 expert-matrix write/transposition in optimized mode; bounded memory; reproducible distinct CPU-only, GPU-only and mixed-policy performance.

## 5. Stage C — Vulkan backend implementation and qualification

**Rationale:** the current experimental code selects `GGML_BACKEND_DEVICE_TYPE_GPU/IGPU`, and already builds GGML `GET_ROWS`, `CONT`, `SWIGLU`, `MUL_MAT` operations through generic backend APIs. Vulkan is therefore potentially supported without a separate scheduler. Its exact type/shape support still must be verified on phone. The initial branch includes `scripts/ep-vulkan-smoke.sh`.

C1. Build native Linux Vulkan:
```bash
cmake -S . -B build-ep-vulkan -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_OPENCL=OFF
cmake --build build-ep-vulkan --target llama-cli test-backend-ops -j4
./build-ep-vulkan/bin/llama-cli --list-devices
./build-ep-vulkan/bin/test-backend-ops -b Vulkan0 -o GET_ROWS
```
Select the actual name printed by `--list-devices`, not a hardcoded Vulkan0 if it differs. Confirm Q4_0 GET_ROWS, F32 MUL_MAT, CONT and SWIGLU operations *with exact shapes and memory layout*. Document support/unsupported cases.

C2. Cross-compile on an Android NDK arm64 toolchain using `GGML_VULKAN=ON`, Vulkan headers/loader and host `glslc`; deploy to a phone that exposes Vulkan 1.x. Use clean, pinned backend versions. Cross-compiled binaries must be run on the device, not on the x86 host. Verify via `adb shell` and the backend device listing.

C3. Run the smoke script or equivalent device-side commands:
```bash
llama-cli --device Vulkan0 -ngl 0 \
  -m MobileMoE-S-QAT-compatible.gguf --expert-policy alternating \
  -t 4 -n 8 -p "The capital of France is" --temp 0
```
`--device` must refer to the **actual** Vulkan backend. `-ngl 0` keeps the rest of the model on CPU to isolate the EP path. Repeat `all-cpu`, `all-gpu`, `alternating`, `rotating` and verify actual Vulkan execution with profiler/device logs; compare logits and runtime to OpenCL and the CPU reference. If Vulkan supports only some expert ops/layouts, add narrowly scoped Vulkan kernels or a backend-specific direct Q4 GEMV and test them; avoid broad generic GGML changes without need.

C4. Performance baseline: compare *matched* CPU+Vulkan versus CPU+OpenCL on the **same phone**, model, weight precision, clocks, token count, CPU affinity and kernel algorithm. Neither is a presumed winner. Include GPU launch count, command submission overhead, weight copies and peak memory.

**Exit:** every selected GPU expert actually runs on Vulkan; all intermediate ops supported; numerical agreement with CPU/OpenCL reference; correct all-CPU behavior even with Vulkan registered; no CPU fallback masquerading as GPU; evidence of real overlap.

## 6. Stage D — model matrix and exact conversion protocol

**Primary real model family: Meta MobileMoE QAT, because current callback explicitly requires it.**
These are official source model specifications, **not** proven GGUF runtime compatibility.

| Source model | Parameters (total) | Layers | d_model | Routed FFN dimension | Routable experts | Top-k | Weight-only estimated INT4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| facebook/MobileMoE-S-QAT | 1.3B | 20 | 768 | 384 | 60 | 4 | 0.68 GB |
| facebook/MobileMoE-M-QAT | 2.8B | 26 | 1024 | 512 | 60 | 4 | 1.48 GB |
| facebook/MobileMoE-L-QAT | 5.3B | 32 | 1280 | 640 | 60 | 4 | 2.75 GB |

Links: https://huggingface.co/facebook/MobileMoE-S-QAT ; https://huggingface.co/facebook/MobileMoE-M-QAT ; https://huggingface.co/facebook/MobileMoE-L-QAT . FAIR noncommercial research license applies: verify allowed use and redistribution. **Weight-only estimates exclude KV cache, per-backend buffers, transposition scratch and OS consumption.**

D1. **Immediately** validate the already-tested S-QAT conversion: read source safetensors and supplied `modeling_mobilemoe.py`; verify QAT block orientation, scale dtype, Q4_0 packing, GGUF `expert_layout` and `qat_output_grouped_g32`; record converter commit. Compare decoded expert matrices to PyTorch reference and logits on a short input. Do not assume an arbitrary third-party Q4 GGUF has the required layout.

D2. **Scaling family**: after S works, M and L provide controlled increases in d_model and expert width with unchanged top-k=4. Use *the same* quantization and kernel path where possible. Confirm complete resident runtime footprint on the device before timing.

D3. **Numerical/architecture control**: test unquantized MobileMoE-S-SFT/Base or equivalent only **after** adding input-grouped/BF16 expert support; these are different checkpoints/quantization and must be compared as distinct experimental workloads, not logically equivalent values.

D4. **Different top-k generalization**: OLMoE-1B-7B (top-8) is a later extension. The current special MobileMoE callback throws outside its QAT SiLU layout. Generalize the executor to model-independent expert weight descriptors, activations, router normalization and optional shared experts before including OLMoE in performance comparisons. Do not silently claim OLMoE support from generic llama.cpp model loading.

D5. Optional prefill workload: multiple tokens per expert, including imbalanced router distributions; implement explicit tests for duplicate expert IDs and group-by-expert reuse. Keep decode M=1 results separate.

## 7. Stage E — first publishable measurement, before contention modeling

**Devices:** primary Snapdragon 8 Gen 3 phone (OnePlus 13R *if that is the exact owned device*, verify `ro.product.model`, SoC and Adreno device), because OpenCL and Vulkan comparisons are possible; Pixel 7/Pro Tensor G2 as CPU/Vulkan replication and existing instrumentation platform. Root/clock access varies by phone; record whether frequency controls are actually available. Do not assume MIF/GPU frequencies can be pinned on OnePlus without testing.

**Define baselines:**
- Original upstream no expert callback: CPU-only and best GPU-capable execution supported by model.
- New executor `all-cpu` and `all-gpu` with **same quantization and native kernel path**.
- Static `1+3`, `2+2`, `3+1` partitions for top-4, with both CPU/GPU assignment orientation and routing slots checked.
- Policy `alternating`, `rotating`, expert-ID, solo-time-balanced, and an offline per-layer best-split oracle. All policies need same routing and warm weights.
- Diagnostic serialized branch execution, explicitly labeled not a performance baseline.
- OpenCL and Vulkan backends tested separately.

**First plot:** MoE-layer latency versus number of experts on CPU (0–4), curves by expert shape S/M/L, token count M=1/4/16, backend (OpenCL/Vulkan), and fixed frequency settings. Show complete wall time and *actual useful computation time*. Do not confuse synchronized backend-operation overlap with concurrent GEMV.

**Instrumentation:** log router top-k, branch assignment, expert IDs, input bytes, packed weight bytes, staging copies, backend allocation, GPU upload, GPU enqueue, CPU on-core spans, device events, per-branch GEMV, final join and router-weighted reduction. Record DRAM bandwidth/latency/PERF counters where accessible and label unavailable counters. Avoid heavy tracing in timed throughput runs.

**Thermal/experiment protocol:** fixed available CPU/GPU/MIF clocks and affinity (or record uncontrolled frequency traces), cooling, randomized configuration order, warmup, >=3 repeated runs per cell before final selection, and 100 matched representative ShareGPT requests after microbenchmark validation. Report pooled token P99 as well as mean and median; report mJ/token, token/s, latency, page faults and peak RSS. Check that all weights remain in DRAM and that no UFS reads contaminate steady-state results.

**Go/no-go:** if a correct optimized mixed split never beats the best single-device baseline by more than variability, record the result and investigate dispatch, memory saturation and GPU grouping before building a dynamic scheduler. A decoder dominated by weight transformation is NOT suitable for contention validation.

## 8. Stage F — synthetic + expert contention characterization

Use the earlier `mem-contention` document as a **hypothesis**, not a proven law. It proposes `T/T0 = 1 + m * (L_now/L_alone - 1)`, effective traffic `e`, a CPU/GPU 2-D latency surface, and fixed-point adjustment `e*T0/T`. That evidence is for one chip and selected kernels; it does not prove universal expert-size or DVFS generalization.

F1. **Synthetic suite:** CPU/GPU sustained sequential reads, pointer-chasing latency probe, strided/random workloads, varied CPU thread count/GPU occupancy. Sample a coarse `L(Uc,Ug)` surface and refine near saturation. Measure achieved bandwidth and corresponding latency, not just requested bytes; characterize CPU and GPU separately.

F2. **Real kernels:** fixed-frequency CPU expert alone, GPU expert alone, matched concurrent pairs, and partial overlap. For every S/M/L shape and M=1/4/16, estimate effective pressure `e` and sensitivity `m` if they are identifiable. Account for cache, driver CPU interference and power limits. Avoid assuming "shared DRAM" is the *only* interaction: shared LLC, system cache, IOMMU, runtime staging and clock/thermal coupling can matter.

F3. **Independent baselines:** (i) no-contention solo-time balancing; (ii) empirical slowdown lookup/regression; (iii) adapted PCCS (already evaluated on Snapdragon in original literature); (iv) latency-surface `e,m` model; (v) oracle measured pairings. Do not call PCCS-on-any-mobile novel. Compare held-out **entire shapes/models**, not random repeated tokens alone. Record calibration time and coverage.

F4. **Partial overlap:** timestamped discrete event simulation: run solo when other engine idle, slowdown while simultaneous; check against actual CPU/GPU device timelines. Launch, join, output copies and other non-memory work must have their own costs.

F5. **Decision quality:** runtime prediction MAPE/RMSE and scheduling regret = (T_selected-T_oracle)/T_oracle, proportion of correctly chosen split, calibration effort, mean/P99 energy and token latency.

## 9. Stage G — online scheduling, then power/DVFS, then NPU

G1. After validated fixed-frequency models, implement policy `predictive` that evaluates all feasible `n_CPU+n_GPU=k` splits for homogeneous top-k and individual expert assignments for heterogeneous costs. Keep router weights and original top-k positions unchanged. For M>1, use counts of tokens routed to each expert and grouped backend kernel efficiency, not just `k`.

G2. Add decision hysteresis/caching for unchanged shapes and clocks; count scheduling overhead at microsecond resolution. Compare policy runtime against layer time and measure worst-case overhead.

G3. Joint frequency optimization only after verified CPU/GPU/MIF controls and frequency-sensitive model calibration; include switching costs and energy. Optimize latency under energy cap or energy under token-latency SLO; compare independent placement-then-DVFS versus joint optimization.

G4. NPU is optional extension. First run one real expert with a compatible Qualcomm Hexagon/QNN backend, including input/output conversion, startup, compilation, dispatch and memory costs; prove a useful region before adding CPU/GPU/NPU scheduling. No Pixel TPU support is required for the first paper.

## 10. Test matrix and acceptance gates

| Gate | Required checks | Evidence artifact |
|---|---|---|
| A: source correctness | Native build; Q4 row-oracle equality; all policy outputs; failure/no-deadlock; no-policy unchanged | `ep-a-tests.json`, logits diffs, CI log |
| C: Vulkan functional | Android Vulkan lists backend; supported operations; GPU dispatch verified; selected expert IDs and outputs correct; no hidden fallback | `ep-vulkan-ops.csv`, `ep-vulkan-trace.json` |
| B: efficient kernel | No per-token whole F32 matrix conversion in optimized mode; alloc/copy accounting; resident weight invariants | phase timeline + memory traces |
| E: feasibility | Best optimized CPU-only/GPU-only vs all top-k splits, repeated and controlled, full FFN latency | raw CSVs, plots, metadata manifest |
| F: model | Held-out shape/test predictor error and scheduler regret; PPCS vs hybrid and no-contention | profiling dataset/model checkpoints |
| G: end-to-end | 100 matched requests, token P99, mean, energy, thermals, cache/memory correctness | full reproduction package |

Also test kernel dimensions not divisible by block size (must reject safely), unsupported quantization, backend unavailable, request cancellation, empty GPU branch, empty CPU branch, model reload, concurrent contexts, ASan/TSan where available and Android platform edge cases.

## 11. Branch/PR breakdown for the coding agent

1. **PR-00 (this branch)**: validate row-parallel decoder, worker error propagation, reusable CPU backend and Vulkan smoke script. Fix compile/test failures. No performance claim.
2. **PR-01**: small, deterministic reference expert benchmark + layer/logit numerical oracle; CPU/OpenCL/Vulkan tests on real S-QAT format.
3. **PR-02**: persistent worker + graph/backend buffer reuse and safe resource teardown; record allocation/copy reduction.
4. **PR-03**: layout-aware quantized direct GEMV kernels with scalar oracle and CPU/OpenCL/Vulkan implementations. Keep preparatory and optimized paths switchable for ablation.
5. **PR-04**: Android Vulkan qualification, backend selection and comparative OpenCL/Vulkan performance report; upstream generic GPU path stays portable.
6. **PR-05**: reproducible split-sweep and end-to-end benchmark harness; S/M/L QAT model matrix.
7. **PR-06**: synthetic memory bench + characterized latency surface + PCCS and alternative model evaluation.
8. **PR-07**: contention-aware scheduler + held-out generalization/ablation and optional DVFS/NPU follow-ons.

**Immediate next agent action:** build the changed branch and produce a short structured report answering: (a) does row decomposition preserve values and use multiple workgroups? (b) do all CPU/OpenCL/Vulkan policy runs pass? (c) which backend ops/shapes fail? (d) what fraction of layer time is still spent on weight conversion, allocation, upload and synchronization? (e) what exact architecture-specific work is needed next? Do not merge until gate A and gate C pass.

## References / code evidence

- Merged baseline: https://github.com/mohammed-alaa40123/llama.cpp/commit/f643507f9e68953767e465aecbf383b23a2075d1
- This branch: https://github.com/mohammed-alaa40123/llama.cpp/tree/research/ep-vulkan-foundation
- Existing Vulkan operation coverage: `docs/ops.md` and `ggml/src/ggml-vulkan/`
- MobileMoE QAT source model cards (Meta): URLs in Stage D.
- Prior academic designs: HybriMoE https://arxiv.org/abs/2504.05897 ; HaX-CoNN https://arxiv.org/abs/2308.05869 ; NPUMoE https://arxiv.org/abs/2604.18788 ; PCCS original publication https://hpss.mines.edu/downloads/micro21.pdf
