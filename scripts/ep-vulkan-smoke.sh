#!/usr/bin/env bash
# Smoke-test the experimental CPU/GPU expert executor with a Vulkan backend.
# This is a correctness/compatibility probe, NOT a performance benchmark.
set -euo pipefail

: "${EP_MODEL:?Set EP_MODEL to a converted MobileMoE QAT GGUF with expert_layout=qat_output_grouped_g32}"
EP_BUILD_DIR="${EP_BUILD_DIR:-build-ep-vulkan}"
EP_DEVICE="${EP_DEVICE:-Vulkan0}"
EP_THREADS="${EP_THREADS:-4}"
EP_PROMPT="${EP_PROMPT:-The capital of France is}"
EP_TOKENS="${EP_TOKENS:-8}"

if [[ "${EP_SKIP_BUILD:-0}" != 1 ]]; then
    cmake -S . -B "${EP_BUILD_DIR}" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_VULKAN=ON \
        -DGGML_OPENCL=OFF
    cmake --build "${EP_BUILD_DIR}" --target llama-cli test-backend-ops -j "${EP_BUILD_JOBS:-4}"
fi

BIN="${EP_BUILD_DIR}/bin/llama-cli"
TEST_BIN="${EP_BUILD_DIR}/bin/test-backend-ops"
if [[ ! -x "${BIN}" || ! -x "${TEST_BIN}" ]]; then
    echo "Missing binaries in ${EP_BUILD_DIR}/bin (for Android, cross-compile on host and push this build)." >&2
    exit 1
fi

echo "== Registered devices (confirm ${EP_DEVICE} appears) =="
"${BIN}" --list-devices

echo "== Backend operation support tests; check for failures =="
"${TEST_BIN}" -b "${EP_DEVICE}" -o GET_ROWS
"${TEST_BIN}" -b "${EP_DEVICE}" -o MUL_MAT
"${TEST_BIN}" -b "${EP_DEVICE}" -o CONT

mkdir -p "${EP_LOG_DIR:-ep-smoke-results}"
out="${EP_LOG_DIR:-ep-smoke-results}"
for policy in all-cpu all-gpu alternating rotating; do
    echo "== EP policy: ${policy} on ${EP_DEVICE} =="
    # --no-host: on Mali the GPU host buffer made the CPU path ~16x slower
    "${BIN}" --device "${EP_DEVICE}" -ngl 0 --no-host \
        -m "${EP_MODEL}" --expert-policy "${policy}" \
        -t "${EP_THREADS}" -n "${EP_TOKENS}" -p "${EP_PROMPT}" \
        --temp 0 > "${out}/${policy}.log" 2>&1 || {
            echo "FAILED: ${policy}; review ${out}/${policy}.log" >&2
            exit 1
        }
done

echo "Completed CPU/Vulkan expert smoke cases. Inspect logs and compare logits"
echo "against the unmodified reference model; this script does not certify"
echo "numerical equivalence, real GPU execution, speedups, or contention."
