# MobileMoE

This implementation adds CPU inference and GGUF conversion for Meta's text-only MobileMoE S/M/L Base architecture. It uses grouped-query attention, consecutive-pair RoPE, parameter-free per-head RMS normalization after RoPE, sigmoid top-k routing, and an always-active shared SwiGLU expert.

## Conversion

Download an authorized Base checkpoint using the Hugging Face CLI, then run:

```sh
python convert_hf_to_gguf.py /path/to/MobileMoE-L-Base --outtype bf16 --outfile MobileMoE-L-Base-BF16.gguf
build/bin/llama-quantize MobileMoE-L-Base-BF16.gguf MobileMoE-L-Base-Q4_K_M.gguf Q4_K_M
build/bin/llama-completion -m MobileMoE-L-Base-Q4_K_M.gguf -p "The capital of France is" -n 32 -ngl 0
```

The converter supports separate and fused QKV tensors and `--fuse-gate-up-exps`. Router weights remain F32. Expert selection uses a post-sigmoid bias; selected routing weights use the unbiased scores, optional sum normalization with an additive `1e-20` epsilon, and the checkpoint's routing scale (2.5 in the released Base models). The shared expert has no learned output gate.

The released RoPE configuration has equal high- and low-frequency cutoffs. Conversion generates the frequency-factor tensor without dividing by their difference. Q/K projection weights are already in consecutive-pair order and must not receive the usual Hugging Face Llama permutation.

Packed QAT checkpoints, dense-interleaved variants, attention bias, temperature tuning, and layers without RoPE/QK normalization are rejected. They need separate validation and support.

## Validation

```sh
python -m pytest tests/test-conversion-mobilemoe.py
build/bin/test-llama-archs -a '^mobilemoe$' -b CPU
```

The architecture test covers model save/reload and mixed batches. Check reference logits using identical token IDs, precision, and positions before drawing quality conclusions. Quantization changes logits; loading and text generation alone are smoke tests, not a quality evaluation. Quantized KV caches and platform-specific backends need their own checks.

## Model license

The original weights and derived GGUF files retain Meta's FAIR Noncommercial Research License, including its Acceptable Use Policy. Keep the original `LICENSE` with any checkpoint distribution and identify the exact source checkpoint/revision. The independently implemented llama.cpp changes use this repository's MIT license.

Source: <https://huggingface.co/facebook/MobileMoE-L-Base>.
