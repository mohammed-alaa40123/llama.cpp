from __future__ import annotations

import math
from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("MobileMoEForCausalLM")
@ModelBase.example("facebook/MobileMoE-L-Base")
class MobileMoEModel(TextModel):
    model_arch = gguf.MODEL_ARCH.MOBILEMOE

    def set_gguf_parameters(self):
        hp = self.hparams
        # These are the released Base models. Reject other attention/FFN layouts.
        if hp.get("interleave_moe_layer_step", 1) != 1:
            raise ValueError("MobileMoE dense interleaving is not supported")
        if not hp.get("use_qk_norm", True) or not all(hp.get("no_rope_layers", [1] * self.block_count)):
            raise ValueError("MobileMoE requires RoPE and QK normalization in every layer")
        if hp.get("attn_temperature_tuning", False) or hp.get("attention_bias", False):
            raise ValueError("MobileMoE attention temperature tuning and bias are not supported")
        if any(t != "full_attention" for t in hp.get("layer_types", [])):
            raise ValueError("Only MobileMoE full attention is supported")
        if hp.get("hidden_act", "silu") != "silu":
            raise ValueError("Only MobileMoE SwiGLU is supported")
        if hp.get("quantization_config"):
            raise ValueError("Packed MobileMoE QAT checkpoints are not supported; use Base BF16 weights")
        if hp["routed_scaling_factor"] <= 0:
            raise ValueError("MobileMoE routed_scaling_factor must be positive")

        super().set_gguf_parameters()
        self.gguf_writer.add_expert_feed_forward_length(hp["intermediate_size"])
        self.gguf_writer.add_expert_shared_feed_forward_length(hp["intermediate_size_mlp"])
        self.gguf_writer.add_expert_weights_scale(hp["routed_scaling_factor"])
        self.gguf_writer.add_expert_weights_norm(hp["norm_topk_prob"])
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        rp = self.rope_parameters.get("full_attention", self.rope_parameters)
        if rp.get("rope_type", "").lower() != "llama3":
            return
        low = rp.get("low_freq_factor", 1.0)
        high = rp.get("high_freq_factor", 4.0)
        dim = self.hparams["head_dim"]
        base = rp["rope_theta"]
        freqs = base ** (-torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        context = rp["original_max_position_embeddings"]
        wavelength = 2 * math.pi / freqs
        factors = torch.where(wavelength > context / low, rp["factor"], 1.0)
        # Equal frequency cutoffs have no interpolation region.
        if high != low:
            smooth = (context / wavelength - low) / (high - low)
            interpolated = 1.0 / ((1.0 - smooth) / rp["factor"] + smooth)
            middle = (wavelength >= context / high) & (wavelength <= context / low)
            factors = torch.where(middle, interpolated, factors)
        yield self.format_tensor_name(gguf.MODEL_TENSOR.ROPE_FREQS), factors

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.endswith(".experts.gate_up_proj"):
            assert bid is not None
            expected = (self.hparams["num_local_experts"], self.hparams["hidden_size"], 2 * self.hparams["intermediate_size"])
            if tuple(data_torch.shape) != expected:
                raise ValueError(f"Unexpected MobileMoE expert shape: {name}: {tuple(data_torch.shape)}, expected {expected}")
            gate, up = data_torch.chunk(2, dim=-1)
            yield from super().modify_tensors(gate.transpose(1, 2).contiguous(), self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE_EXP, bid), bid)
            yield from super().modify_tensors(up.transpose(1, 2).contiguous(), self.format_tensor_name(gguf.MODEL_TENSOR.FFN_UP_EXP, bid), bid)
            return
        if name.endswith(".experts.down_proj"):
            assert bid is not None
            expected = (self.hparams["num_local_experts"], self.hparams["intermediate_size"], self.hparams["hidden_size"])
            if tuple(data_torch.shape) != expected:
                raise ValueError(f"Unexpected MobileMoE expert shape: {name}: {tuple(data_torch.shape)}, expected {expected}")
            yield self.map_tensor_name(name + ".weight"), data_torch.transpose(1, 2).contiguous()
            return
        if name.endswith(".expert_bias"):
            name += ".bias"
        yield from super().modify_tensors(data_torch, name, bid)
