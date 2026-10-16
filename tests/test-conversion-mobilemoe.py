from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gguf-py"))

import gguf
from conversion.mobilemoe import MobileMoEModel


@pytest.fixture
def model():
    m = MobileMoEModel.__new__(MobileMoEModel)
    m.hparams = {
        "num_hidden_layers": 2, "hidden_size": 8, "intermediate_size": 6,
        "intermediate_size_mlp": 16, "num_local_experts": 4,
        "num_experts_per_tok": 2, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 4,
        "routed_scaling_factor": 2.5, "norm_topk_prob": True,
    }
    m.block_count = 2
    m.tensor_map = gguf.get_tensor_name_map(m.model_arch, m.block_count)
    m.fuse_qkv = False
    m.fuse_gate_up_exps = False
    m.ftype = gguf.LlamaFileType.ALL_F32
    m.gguf_writer = gguf.GGUFWriter(None, "mobilemoe")
    m.rope_parameters = {}
    return m


def test_expert_layout_and_postgate(model):
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(3, 8, generator=gen)
    gate_up = torch.randn(4, 8, 12, generator=gen)
    down = torch.randn(4, 6, 8, generator=gen)
    gate_name, gate = list(model.modify_tensors(gate_up, "model.layers.0.feed_forward.experts.gate_up_proj", 0))[0]
    up_name, up = list(model.modify_tensors(gate_up, "model.layers.0.feed_forward.experts.gate_up_proj", 0))[1]
    down_name, down_out = next(iter(model.modify_tensors(down, "model.layers.0.feed_forward.experts.down_proj", 0)))
    assert (gate_name, up_name, down_name) == ("blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight")
    scores = torch.sigmoid(torch.randn(3, 4, generator=gen))
    bias = torch.tensor([0.0, 1.0, -1.0, 0.2])
    selected = (scores + bias).topk(2, dim=-1).indices
    weights = scores.gather(1, selected)
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * 2.5
    expected = torch.zeros_like(x)
    actual = torch.zeros_like(x)
    for t in range(3):
        for k in range(2):
            e = selected[t, k]
            g, u = (x[t] @ gate_up[e]).chunk(2)
            expected[t] += ((F.silu(g) * u) @ down[e]) * weights[t, k]
            actual[t] += F.linear(F.silu(F.linear(x[t], gate[e])) * F.linear(x[t], up[e]), down_out[e]) * weights[t, k]
    torch.testing.assert_close(actual, expected)


def test_selection_bias_and_shared_names(model):
    names = {
        "expert_bias": "blk.1.exp_probs_b.bias",
        "router.weight": "blk.1.ffn_gate_inp.weight",
        "shared_expert.gate_proj.weight": "blk.1.ffn_gate_shexp.weight",
        "shared_expert.up_proj.weight": "blk.1.ffn_up_shexp.weight",
        "shared_expert.down_proj.weight": "blk.1.ffn_down_shexp.weight",
    }
    for suffix, expected in names.items():
        actual, _ = next(iter(model.modify_tensors(torch.zeros(4), "model.layers.1.feed_forward." + suffix, 1)))
        assert actual == expected


@pytest.mark.parametrize("high", [1.0, 4.0])
def test_rope_cutoffs(model, high):
    model.hparams["head_dim"] = 64
    model.rope_parameters = {
        "rope_type": "llama3", "rope_theta": 500000.0, "factor": 16.0,
        "low_freq_factor": 1.0, "high_freq_factor": high,
        "original_max_position_embeddings": 8192,
    }
    name, factors = next(iter(model.generate_extra_tensors()))
    assert name == "rope_freqs.weight"
    assert torch.isfinite(factors).all()
    assert factors[0] == 1.0 and factors[-1] == 16.0
    assert torch.all(factors[1:] >= factors[:-1])


@pytest.mark.parametrize("key,value", [("interleave_moe_layer_step", 2), ("use_qk_norm", False), ("attention_bias", True), ("attn_temperature_tuning", True), ("no_rope_layers", [1, 0]), ("layer_types", ["sliding_attention"]), ("quantization_config", {"bits": 4})])
def test_unsupported_config_rejected(model, key, value):
    model.hparams[key] = value
    with pytest.raises(ValueError):
        model.set_gguf_parameters()


def test_metadata(model):
    model.set_gguf_parameters()
    fields = model.gguf_writer.kv_data[0]
    assert fields["mobilemoe.expert_weights_scale"].value == 2.5
    assert fields["mobilemoe.expert_weights_norm"].value is True
    assert fields["mobilemoe.expert_feed_forward_length"].value == 6
    assert fields["mobilemoe.expert_shared_feed_forward_length"].value == 16


def test_fused_experts(model):
    model.fuse_gate_up_exps = True
    model._gate_exp_buffer = {}
    model._up_exp_buffer = {}
    source = torch.arange(4 * 8 * 12, dtype=torch.float32).reshape(4, 8, 12)
    name, fused = next(iter(model.modify_tensors(source, "model.layers.0.feed_forward.experts.gate_up_proj", 0)))
    assert name == "blk.0.ffn_gate_up_exps.weight"
    torch.testing.assert_close(fused, source.transpose(1, 2).contiguous())


@pytest.mark.parametrize("suffix", ["gate_up_proj", "down_proj"])
def test_wrong_expert_shape(model, suffix):
    with pytest.raises(ValueError, match="Unexpected MobileMoE expert shape"):
        list(model.modify_tensors(torch.zeros(4, 8, 6), "model.layers.0.feed_forward.experts." + suffix, 0))
