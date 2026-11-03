#include "models.h"

void llama_model_mobilemoe::load_arch_hparams(llama_model_loader & ml) {
    llama_model_exaone_moe::load_arch_hparams(ml);
    type = LLM_TYPE_UNKNOWN;

    std::string layout;
    ml.get_key(LLM_KV_EXPERT_LAYOUT, layout);
    if (layout == "qat_output_grouped_g32") {
        hparams.moe_qat_output_grouped = true;
    } else if (layout != "input_grouped") {
        throw std::runtime_error("unsupported MobileMoE expert layout: " + layout);
    }
}

void llama_model_mobilemoe::load_arch_tensors(llama_model_loader & ml) {
    if (!hparams.moe_qat_output_grouped) {
        llama_model_exaone_moe::load_arch_tensors(ml);
        return;
    }
    if (hparams.n_layer_dense_lead != 0 || hparams.n_layer_nextn != 0) {
        throw std::runtime_error("MobileMoE QAT layout requires routed experts in every layer and no NextN layers");
    }

    LLAMA_LOAD_LOCALS;
    const int64_t n_ff_exp   = hparams.n_ff_exp();
    const int64_t n_ff_shexp = hparams.n_ff_shexp > 0 ? hparams.n_ff_shexp : n_ff_exp;
    const int64_t n_qo_dim   = n_head * hparams.n_embd_head_k();
    const int64_t n_kv_dim   = n_head_kv * hparams.n_embd_head_k();
    if (n_expert == 0 || n_expert_used == 0) {
        throw std::runtime_error("MobileMoE requires n_expert and n_expert_used > 0");
    }

    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];
        create_tensor_qkv(layer, i, n_embd, n_qo_dim, n_kv_dim, n_kv_dim, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_qo_dim, n_embd}, 0);
        layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.ffn_norm    = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_ff_exp, n_embd, n_expert}, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "weight", i), {n_ff_exp, n_embd, n_expert}, 0);
        if (layer.ffn_gate_exps->type != GGML_TYPE_Q4_0 ||
            layer.ffn_down_exps->type != GGML_TYPE_Q4_0 ||
            layer.ffn_up_exps->type   != GGML_TYPE_Q4_0) {
            throw std::runtime_error("MobileMoE qat_output_grouped_g32 requires Q4_0 experts");
        }
        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_shexp}, 0);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp, n_embd}, 0);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP, "weight", i), {n_embd, n_ff_shexp}, 0);
    }
}
