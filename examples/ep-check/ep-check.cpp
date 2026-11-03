// Numerical check of the CPU/GPU expert executor (--expert-policy).
//
// For each policy, run the same tokens (prompt as one batch, then teacher-forced decode steps) and:
//  - check every routed expert output against a double precision reference decoded from the GGUF bytes,
//    with F32 weights ((q-8)*d) and with BF16-rounded weights (the model's declared materialization)
//  - compare logits with the first policy (default "none" = normal llama.cpp path)
//  - compare layer 0 expert outputs between policies (identical inputs)
//
//   llama-ep-check -m model.gguf [common args] --policies none,all-cpu,all-gpu,cpu-first-2 [--out DIR] [--steps N]

#include "arg.h"
#include "common.h"
#include "expert-policy.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cinttypes>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// ---- independent Q4_0 decoder (does not use ggml) ----

static float fp16_to_f32(uint16_t h) {
    const uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1f, man = h & 0x3ff;
    float v;
    if (exp == 0) {
        v = std::ldexp((float) man, -24);
    } else if (exp == 31) {
        v = man ? NAN : INFINITY;
    } else {
        v = std::ldexp((float) (man | 0x400), (int) exp - 25);
    }
    return sign ? -v : v;
}

static float bf16_round(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    if ((u & 0x7fffffff) > 0x7f800000) {
        u = (u >> 16 | 64) << 16; // quiet NaN
    } else {
        u = ((u + (0x7fff + ((u >> 16) & 1))) >> 16) << 16; // round to nearest even
    }
    memcpy(&f, &u, 4);
    return f;
}

struct gguf_weights {
    FILE * f = nullptr;
    gguf_context * gctx = nullptr;
    ggml_context * meta = nullptr;
    size_t data_offset = 0;

    bool open(const std::string & path) {
        gguf_init_params ip = { true, &meta };
        gctx = gguf_init_from_file(path.c_str(), ip);
        f = fopen(path.c_str(), "rb");
        if (gctx) {
            data_offset = gguf_get_data_offset(gctx);
        }
        return gctx && f;
    }
    ~gguf_weights() {
        if (f) fclose(f);
        if (gctx) gguf_free(gctx);
        if (meta) ggml_free(meta);
    }

    // W[o + n_out*r] for expert e of tensor name; ne = {n_out, n_in, n_expert}, Q4_0 blocks along n_out
    bool expert(const std::string & name, int e, std::vector<float> & w, int64_t & n_out, int64_t & n_in) {
        const int64_t id = gguf_find_tensor(gctx, name.c_str());
        if (id < 0) return false;
        const ggml_tensor * t = ggml_get_tensor(meta, name.c_str());
        if (!t || t->type != GGML_TYPE_Q4_0 || t->ne[0] % 32) return false;
        n_out = t->ne[0];
        n_in  = t->ne[1];
        const size_t row_bytes = (size_t) (n_out / 32) * 18;
        std::vector<uint8_t> raw(row_bytes * n_in);
        const size_t off = data_offset + gguf_get_tensor_offset(gctx, id) + (size_t) e * row_bytes * n_in;
        if (fseek(f, (long) off, SEEK_SET) != 0 || fread(raw.data(), 1, raw.size(), f) != raw.size()) return false;
        w.resize(n_out * n_in);
        for (int64_t r = 0; r < n_in; ++r) {
            for (int64_t b = 0; b < n_out / 32; ++b) {
                const uint8_t * blk = raw.data() + r * row_bytes + b * 18;
                uint16_t dh;
                memcpy(&dh, blk, 2);
                const float d = fp16_to_f32(dh);
                for (int j = 0; j < 16; ++j) {
                    w[r * n_out + b * 32 + j]      = (float) ((blk[2 + j] & 0xf) - 8) * d;
                    w[r * n_out + b * 32 + j + 16] = (float) ((blk[2 + j] >> 4) - 8) * d;
                }
            }
        }
        return true;
    }
};

// y = down(silu(gate x) * up x), double accumulation
static void ref_expert(const std::vector<float> & up, const std::vector<float> & gate, const std::vector<float> & down,
                       int64_t n_ff, int64_t n_embd, const float * x, bool bf16, std::vector<double> & y) {
    auto wv = [&](float v) { return (double) (bf16 ? bf16_round(v) : v); };
    std::vector<double> a(n_ff, 0.0), g(n_ff, 0.0);
    for (int64_t r = 0; r < n_embd; ++r) {
        const double xr = x[r];
        for (int64_t o = 0; o < n_ff; ++o) {
            a[o] += wv(up[r * n_ff + o]) * xr;
            g[o] += wv(gate[r * n_ff + o]) * xr;
        }
    }
    for (int64_t o = 0; o < n_ff; ++o) {
        a[o] *= g[o] / (1.0 + std::exp(-g[o]));
    }
    y.assign(n_embd, 0.0);
    for (int64_t r = 0; r < n_ff; ++r) {
        for (int64_t o = 0; o < n_embd; ++o) {
            y[o] += wv(down[r * n_embd + o]) * a[r];
        }
    }
}

static double rel_l2(const float * got, const std::vector<double> & ref) {
    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        num += (got[i] - ref[i]) * (got[i] - ref[i]);
        den += ref[i] * ref[i];
    }
    return std::sqrt(num / std::max(den, 1e-300));
}

// ---- per run state ----

struct slot_stat {
    int    n = 0;
    double max_f32 = 0, max_bf16 = 0, sum_f32 = 0, sum_bf16 = 0;
    void add(double e32, double e16) {
        n++;
        max_f32 = std::max(max_f32, e32); max_bf16 = std::max(max_bf16, e16);
        sum_f32 += e32; sum_bf16 += e16;
    }
};

struct run_state {
    std::string policy;
    common_expert_policy_config cfg;
    gguf_weights * weights = nullptr;
    uint64_t invocation = 0;
    std::map<int, std::vector<float>> norm;   // layer -> ffn input (no-policy path)
    std::map<int, std::vector<int32_t>> topk; // layer -> ids (no-policy path)
    slot_stat on_cpu, on_gpu, plain;          // plain: normal llama.cpp path
    std::vector<float> layer0_out, layer0_in;
    std::vector<std::vector<float>> final_norm; // result_norm per decode call (input of the output matmul)
    std::map<int, std::vector<float>> l_out;    // layer output (residual stream) of the first decode call
    bool layer0_done = false;
    int64_t n_failed_read = 0;
    std::vector<std::vector<float>> logits;
    std::vector<llama_token> argmax;
};

static int layer_of(const char * name, const char * prefix) {
    int il;
    const size_t n = strlen(prefix);
    if (strncmp(name, prefix, n) == 0 && sscanf(name + n, "%d", &il) == 1) return il;
    return -1;
}

static std::vector<float> get_f32(const ggml_tensor * t) {
    std::vector<float> v(ggml_nelements(t));
    GGML_ASSERT(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    return v;
}

static std::vector<int32_t> get_i32(const ggml_tensor * t) {
    std::vector<int32_t> v(ggml_nelements(t));
    GGML_ASSERT(t->type == GGML_TYPE_I32);
    if (ggml_is_contiguous(t)) {
        ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    } else { // ffn_moe_topk is a view of argsort; gather rows
        for (int64_t j = 0; j < t->ne[1]; ++j) {
            ggml_backend_tensor_get(t, v.data() + j * t->ne[0], j * t->nb[1], t->ne[0] * sizeof(int32_t));
        }
    }
    return v;
}

static void check_layer(run_state & st, int il, const std::vector<float> & x, const std::vector<int32_t> & ids,
                        const std::vector<float> & out, int64_t d, int64_t k, int64_t n, const std::vector<uint8_t> * gpu) {
    std::vector<float> up, gate, down;
    int64_t n_ff = 0, n_e = 0, a = 0, b = 0;
    char name[64];
    for (int64_t i = 0; i < k * n; ++i) {
        const int e = ids[i];
        snprintf(name, sizeof(name), "blk.%d.ffn_up_exps.weight", il);
        bool ok = st.weights->expert(name, e, up, n_ff, n_e);
        snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps.weight", il);
        ok = ok && st.weights->expert(name, e, gate, a, b);
        snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.weight", il);
        ok = ok && st.weights->expert(name, e, down, a, b);
        if (!ok || n_e != d) {
            st.n_failed_read++;
            continue;
        }
        std::vector<double> y32, y16;
        const float * xi = x.data() + (i / k) * d;
        ref_expert(up, gate, down, n_ff, d, xi, false, y32);
        ref_expert(up, gate, down, n_ff, d, xi, true, y16);
        const float * got = out.data() + i * d;
        const double e32 = rel_l2(got, y32), e16 = rel_l2(got, y16);
        if (!gpu) {
            st.plain.add(e32, e16);
        } else {
            ((*gpu)[i] ? st.on_gpu : st.on_cpu).add(e32, e16);
        }
    }
}

static bool hook(ggml_tensor * t, bool ask, void * data) {
    auto & st = *static_cast<run_state *>(data);
    const bool want = layer_of(t->name, "expert_parallel-") >= 0 || layer_of(t->name, "ffn_norm-") >= 0 ||
                      layer_of(t->name, "ffn_moe_topk-") >= 0 || layer_of(t->name, "ffn_moe_down-") >= 0 ||
                      strcmp(t->name, "result_norm") == 0 || layer_of(t->name, "l_out-") >= 0;
    if (ask) return want;
    if (!want) return true;
    int il;
    if (strcmp(t->name, "result_norm") == 0) {
        st.final_norm.push_back(get_f32(t));
    } else if ((il = layer_of(t->name, "l_out-")) >= 0) {
        if (!st.l_out.count(il)) st.l_out[il] = get_f32(t);
    } else if ((il = layer_of(t->name, "expert_parallel-")) >= 0) {
        const auto x = get_f32(t->src[0]);
        const auto ids = get_i32(t->src[1]);
        const auto out = get_f32(t);
        const int64_t d = t->ne[0], k = t->src[1]->ne[0], n = t->src[1]->ne[1];
        std::vector<uint8_t> gpu(k * n, 0);
        st.cfg.callback(il, st.invocation++, (int32_t) n, (int32_t) k, ids.data(), gpu.data(), st.cfg.user_data);
        check_layer(st, il, x, ids, out, d, k, n, &gpu);
        if (il == 0 && !st.layer0_done) {
            st.layer0_out = out; st.layer0_in = x; st.layer0_done = true;
        }
    } else if ((il = layer_of(t->name, "ffn_norm-")) >= 0) {
        st.norm[il] = get_f32(t);
    } else if ((il = layer_of(t->name, "ffn_moe_topk-")) >= 0) {
        st.topk[il] = get_i32(t);
    } else if ((il = layer_of(t->name, "ffn_moe_down-")) >= 0) {
        const auto out = get_f32(t);
        const int64_t d = t->ne[0], k = t->ne[1], n = t->ne[2];
        check_layer(st, il, st.norm.at(il), st.topk.at(il), out, d, k, n, nullptr);
        if (il == 0 && !st.layer0_done) {
            st.layer0_out = out; st.layer0_in = st.norm.at(il); st.layer0_done = true;
        }
    }
    return true;
}

static llama_token argmax_of(const std::vector<float> & v) {
    return (llama_token) (std::max_element(v.begin(), v.end()) - v.begin());
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string policies = "none,all-cpu,all-gpu,alternating,rotating,cpu-first-1,cpu-first-2,cpu-first-3";
    std::string out_dir = ".";
    int steps = 8;
    std::vector<char *> rest = { argv[0] };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--policies" && i + 1 < argc) { policies = argv[++i]; }
        else if (a == "--out" && i + 1 < argc) { out_dir = argv[++i]; }
        else if (a == "--steps" && i + 1 < argc) { steps = atoi(argv[++i]); }
        else { rest.push_back(argv[i]); }
    }

    common_params params;
    common_init();
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    if (params.prompt.empty()) {
        params.prompt = "The capital of France is";
    }
    params.warmup = false;

    llama_backend_init();
    auto mparams = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (!model) {
        LOG_ERR("cannot load model\n");
        return 1;
    }
    gguf_weights weights;
    if (!weights.open(params.model.path)) {
        LOG_ERR("cannot read gguf weights\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> prompt = common_tokenize(vocab, params.prompt, llama_vocab_get_add_bos(vocab), true);

    std::vector<std::string> names;
    {
        std::stringstream ss(policies);
        std::string p;
        while (std::getline(ss, p, ',')) names.push_back(p);
    }

    std::vector<run_state> runs(names.size());
    std::vector<llama_token> forced; // teacher tokens: greedy choices of the first policy
    bool all_ok = true;
    for (size_t r = 0; r < names.size(); ++r) {
        auto & st = runs[r];
        st.policy = names[r];
        st.weights = &weights;
        common_params p = params;
        p.expert_policy = st.policy == "none" ? "" : st.policy;
        st.cfg = common_expert_policy_from_name(p.expert_policy);
        const std::string trace = out_dir + "/trace-" + st.policy + ".csv";
        if (st.cfg.callback) {
            setenv("LLAMA_EP_TRACE", trace.c_str(), 1);
        }
        auto cparams = common_context_params_to_llama(p);
        cparams.n_ctx = std::max<uint32_t>(256, prompt.size() + steps + 8);
        cparams.cb_eval = hook;
        cparams.cb_eval_user_data = &st;
        llama_context * ctx = llama_init_from_model(model, cparams);
        if (!ctx) {
            LOG_ERR("%s: context creation failed\n", st.policy.c_str());
            return 1;
        }
        auto decode = [&](std::vector<llama_token> toks) {
            if (llama_decode(ctx, llama_batch_get_one(toks.data(), (int32_t) toks.size())) != 0) return false;
            const float * lg = llama_get_logits_ith(ctx, -1);
            st.logits.emplace_back(lg, lg + n_vocab);
            st.argmax.push_back(argmax_of(st.logits.back()));
            return true;
        };
        bool ok = decode(prompt);
        for (int s = 0; ok && s < steps; ++s) {
            if (r == 0) forced.push_back(st.argmax.back());
            ok = decode({ forced[s] });
        }
        llama_free(ctx); // writes the trace
        unsetenv("LLAMA_EP_TRACE");
        if (!ok) {
            LOG_ERR("%s: decode failed\n", st.policy.c_str());
            all_ok = false;
        }
    }

    // ---- report ----
    const std::string json_path = out_dir + "/ep-check.json";
    FILE * jf = fopen(json_path.c_str(), "w");
    fprintf(jf, "{\n  \"model\": \"%s\",\n  \"prompt_tokens\": %zu,\n  \"steps\": %d,\n  \"n_threads\": %d,\n  \"runs\": [\n",
            params.model.path.c_str(), prompt.size(), steps, params.cpuparams.n_threads);
    const auto & ref = runs[0];
    printf("\n%-14s %8s %10s %10s %8s %10s %10s %8s %10s %10s %9s %9s %7s %9s %9s\n", "policy", "cpu_n", "cpu_f32", "cpu_bf16",
           "gpu_n", "gpu_f32", "gpu_bf16", "plain_n", "pl_f32", "pl_bf16", "lgt_max", "lgt_rms", "argmax", "l0_diff", "hid_diff");
    for (size_t r = 0; r < runs.size(); ++r) {
        const auto & st = runs[r];
        double lmax = 0, lsum = 0;
        size_t ln = 0, match = 0, nsteps = std::min(st.logits.size(), ref.logits.size());
        for (size_t s = 0; s < nsteps; ++s) {
            for (int v = 0; v < n_vocab; ++v) {
                const double dd = st.logits[s][v] - ref.logits[s][v];
                lmax = std::max(lmax, std::fabs(dd)); lsum += dd * dd; ln++;
            }
            match += st.argmax[s] == ref.argmax[s];
        }
        double hmax = 0;
        for (size_t s = 0; s < std::min(st.final_norm.size(), ref.final_norm.size()); ++s) {
            const auto & a = st.final_norm[s];
            const auto & b = ref.final_norm[s];
            for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) hmax = std::max(hmax, (double) std::fabs(a[i] - b[i]));
        }
        double l0 = -1;
        bool same_in = st.layer0_in == ref.layer0_in;
        if (st.layer0_out.size() == ref.layer0_out.size()) {
            l0 = 0;
            for (size_t i = 0; i < st.layer0_out.size(); ++i) l0 = std::max(l0, (double) std::fabs(st.layer0_out[i] - ref.layer0_out[i]));
        }
        auto worst = [](const slot_stat & s) { return s.n ? s.max_f32 : 0.0; };
        printf("%-14s %8d %10.2e %10.2e %8d %10.2e %10.2e %8d %10.2e %10.2e %9.2e %9.2e %3zu/%-3zu %9.2e %9.2e%s\n", st.policy.c_str(),
               st.on_cpu.n, st.on_cpu.max_f32, st.on_cpu.max_bf16, st.on_gpu.n, st.on_gpu.max_f32, st.on_gpu.max_bf16,
               st.plain.n, st.plain.max_f32, st.plain.max_bf16, lmax, std::sqrt(lsum / std::max<size_t>(ln, 1)), match, nsteps, l0, hmax,
               same_in ? "" : " (layer0 input differs)");
        (void) worst;
        auto js = [&](const char * key, const slot_stat & s) {
            fprintf(jf, "      \"%s\": {\"n\": %d, \"max_rel_l2_f32\": %.6e, \"max_rel_l2_bf16\": %.6e, \"mean_rel_l2_f32\": %.6e, \"mean_rel_l2_bf16\": %.6e},\n",
                    key, s.n, s.max_f32, s.max_bf16, s.n ? s.sum_f32 / s.n : 0.0, s.n ? s.sum_bf16 / s.n : 0.0);
        };
        fprintf(jf, "    {\n      \"policy\": \"%s\",\n", st.policy.c_str());
        js("cpu_slots", st.on_cpu); js("gpu_slots", st.on_gpu); js("plain_slots", st.plain);
        fprintf(jf, "      \"weight_read_failures\": %" PRId64 ",\n", st.n_failed_read);
        fprintf(jf, "      \"logits_vs_first\": {\"max_abs\": %.6e, \"rms\": %.6e, \"argmax_match\": %zu, \"positions\": %zu},\n",
                lmax, std::sqrt(lsum / std::max<size_t>(ln, 1)), match, nsteps);
        fprintf(jf, "      \"final_hidden_max_abs_vs_first\": %.6e,\n", hmax);
        // residual stream drift per layer (first decode call): max abs diff, and rms diff / rms of reference
        fprintf(jf, "      \"l_out_drift\": [");
        for (const auto & [il, a] : st.l_out) {
            if (!ref.l_out.count(il) || ref.l_out.at(il).size() != a.size()) continue;
            const auto & b = ref.l_out.at(il);
            double mx = 0, d2 = 0, r2 = 0;
            for (size_t i = 0; i < a.size(); ++i) { const double dd = a[i] - b[i]; mx = std::max(mx, std::fabs(dd)); d2 += dd * dd; r2 += (double) b[i] * b[i]; }
            fprintf(jf, "%s{\"layer\": %d, \"max_abs\": %.3e, \"rel_rms\": %.3e}", il ? ", " : "", il, mx, std::sqrt(d2 / std::max(r2, 1e-300)));
        }
        fprintf(jf, "],\n");
        fprintf(jf, "      \"layer0_in_identical\": %s,\n      \"layer0_out_max_abs_vs_first\": %.6e,\n", same_in ? "true" : "false", l0);
        fprintf(jf, "      \"argmax\": [");
        for (size_t s = 0; s < st.argmax.size(); ++s) fprintf(jf, "%s%d", s ? ", " : "", st.argmax[s]);
        fprintf(jf, "]\n    }%s\n", r + 1 < runs.size() ? "," : "");
        all_ok = all_ok && st.n_failed_read == 0;
    }
    fprintf(jf, "  ]\n}\n");
    fclose(jf);
    printf("\nrel_l2 columns: worst relative L2 error of one expert output vs the double reference with F32 or BF16 weights\n");
    printf("lgt_*: logits vs %s on the same tokens; l0_diff: max abs diff of layer 0 expert outputs vs %s\n", ref.policy.c_str(), ref.policy.c_str());
    printf("hid_diff: max abs diff of the final normed hidden state (input of the output matmul) vs %s\n", ref.policy.c_str());
    printf("wrote %s\n", json_path.c_str());

    llama_model_free(model);
    llama_backend_free();
    return all_ok ? 0 : 1;
}
