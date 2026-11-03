#include "expert-policy.h"

#include <list>
#include <stdexcept>

enum class common_expert_policy { all_cpu, all_gpu, alternating, expert_id, rotating, cpu_first, cpu_last };

struct common_expert_policy_data {
    common_expert_policy policy;
    int32_t n_cpu; // cpu_first, cpu_last: number of top-k slots per token on CPU
};

static void common_expert_policy_callback(int32_t, uint64_t invocation, int32_t n_tokens, int32_t top_k, const int32_t * expert_ids, uint8_t * use_gpu, void * data) {
    const auto & cfg = *static_cast<const common_expert_policy_data *>(data);
    for (int32_t i = 0; i < n_tokens*top_k; ++i) {
        const int32_t slot = i % top_k;
        switch (cfg.policy) {
            case common_expert_policy::all_cpu:
                use_gpu[i] = 0;
                break;
            case common_expert_policy::all_gpu:
                use_gpu[i] = 1;
                break;
            case common_expert_policy::alternating:
                // Odd top-K slots run on GPU; even slots run on CPU.
                use_gpu[i] = slot % 2 == 1;
                break;
            case common_expert_policy::expert_id:
                // Route by the selected expert's ID, independent of its slot.
                use_gpu[i] = expert_ids[i] % 2 == 1;
                break;
            case common_expert_policy::rotating:
                // Shift the GPU slots after each layer invocation.
                use_gpu[i] = (slot + invocation) % 3 == 0;
                break;
            case common_expert_policy::cpu_first:
                use_gpu[i] = slot >= cfg.n_cpu;
                break;
            case common_expert_policy::cpu_last:
                use_gpu[i] = slot < top_k - cfg.n_cpu;
                break;
        }
    }
}

common_expert_policy_config common_expert_policy_from_name(const std::string & name) {
    if (name.empty()) {
        return {};
    }
    static const struct {
        const char * name;
        common_expert_policy policy;
    } policies[] = {
        { "all-cpu",     common_expert_policy::all_cpu     },
        { "all-gpu",     common_expert_policy::all_gpu     },
        { "alternating", common_expert_policy::alternating },
        { "expert-id",   common_expert_policy::expert_id   },
        { "rotating",    common_expert_policy::rotating    },
        { "cpu-first-",  common_expert_policy::cpu_first   },
        { "cpu-last-",   common_expert_policy::cpu_last    },
    };
    // policy data must stay valid while any context uses it
    static std::list<common_expert_policy_data> storage;
    for (const auto & entry : policies) {
        const std::string prefix = entry.name;
        if (entry.policy == common_expert_policy::cpu_first || entry.policy == common_expert_policy::cpu_last) {
            if (name.rfind(prefix, 0) != 0 || name.size() == prefix.size()) {
                continue;
            }
            size_t pos = 0;
            const int n_cpu = std::stoi(name.substr(prefix.size()), &pos);
            if (pos != name.size() - prefix.size() || n_cpu < 0) {
                throw std::invalid_argument("invalid expert policy: " + name);
            }
            storage.push_back({ entry.policy, n_cpu });
            return { common_expert_policy_callback, &storage.back() };
        }
        if (name == prefix) {
            storage.push_back({ entry.policy, 0 });
            return { common_expert_policy_callback, &storage.back() };
        }
    }
    throw std::invalid_argument("unknown expert policy: " + name);
}
