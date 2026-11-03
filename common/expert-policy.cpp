#include "expert-policy.h"

#include <stdexcept>

enum class common_expert_policy { all_cpu, all_gpu, alternating, expert_id, rotating };

static void common_expert_policy_callback(int32_t, uint64_t invocation, int32_t n_tokens, int32_t top_k, const int32_t * expert_ids, uint8_t * use_gpu, void * data) {
    const auto policy = *static_cast<const common_expert_policy *>(data);
    for (int32_t i = 0; i < n_tokens*top_k; ++i) {
        const int32_t slot = i % top_k;
        switch (policy) {
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
        }
    }
}

common_expert_policy_config common_expert_policy_from_name(const std::string & name) {
    if (name.empty()) {
        return {};
    }
    static struct {
        const char * name;
        common_expert_policy policy;
    } policies[] = {
        { "all-cpu",     common_expert_policy::all_cpu     },
        { "all-gpu",     common_expert_policy::all_gpu     },
        { "alternating", common_expert_policy::alternating },
        { "expert-id",   common_expert_policy::expert_id   },
        { "rotating",    common_expert_policy::rotating    },
    };
    for (auto & entry : policies) {
        if (name == entry.name) {
            return { common_expert_policy_callback, &entry.policy };
        }
    }
    throw std::invalid_argument("unknown expert policy: " + name);
}
