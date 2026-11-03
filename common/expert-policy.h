#pragma once

#include "llama.h"

#include <string>

struct common_expert_policy_config {
    llama_expert_callback callback = nullptr;
    void * user_data = nullptr;
};

common_expert_policy_config common_expert_policy_from_name(const std::string & name);
