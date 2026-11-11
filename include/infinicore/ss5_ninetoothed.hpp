#pragma once

#include <infinicore/tensor.hpp>
#include <cstdint>

namespace infinicore::ss5_nt {

bool attention(Tensor output, const Tensor &query, const Tensor &key, const Tensor &value,
               bool has_alibi, float scale, bool is_causal);
Tensor adaln(const Tensor &normalized, const Tensor &scale, const Tensor &shift);
Tensor gated_residual(const Tensor &residual, const Tensor &branch, const Tensor &gate);

} // namespace infinicore::ss5_nt

extern "C" unsigned ss5_nt_switches();
extern "C" uint64_t ss5_nt_calls(unsigned operation);
