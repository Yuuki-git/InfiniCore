#include "infinicore/ss5_ninetoothed.hpp"
#include "infinicore/ops/common/op.hpp"

#include "ss5_s3_adaln.h"
#include "ss5_s3_gate.h"
#include "ss5_s4_attention.h"

#include <atomic>
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

extern "C" const char *ss5_launch_phase_ss5_s3_adaln();
extern "C" const char *ss5_launch_phase_ss5_s3_gate();
extern "C" const char *ss5_launch_phase_ss5_s4_attention();

namespace {

bool enabled(const char *name) {
    const char *value = std::getenv(name);
    return value && std::strcmp(value, "1") == 0;
}

const bool use_attention = enabled("SS5_NT_ATTN");
const bool use_fusion = enabled("SS5_NT_ADALN");
std::atomic<uint64_t> calls[3]{};

using infinicore::Tensor;

bool aligned(const Tensor &tensor, const infinicore::Device &device) {
    return tensor && tensor->device() == device && tensor->dtype() == infinicore::DataType::BF16
        && tensor->data() && reinterpret_cast<uintptr_t>(tensor->data()) % 16 == 0;
}

bool fusion_layout(const Tensor &tensor, const Tensor &reference) {
    if (!aligned(tensor, reference->device()) || tensor->shape() != reference->shape()) {
        return false;
    }
    const auto rank = tensor->ndim();
    if ((rank != 2 && rank != 3) || (rank == 3 && tensor->size(0) != 1)) {
        return false;
    }
    const auto rows = tensor->size(rank - 2);
    const auto stride = tensor->stride(rank - 2);
    return tensor->size(rank - 1) == 5376 && tensor->stride(rank - 1) == 1
        && rows > 0 && stride >= 5376 && stride % 16 == 0 && stride <= INT32_MAX
        && rows <= static_cast<size_t>((INT32_MAX - 5376) / stride);
}

bool fusion_inputs(const Tensor &a, const Tensor &b, const Tensor &c) {
    return a && a->device().getType() == infinicore::Device::Type::HYGON
        && fusion_layout(a, a) && fusion_layout(b, a) && fusion_layout(c, a);
}

struct View2 {
    uint64_t shape[2];
    int64_t strides[2];
    void *data;

    explicit View2(const Tensor &tensor)
        : shape{tensor->size(tensor->ndim() - 2), tensor->size(tensor->ndim() - 1)},
          strides{tensor->stride(tensor->ndim() - 2), tensor->stride(tensor->ndim() - 1)},
          data(const_cast<std::byte *>(tensor->data())) {}

    NineToothedTensor view() { return {data, shape, strides}; }
};

struct FusionPlan {
    infinicore::graph::GraphTensor output, a, b, c;
};

void *plan_fusion(Tensor output, const Tensor &a, const Tensor &b, const Tensor &c) {
    return new FusionPlan{infinicore::graph::GraphTensor(output), infinicore::graph::GraphTensor(a),
                          infinicore::graph::GraphTensor(b), infinicore::graph::GraphTensor(c)};
}

void check_status(int status, const char *operation, const char *phase) {
    if (status) {
        throw std::runtime_error(std::string(operation) + " at " + phase + " returned " + std::to_string(status));
    }
}

void run_adaln(void *opaque) {
    auto *p = static_cast<FusionPlan *>(opaque);
    View2 output(p->output), x(p->a), scale(p->b), shift(p->c);
    auto status = launch_ss5_s3_adaln(infinicore::context::getStream(), x.view(), scale.view(), shift.view(), output.view());
    check_status(status, "NineToothed AdaLN", ss5_launch_phase_ss5_s3_adaln());
    calls[1].fetch_add(1, std::memory_order_relaxed);
}

void run_gate(void *opaque) {
    auto *p = static_cast<FusionPlan *>(opaque);
    View2 output(p->output), residual(p->a), branch(p->b), gate(p->c);
    auto status = launch_ss5_s3_gate(infinicore::context::getStream(), residual.view(), branch.view(), gate.view(), output.view());
    check_status(status, "NineToothed gated residual", ss5_launch_phase_ss5_s3_gate());
    calls[2].fetch_add(1, std::memory_order_relaxed);
}

void cleanup_fusion(void **opaque) {
    delete static_cast<FusionPlan *>(*opaque);
    *opaque = nullptr;
}

bool attention_layout(const Tensor &tensor, const infinicore::Device &device, bool packed) {
    if (!aligned(tensor, device) || tensor->shape() != infinicore::Shape{1, 27570, 7, 128}) {
        return false;
    }
    const infinicore::Strides strides = packed ? infinicore::Strides{74108160, 2688, 128, 1}
                                               : infinicore::Strides{24702720, 896, 128, 1};
    return tensor->strides() == strides;
}

bool overlap(const Tensor &output, const Tensor &input, size_t input_elements) {
    const auto out = reinterpret_cast<uintptr_t>(output->data());
    const auto in = reinterpret_cast<uintptr_t>(input->data());
    return out < in + input_elements * 2 && in < out + 24702720 * 2;
}

struct View4 {
    uint64_t shape[4] = {1, 7, 27570, 128};
    int64_t strides[4];
    void *data;

    explicit View4(const Tensor &tensor)
        : strides{tensor->stride(0), tensor->stride(2), tensor->stride(1), tensor->stride(3)},
          data(const_cast<std::byte *>(tensor->data())) {}

    NineToothedTensor view() { return {data, shape, strides}; }
};

} // namespace

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(SS5AdaLN, Tensor, const Tensor &, const Tensor &, const Tensor &);
INFINICORE_GRAPH_OP_CLASS(SS5GatedResidual, Tensor, const Tensor &, const Tensor &, const Tensor &);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(SS5AdaLN);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(SS5GatedResidual);

SS5AdaLN::SS5AdaLN(Tensor output, const Tensor &x, const Tensor &scale, const Tensor &shift) {
    INFINICORE_GRAPH_OP_DISPATCH(output->device().getType(), output, x, scale, shift);
}

void SS5AdaLN::execute(Tensor output, const Tensor &x, const Tensor &scale, const Tensor &shift) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(SS5AdaLN, output, x, scale, shift);
}

SS5GatedResidual::SS5GatedResidual(Tensor output, const Tensor &residual, const Tensor &branch, const Tensor &gate) {
    INFINICORE_GRAPH_OP_DISPATCH(output->device().getType(), output, residual, branch, gate);
}

void SS5GatedResidual::execute(Tensor output, const Tensor &residual, const Tensor &branch, const Tensor &gate) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(SS5GatedResidual, output, residual, branch, gate);
}

static bool registered = [] {
    const auto device = Device::Type::HYGON;
    SS5AdaLN::plan_dispatcher().registerDevice(device, &plan_fusion);
    SS5AdaLN::run_dispatcher().registerDevice(device, &run_adaln);
    SS5AdaLN::cleanup_dispatcher().registerDevice(device, &cleanup_fusion);
    SS5GatedResidual::plan_dispatcher().registerDevice(device, &plan_fusion);
    SS5GatedResidual::run_dispatcher().registerDevice(device, &run_gate);
    SS5GatedResidual::cleanup_dispatcher().registerDevice(device, &cleanup_fusion);
    return true;
}();

} // namespace infinicore::op

namespace infinicore::ss5_nt {

bool attention(Tensor output, const Tensor &query, const Tensor &key, const Tensor &value,
               bool has_alibi, float scale, bool is_causal) {
    if (!use_attention || has_alibi || is_causal || !output
        || output->device().getType() != Device::Type::HYGON) {
        return false;
    }
    const auto device = output->device();
    if (!attention_layout(output, device, false) || !attention_layout(query, device, false)
        || !attention_layout(key, device, false) || !attention_layout(value, device, true)
        || overlap(output, query, 24702720) || overlap(output, key, 24702720)
        || overlap(output, value, 27569 * 2688 + 896)) {
        return false;
    }
    View4 out(output), q(query), k(key), v(value);
    double scale_storage = scale;
    NineToothedTensor factor{&scale_storage, nullptr, nullptr};
    auto status = launch_ss5_s4_attention(context::getStream(), q.view(), k.view(), v.view(), factor, out.view());
    check_status(status, "NineToothed attention", ss5_launch_phase_ss5_s4_attention());
    calls[0].fetch_add(1, std::memory_order_relaxed);
    return true;
}

Tensor adaln(const Tensor &normalized, const Tensor &scale, const Tensor &shift) {
    if (!use_fusion || !fusion_inputs(normalized, scale, shift)) {
        return {};
    }
    auto output = Tensor::empty(normalized->shape(), normalized->dtype(), normalized->device());
    op::SS5AdaLN::execute(output, normalized, scale, shift);
    return output;
}

Tensor gated_residual(const Tensor &residual, const Tensor &branch, const Tensor &gate) {
    if (!use_fusion || !fusion_inputs(residual, branch, gate)) {
        return {};
    }
    auto output = Tensor::empty(residual->shape(), residual->dtype(), residual->device());
    op::SS5GatedResidual::execute(output, residual, branch, gate);
    return output;
}

} // namespace infinicore::ss5_nt

extern "C" unsigned ss5_nt_switches() {
    return unsigned(use_attention) | (unsigned(use_fusion) << 1);
}

extern "C" uint64_t ss5_nt_calls(unsigned operation) {
    return operation < 3 ? calls[operation].load(std::memory_order_relaxed) : 0;
}
