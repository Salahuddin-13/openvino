// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {
Output<Node> apply_smooth_l1(const NodeContext& context,
                              Output<Node> input,
                              Output<Node> target,
                              double beta) {
    // diff = |input - target|
    auto diff = context.mark_node(std::make_shared<v1::Subtract>(input, target));
    diff = context.mark_node(std::make_shared<v0::Abs>(diff));

    if (beta == 0.0) {
        // pure L1 loss
        return diff;
    }

    // l2_case = 0.5 * diff^2 / beta
    auto const_2 = context.mark_node(v0::Constant::create(element::f32, Shape{}, {2.0f}));
    const_2 = context.mark_node(std::make_shared<v1::ConvertLike>(const_2, diff));
    auto const_half = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0.5f}));
    const_half = context.mark_node(std::make_shared<v1::ConvertLike>(const_half, diff));
    auto beta_const = context.mark_node(v0::Constant::create(element::f32, Shape{}, {static_cast<float>(beta)}));
    beta_const = context.mark_node(std::make_shared<v1::ConvertLike>(beta_const, diff));
    auto half_beta = context.mark_node(v0::Constant::create(element::f32, Shape{}, {static_cast<float>(0.5 * beta)}));
    half_beta = context.mark_node(std::make_shared<v1::ConvertLike>(half_beta, diff));

    auto diff_sq = context.mark_node(std::make_shared<v1::Power>(diff, const_2));
    auto l2_case = context.mark_node(std::make_shared<v1::Multiply>(const_half, diff_sq));
    l2_case = context.mark_node(std::make_shared<v1::Divide>(l2_case, beta_const));

    // l1_case = diff - 0.5 * beta
    auto l1_case = context.mark_node(std::make_shared<v1::Subtract>(diff, half_beta));

    // select based on diff < beta
    auto cond = context.mark_node(std::make_shared<v1::Less>(diff, beta_const));
    return context.mark_node(std::make_shared<v1::Select>(cond, l2_case, l1_case));
}

Output<Node> apply_reduction(const NodeContext& context, Output<Node> loss, int64_t reduction) {
    if (reduction == 0) {
        // none
        return loss;
    }
    auto axes = context.mark_node(get_axes_range(context, 0));
    if (reduction == 1) {
        // mean
        return context.mark_node(std::make_shared<v1::ReduceMean>(loss, axes, false));
    }
    // sum
    return context.mark_node(std::make_shared<v1::ReduceSum>(loss, axes, false));
}
}  // namespace

OutputVector translate_smooth_l1_loss(const NodeContext& context) {
    // aten::smooth_l1_loss(Tensor self, Tensor target, int reduction=1, float beta=1.0) -> Tensor
    num_inputs_check(context, 2, 4);
    auto input = context.get_input(0);
    auto target = context.get_input(1);
    int64_t reduction = 1;
    double beta = 1.0;
    if (!context.input_is_none(2)) {
        reduction = context.const_input<int64_t>(2);
    }
    if (!context.input_is_none(3)) {
        beta = context.const_input<double>(3);
    }
    auto loss = apply_smooth_l1(context, input, target, beta);
    return {apply_reduction(context, loss, reduction)};
}

OutputVector translate_smooth_l1_loss_fx(const NodeContext& context) {
    // aten.smooth_l1_loss.default — same inputs/signature as TorchScript variant
    return translate_smooth_l1_loss(context);
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
