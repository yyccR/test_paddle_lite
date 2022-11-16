#pragma once
#include "paddle_lite_factory_helper.h"

USE_LITE_KERNEL(correlation, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(fc, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(fc, kX86, kInt8, kNCHW, int8_out);
USE_LITE_KERNEL(fc, kX86, kInt8, kNCHW, fp32_out);
USE_LITE_KERNEL(grid_sampler, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(sequence_mask, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(sequence_mask, kHost, kFloat, kNCHW, int32);
USE_LITE_KERNEL(sequence_mask, kHost, kFloat, kNCHW, int64);
USE_LITE_KERNEL(feed, kHost, kAny, kAny, def);
USE_LITE_KERNEL(reshape, kHost, kAny, kAny, def);
USE_LITE_KERNEL(reshape2, kHost, kAny, kAny, def);
USE_LITE_KERNEL(flatten, kHost, kAny, kAny, def);
USE_LITE_KERNEL(flatten2, kHost, kAny, kAny, def);
USE_LITE_KERNEL(sequence_arithmetic, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(search_seq_arithmetic, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(unsqueeze, kHost, kAny, kAny, def);
USE_LITE_KERNEL(unsqueeze2, kHost, kAny, kAny, def);
USE_LITE_KERNEL(stack, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(sequence_pool, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(yolo_box, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(instance_norm, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(top_k_v2, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(gaussian_random, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(fetch, kHost, kAny, kAny, def);
USE_LITE_KERNEL(matmul, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(sequence_reshape, kX86, kInt64, kNCHW, def);
USE_LITE_KERNEL(sequence_reshape, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(fill_constant_batch_size_like, kHost, kAny, kNCHW, def);
USE_LITE_KERNEL(conv2d, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(depthwise_conv2d, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(conv2d, kX86, kInt8, kNCHW, int8_out);
USE_LITE_KERNEL(conv2d, kX86, kInt8, kNCHW, fp32_out);
USE_LITE_KERNEL(depthwise_conv2d, kX86, kInt8, kNCHW, int8_out);
USE_LITE_KERNEL(depthwise_conv2d, kX86, kInt8, kNCHW, fp32_out);
USE_LITE_KERNEL(im2sequence, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(roi_perspective_transform, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(gru, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(logical_xor, kHost, kAny, kAny, def);
USE_LITE_KERNEL(logical_and, kHost, kAny, kAny, def);
USE_LITE_KERNEL(logical_or, kHost, kAny, kAny, def);
USE_LITE_KERNEL(logical_not, kHost, kAny, kAny, def);
USE_LITE_KERNEL(box_clip, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(inverse, kHost, kFloat, kNCHW, fp32);
USE_LITE_KERNEL(reduce_all, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(reduce_any, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(collect_fpn_proposals, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(crop_tensor, kHost, kFloat, kAny, def);
USE_LITE_KERNEL(crop_tensor, kHost, kFloat, kAny, int32_precision);
USE_LITE_KERNEL(polygon_box_transform, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(deformable_conv, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(range, kHost, kFloat, kAny, def);
USE_LITE_KERNEL(range, kHost, kInt64, kAny, def);
USE_LITE_KERNEL(range, kHost, kInt32, kAny, def);
USE_LITE_KERNEL(range, kHost, kFloat, kAny, int32);
USE_LITE_KERNEL(range, kHost, kFloat, kAny, int64);
USE_LITE_KERNEL(box_coder, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(prior_box, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(density_prior_box, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(flip, kHost, kAny, kNCHW, flip_fp32);
USE_LITE_KERNEL(flip, kHost, kAny, kNCHW, flip_i64);
USE_LITE_KERNEL(cos_sim, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(flatten_contiguous_range, kHost, kAny, kAny, def);
USE_LITE_KERNEL(sampling_id, kHost, kAny, kAny, float32);
USE_LITE_KERNEL(elementwise_add, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_add, kX86, kFloat, kNCHW, int32);
USE_LITE_KERNEL(elementwise_add, kX86, kFloat, kNCHW, int64);
USE_LITE_KERNEL(fusion_elementwise_add_activation, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_sub, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_sub, kX86, kFloat, kNCHW, int32);
USE_LITE_KERNEL(elementwise_sub, kX86, kFloat, kNCHW, int64);
USE_LITE_KERNEL(fusion_elementwise_sub_activation, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_mul, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_mul, kX86, kFloat, kNCHW, int32);
USE_LITE_KERNEL(elementwise_mul, kX86, kFloat, kNCHW, int64);
USE_LITE_KERNEL(fusion_elementwise_mul_activation, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_div, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_div, kX86, kFloat, kNCHW, int32);
USE_LITE_KERNEL(elementwise_div, kX86, kFloat, kNCHW, int64);
USE_LITE_KERNEL(fusion_elementwise_div_activation, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_floordiv, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_floordiv, kX86, kFloat, kNCHW, int32);
USE_LITE_KERNEL(elementwise_floordiv, kX86, kFloat, kNCHW, int64);
USE_LITE_KERNEL(elementwise_pow, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_pow, kX86, kFloat, kNCHW, int32);
USE_LITE_KERNEL(elementwise_pow, kX86, kFloat, kNCHW, int64);
USE_LITE_KERNEL(elementwise_mod, kX86, kFloat, kNCHW, int32);
USE_LITE_KERNEL(elementwise_mod, kX86, kFloat, kNCHW, int64);
USE_LITE_KERNEL(elementwise_max, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_max, kX86, kFloat, kNCHW, int32);
USE_LITE_KERNEL(elementwise_max, kX86, kFloat, kNCHW, int64);
USE_LITE_KERNEL(fusion_elementwise_max_activation, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(fusion_elementwise_min_activation, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_min, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_min, kX86, kFloat, kNCHW, int32);
USE_LITE_KERNEL(elementwise_min, kX86, kFloat, kNCHW, int64);
USE_LITE_KERNEL(gru_unit, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(write_to_array, kHost, kAny, kAny, def);
USE_LITE_KERNEL(lod_array_length, kHost, kAny, kAny, def);
USE_LITE_KERNEL(uniform_random, kHost, kAny, kAny, def);
USE_LITE_KERNEL(retinanet_detection_output, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(sequence_pad, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(sequence_pad, kHost, kFloat, kNCHW, int32);
USE_LITE_KERNEL(sequence_pad, kHost, kFloat, kNCHW, int64);
USE_LITE_KERNEL(search_attention_padding_mask, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(fill_zeros_like, kHost, kFloat, kNCHW, float32);
USE_LITE_KERNEL(fill_zeros_like, kHost, kFloat, kNCHW, int32);
USE_LITE_KERNEL(fill_zeros_like, kHost, kFloat, kNCHW, int64);
USE_LITE_KERNEL(lod_reset, kHost, kAny, kNCHW, def);
USE_LITE_KERNEL(fill_any_like, kHost, kAny, kNCHW, def);
USE_LITE_KERNEL(fill_zeros_like, kHost, kAny, kNCHW, def);
USE_LITE_KERNEL(beam_search, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(select_input, kHost, kAny, kNCHW, def);
USE_LITE_KERNEL(crop, kHost, kFloat, kAny, def);
USE_LITE_KERNEL(crop, kHost, kInt32, kAny, def);
USE_LITE_KERNEL(stack, kHost, kFloat, kAny, def);
USE_LITE_KERNEL(stack, kHost, kFloat, kAny, int32_def);
USE_LITE_KERNEL(stack, kHost, kFloat, kAny, int64_def);
USE_LITE_KERNEL(slice, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(slice, kX86, kFloat, kNCHW, int32);
USE_LITE_KERNEL(slice, kX86, kFloat, kNCHW, int64);
USE_LITE_KERNEL(pixel_shuffle, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(anchor_generator, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(sequence_unpad, kHost, kFloat, kAny, float32);
USE_LITE_KERNEL(sequence_unpad, kHost, kFloat, kAny, int64);
USE_LITE_KERNEL(squeeze, kHost, kAny, kAny, def);
USE_LITE_KERNEL(squeeze2, kHost, kAny, kAny, def);
USE_LITE_KERNEL(beam_search_decode, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(scatter_nd_add, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(scatter_nd_add, kHost, kFloat, kNCHW, float32_int64);
USE_LITE_KERNEL(scatter_nd_add, kHost, kFloat, kNCHW, int32_int32);
USE_LITE_KERNEL(scatter_nd_add, kHost, kFloat, kNCHW, int32_int64);
USE_LITE_KERNEL(scatter_nd_add, kHost, kFloat, kNCHW, int64_int32);
USE_LITE_KERNEL(scatter_nd_add, kHost, kFloat, kNCHW, int64_int64);
USE_LITE_KERNEL(matrix_nms, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(multiclass_nms, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(multiclass_nms2, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(multiclass_nms3, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(tensor_array_to_tensor, kHost, kAny, kNCHW, def);
USE_LITE_KERNEL(expand_as, kHost, kFloat, kAny, def);
USE_LITE_KERNEL(expand_as, kHost, kFloat, kAny, int64);
USE_LITE_KERNEL(cumsum, kHost, kFloat, kAny, float32);
USE_LITE_KERNEL(cumsum, kHost, kFloat, kAny, int32);
USE_LITE_KERNEL(cumsum, kHost, kFloat, kAny, int64);
USE_LITE_KERNEL(layer_norm, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(scale, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(scale, kX86, kFloat, kNCHW, int32);
USE_LITE_KERNEL(scale, kX86, kFloat, kNCHW, int64);
USE_LITE_KERNEL(var_conv_2d, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(unique_with_counts, kHost, kAny, kAny, def);
USE_LITE_KERNEL(relu, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(leaky_relu, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(relu_clipped, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(prelu, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(sigmoid, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(tanh, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(swish, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(relu6, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(log, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(exp, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(floor, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(hard_sigmoid, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(rsqrt, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(square, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(hard_swish, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(reciprocal, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(abs, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(thresholded_relu, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(elu, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(softplus, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(ctc_align, kHost, kInt64, kNCHW, def);
USE_LITE_KERNEL(ctc_align, kHost, kInt32, kNCHW, def);
USE_LITE_KERNEL(shape, kHost, kAny, kAny, def);
USE_LITE_KERNEL(conditional_block, kHost, kAny, kAny, def);
USE_LITE_KERNEL(gather, kHost, kFloat, kNCHW, int32int32);
USE_LITE_KERNEL(gather, kHost, kFloat, kNCHW, int64int64);
USE_LITE_KERNEL(gather, kHost, kFloat, kNCHW, int64int32);
USE_LITE_KERNEL(gather, kHost, kFloat, kNCHW, int32int64);
USE_LITE_KERNEL(reduce_mean, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(reduce_sum, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(reduce_sum, kX86, kFloat, kNCHW, int32);
USE_LITE_KERNEL(reduce_sum, kX86, kFloat, kNCHW, int64);
USE_LITE_KERNEL(reduce_prod, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(reduce_prod, kX86, kFloat, kNCHW, int32);
USE_LITE_KERNEL(reduce_prod, kX86, kFloat, kNCHW, int64);
USE_LITE_KERNEL(reduce_max, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(reduce_max, kX86, kFloat, kNCHW, int32);
USE_LITE_KERNEL(reduce_max, kX86, kFloat, kNCHW, int64);
USE_LITE_KERNEL(reduce_min, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(reduce_min, kX86, kFloat, kNCHW, int32);
USE_LITE_KERNEL(reduce_min, kX86, kFloat, kNCHW, int64);
USE_LITE_KERNEL(distribute_fpn_proposals, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(unfold, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(unfold, kHost, kFloat, kNCHW, def_int32);
USE_LITE_KERNEL(unfold, kHost, kFloat, kNCHW, def_int64);
USE_LITE_KERNEL(unfold, kHost, kInt8, kNCHW, def_int8);
USE_LITE_KERNEL(write_back, kHost, kAny, kAny, write_back);
USE_LITE_KERNEL(roi_align, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(expand, kHost, kAny, kAny, def);
USE_LITE_KERNEL(mul, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(reverse, kHost, kAny, kNCHW, fp32);
USE_LITE_KERNEL(search_group_padding, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(sin, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(search_seq_depadding, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(gather_tree, kHost, kFloat, kAny, int32);
USE_LITE_KERNEL(gather_tree, kHost, kFloat, kAny, int64);
USE_LITE_KERNEL(arg_max, kHost, kAny, kNCHW, fp32);
USE_LITE_KERNEL(arg_max, kHost, kAny, kNCHW, int64);
USE_LITE_KERNEL(arg_max, kHost, kAny, kNCHW, int32);
USE_LITE_KERNEL(arg_max, kHost, kAny, kNCHW, int16);
USE_LITE_KERNEL(arg_max, kHost, kAny, kNCHW, uint8);
USE_LITE_KERNEL(print, kHost, kAny, kAny, def);
USE_LITE_KERNEL(strided_slice, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(strided_slice, kHost, kFloat, kNCHW, def_int32);
USE_LITE_KERNEL(strided_slice, kHost, kFloat, kNCHW, def_int64);
USE_LITE_KERNEL(increment, kHost, kAny, kNCHW, def);
USE_LITE_KERNEL(sequence_expand, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(sequence_expand, kHost, kFloat, kNCHW, int32);
USE_LITE_KERNEL(sequence_expand, kHost, kFloat, kNCHW, int64);
USE_LITE_KERNEL(where_index, kHost, kAny, kAny, def);
USE_LITE_KERNEL(equal, kHost, kFloat, kAny, def);
USE_LITE_KERNEL(equal, kHost, kInt64, kAny, def);
USE_LITE_KERNEL(equal, kHost, kFloat, kAny, int64);
USE_LITE_KERNEL(equal, kHost, kInt32, kAny, def);
USE_LITE_KERNEL(equal, kHost, kFloat, kAny, int32);
USE_LITE_KERNEL(not_equal, kHost, kFloat, kAny, def);
USE_LITE_KERNEL(not_equal, kHost, kFloat, kAny, int32);
USE_LITE_KERNEL(not_equal, kHost, kFloat, kAny, int64);
USE_LITE_KERNEL(less_than, kHost, kFloat, kAny, def);
USE_LITE_KERNEL(less_than, kHost, kInt32, kAny, def);
USE_LITE_KERNEL(less_than, kHost, kFloat, kAny, int32);
USE_LITE_KERNEL(less_than, kHost, kInt64, kAny, def);
USE_LITE_KERNEL(less_than, kHost, kFloat, kAny, int64);
USE_LITE_KERNEL(less_equal, kHost, kFloat, kAny, def);
USE_LITE_KERNEL(less_equal, kHost, kInt64, kAny, def);
USE_LITE_KERNEL(less_equal, kHost, kFloat, kAny, int64);
USE_LITE_KERNEL(less_equal, kHost, kFloat, kAny, int32);
USE_LITE_KERNEL(greater_than, kHost, kFloat, kAny, def);
USE_LITE_KERNEL(greater_than, kHost, kFloat, kAny, def_bool);
USE_LITE_KERNEL(greater_than, kHost, kFloat, kAny, def_int32);
USE_LITE_KERNEL(greater_than, kHost, kInt64, kAny, def);
USE_LITE_KERNEL(greater_than, kHost, kFloat, kAny, def_int64);
USE_LITE_KERNEL(greater_equal, kHost, kFloat, kAny, def);
USE_LITE_KERNEL(greater_equal, kHost, kFloat, kAny, def_int64);
USE_LITE_KERNEL(greater_equal, kHost, kFloat, kAny, def_int32);
USE_LITE_KERNEL(index_select, kHost, kAny, kNCHW, fp32);
USE_LITE_KERNEL(index_select, kHost, kAny, kNCHW, int32);
USE_LITE_KERNEL(index_select, kHost, kAny, kNCHW, int16);
USE_LITE_KERNEL(index_select, kHost, kAny, kNCHW, int8);
USE_LITE_KERNEL(cast, kHost, kAny, kNCHW, def);
USE_LITE_KERNEL(sequence_concat, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(argsort, kHost, kFloat, kAny, argsort_fp32);
USE_LITE_KERNEL(argsort, kHost, kFloat, kAny, argsort_int32);
USE_LITE_KERNEL(argsort, kHost, kFloat, kAny, argsort_int64);
USE_LITE_KERNEL(sequence_softmax, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(conv2d_transpose, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(batch_norm, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(sync_batch_norm, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(split, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(split, kHost, kFloat, kNCHW, int32);
USE_LITE_KERNEL(split, kHost, kFloat, kNCHW, int64);
USE_LITE_KERNEL(split, kHost, kInt64, kNCHW, def);
USE_LITE_KERNEL(softmax, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(search_seq_softmax, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(pad3d, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(transpose, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(transpose2, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(read_from_array, kHost, kAny, kAny, def);
USE_LITE_KERNEL(gather_nd, kHost, kAny, kAny, def);
USE_LITE_KERNEL(pool2d, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(linspace, kHost, kFloat, kAny, float32);
USE_LITE_KERNEL(linspace, kHost, kInt32, kAny, int32);
USE_LITE_KERNEL(lookup_table, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(lookup_table_v2, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(group_norm, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(max_pool2d_with_index, kHost, kFloat, kNCHW, fp32);
USE_LITE_KERNEL(crf_decoding, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(pad2d, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(match_matrix_tensor, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(tril_triu, kHost, kAny, kNCHW, float32);
USE_LITE_KERNEL(top_k, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(assign_value, kHost, kAny, kNCHW, def);
USE_LITE_KERNEL(cos, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(expand_v2, kHost, kFloat, kAny, def);
USE_LITE_KERNEL(expand_v2, kHost, kFloat, kAny, def_int32);
USE_LITE_KERNEL(expand_v2, kHost, kFloat, kAny, def_int64);
USE_LITE_KERNEL(density_prior_box, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(norm, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(p_norm, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(log_softmax, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(calib, kX86, kInt8, kNCHW, fp32_to_int8);
USE_LITE_KERNEL(calib, kX86, kInt8, kNCHW, int8_to_fp32);
USE_LITE_KERNEL(calib, kX86, kFloat, kNCHW, int32_to_fp32);
USE_LITE_KERNEL(calib, kX86, kFloat, kNCHW, fp32_to_int32);
USE_LITE_KERNEL(calib, kX86, kFloat, kNCHW, int32_to_int64);
USE_LITE_KERNEL(calib, kX86, kFloat, kNCHW, int64_to_int32);
USE_LITE_KERNEL(calib, kX86, kFloat, kNCHW, int64_to_fp32);
USE_LITE_KERNEL(calib_once, kX86, kInt8, kNCHW, fp32_to_int8);
USE_LITE_KERNEL(calib_once, kX86, kInt8, kNCHW, int8_to_fp32);
USE_LITE_KERNEL(calib_once, kX86, kFloat, kNCHW, int32_to_fp32);
USE_LITE_KERNEL(calib_once, kX86, kFloat, kNCHW, fp32_to_int32);
USE_LITE_KERNEL(calib_once, kX86, kFloat, kNCHW, int32_to_int64);
USE_LITE_KERNEL(calib_once, kX86, kFloat, kNCHW, int64_to_int32);
USE_LITE_KERNEL(calib_once, kX86, kFloat, kNCHW, int64_to_fp32);
USE_LITE_KERNEL(assign, kHost, kAny, kAny, def);
USE_LITE_KERNEL(assign, kHost, kAny, kAny, def_tensor_array);
USE_LITE_KERNEL(concat, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(concat, kX86, kFloat, kNCHW, int32);
USE_LITE_KERNEL(concat, kX86, kFloat, kNCHW, int64);
USE_LITE_KERNEL(shuffle_channel, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(while, kHost, kAny, kAny, def);
USE_LITE_KERNEL(search_seq_fc, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(sequence_conv, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(bilinear_interp, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(nearest_interp, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(bilinear_interp_v2, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(nearest_interp_v2, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(pow, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(search_fc, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(fill_constant, kHost, kAny, kNCHW, def);
USE_LITE_KERNEL(sequence_expand_as, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(sequence_expand_as, kX86, kFloat, kNCHW, int32);
USE_LITE_KERNEL(sequence_expand_as, kX86, kFloat, kNCHW, int64);
USE_LITE_KERNEL(unbind, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(unbind, kHost, kFloat, kNCHW, def_int64);
USE_LITE_KERNEL(search_grnn, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(generate_proposals_v2, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(search_aligned_mat_mul, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(is_empty, kHost, kAny, kAny, def);
USE_LITE_KERNEL(one_hot, kHost, kAny, kAny, def);
USE_LITE_KERNEL(one_hot_v2, kHost, kAny, kAny, def);
USE_LITE_KERNEL(one_hot_v2, kHost, kAny, kAny, one_hot_v2_int32);
USE_LITE_KERNEL(box_coder, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(unstack, kHost, kFloat, kAny, def);
USE_LITE_KERNEL(unstack, kHost, kFloat, kAny, unstack_int32);
USE_LITE_KERNEL(tile, kHost, kFloat, kNCHW, def_float);
USE_LITE_KERNEL(tile, kHost, kFloat, kNCHW, def_int32);
USE_LITE_KERNEL(tile, kHost, kFloat, kNCHW, def_int64);
USE_LITE_KERNEL(tile, kHost, kFloat, kNCHW, def_int8);
USE_LITE_KERNEL(tile, kHost, kFloat, kNCHW, def_bool);
USE_LITE_KERNEL(generate_proposals, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(where, kHost, kAny, kAny, def);
USE_LITE_KERNEL(sequence_topk_avg_pooling, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(dropout, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(gather, kX86, kFloat, kNCHW, int32int32);
USE_LITE_KERNEL(gather, kX86, kFloat, kNCHW, int64int64);
USE_LITE_KERNEL(gather, kX86, kFloat, kNCHW, int64int32);
USE_LITE_KERNEL(gather, kX86, kFloat, kNCHW, int32int64);
USE_LITE_KERNEL(meshgrid, kHost, kFloat, kAny, float32);
USE_LITE_KERNEL(meshgrid, kHost, kFloat, kAny, int32);
USE_LITE_KERNEL(clip, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(merge_lod_tensor, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(sequence_reverse, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(sequence_reverse, kX86, kFloat, kNCHW, int32);
USE_LITE_KERNEL(sequence_reverse, kX86, kFloat, kNCHW, int64);
USE_LITE_KERNEL(rnn, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(square, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(relu, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(leaky_relu, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(tanh, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(gelu, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(softsign, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(sigmoid, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(relu6, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(sqrt, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(rsqrt, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(mish, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(hard_swish, kX86, kFloat, kNCHW, def);