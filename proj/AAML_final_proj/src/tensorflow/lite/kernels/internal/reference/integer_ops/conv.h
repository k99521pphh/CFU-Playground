/**************************************************************************/
// FILE NAME: conv.v
// VERSRION: 1.0
// DATE: JAN 08, 2024
// AUTHOR: Kuan-Wei Chen, NYCU IEE
// CODE TYPE: CPP
// DESCRIPTION: 2024 FALL AAML / Final Project
// MODIFICATION HISTORY:
// Date                 Description
// 
/**************************************************************************/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_

#include <algorithm>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"

#include "perf.h"
#include "cfu.h"

//#include "playground_util/print_params.h"

namespace tflite {
namespace reference_integer_ops {

// Fixed-point per-channel-quantization convolution reference kernel.
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {

//  print_conv_params(params, input_shape, filter_shape, output_shape);

  // Get parameters.
  // const int32_t input_offset = params.input_offset;  // r = s(q - Z)
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t output_offset = params.output_offset;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  // const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);


  int8_t im2col[32*32][9*64]; // im2col[output_height*output_width][filter_height*filter_width*filter_input_depth]
  int8_t fr2row[9*64][64];    // fr2row[output_depth][filter_height*filter_width*filter_input_depth]

  for(int i = 0; i < 32*32; i++)
    for(int j = 0; j < 9*64; j++)
      im2col[i][j] = -128;

  int row_index = 0;
  int col_index = 0;

  // Image to Column
  for (int out_y = 0; out_y < output_height; ++out_y) {
    const int in_y_origin = (out_y * stride_height) - pad_height;
    for (int out_x = 0; out_x < output_width; ++out_x) {
      const int in_x_origin = (out_x * stride_width) - pad_width;
      for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
        const int in_y = in_y_origin + filter_y;
        for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
          const int in_x = in_x_origin + filter_x;
          // Zero padding by omitting the areas outside the image.
          const bool is_point_inside_image = (in_x >= 0) && (in_x < input_width) && (in_y >= 0) && (in_y < input_height);

          if (!is_point_inside_image) {
            continue;
          }

          for (int in_channel = 0; in_channel < filter_input_depth; ++in_channel) {
            row_index = out_y * output_width + out_x;
            col_index = filter_height * filter_width * in_channel + filter_y * filter_width + filter_x;
            im2col[row_index][col_index] = *((int8_t *)(input_data + Offset(input_shape, 0, in_y, in_x, in_channel)));
          }
        }
      }
    }
  }

  // Filter to Row
  for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
      for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
          for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              for (int in_channel = 0; in_channel < filter_input_depth; ++in_channel) {
                  row_index = filter_height * filter_width * in_channel + filter_y * filter_width + filter_x;
                  col_index = out_channel;
                  fr2row[row_index][col_index] = *((int8_t *)(filter_data + Offset(filter_shape, out_channel, filter_y, filter_x, in_channel)));
              }
          }
      }
  }

  // Reset
  int32_t cfu_out = cfu_op0(0, 0, 0);

  // Set dimension K
  int K = filter_height * filter_width * input_depth;
  cfu_out = cfu_op0(21, K, 0);

  for(int out_channel = 0; out_channel < output_depth; out_channel += 4){
    for(int kk = 0; kk < K; kk++){
      uint8_t b1 = fr2row[kk][out_channel  ];
      uint8_t b2 = fr2row[kk][out_channel+1];
      uint8_t b3 = fr2row[kk][out_channel+2];
      uint8_t b4 = fr2row[kk][out_channel+3];
      uint32_t in_b = ((b1 << 24) | (b2 << 16) | (b3 << 8) | b4);

      // Write Global Buffer B
      cfu_out = cfu_op0(2, kk, in_b);
    }


    for(int slide = 0; slide < output_height*output_width; slide += 4){
      for(int kk = 0; kk < K; kk++){
        uint8_t a1 = im2col[slide  ][kk];
        uint8_t a2 = im2col[slide+1][kk];
        uint8_t a3 = im2col[slide+2][kk];
        uint8_t a4 = im2col[slide+3][kk];
        uint32_t in_a = ((a1 << 24) | (a2 << 16) | (a3 << 8) | a4);

        // Write Global Buffer A
        cfu_out = cfu_op0(1, kk, in_a);
      }

      // Start CFU
      cfu_out = cfu_op0(3, 0, 0);

      // Wait CFU
      cfu_out = 1;
      while(cfu_out){
        cfu_out = cfu_op0(4, 0, 0);
      }

      int32_t acc[16];
      acc[0]  = (bias_data) ? cfu_op0( 5, 0, 0) + bias_data[out_channel    ] : cfu_op0( 5, 0, 0);
      acc[1]  = (bias_data) ? cfu_op0( 6, 0, 0) + bias_data[out_channel + 1] : cfu_op0( 6, 0, 0);
      acc[2]  = (bias_data) ? cfu_op0( 7, 0, 0) + bias_data[out_channel + 2] : cfu_op0( 7, 0, 0);
      acc[3]  = (bias_data) ? cfu_op0( 8, 0, 0) + bias_data[out_channel + 3] : cfu_op0( 8, 0, 0);

      acc[4]  = (bias_data) ? cfu_op0( 9, 0, 0) + bias_data[out_channel    ] : cfu_op0( 9, 0, 0);
      acc[5]  = (bias_data) ? cfu_op0(10, 0, 0) + bias_data[out_channel + 1] : cfu_op0(10, 0, 0);
      acc[6]  = (bias_data) ? cfu_op0(11, 0, 0) + bias_data[out_channel + 2] : cfu_op0(11, 0, 0);
      acc[7]  = (bias_data) ? cfu_op0(12, 0, 0) + bias_data[out_channel + 3] : cfu_op0(12, 0, 0);

      acc[8]  = (bias_data) ? cfu_op0(13, 0, 0) + bias_data[out_channel    ] : cfu_op0(13, 0, 0);
      acc[9]  = (bias_data) ? cfu_op0(14, 0, 0) + bias_data[out_channel + 1] : cfu_op0(14, 0, 0);
      acc[10] = (bias_data) ? cfu_op0(15, 0, 0) + bias_data[out_channel + 2] : cfu_op0(15, 0, 0);
      acc[11] = (bias_data) ? cfu_op0(16, 0, 0) + bias_data[out_channel + 3] : cfu_op0(16, 0, 0);

      acc[12] = (bias_data) ? cfu_op0(17, 0, 0) + bias_data[out_channel    ] : cfu_op0(17, 0, 0);
      acc[13] = (bias_data) ? cfu_op0(18, 0, 0) + bias_data[out_channel + 1] : cfu_op0(18, 0, 0);
      acc[14] = (bias_data) ? cfu_op0(19, 0, 0) + bias_data[out_channel + 2] : cfu_op0(19, 0, 0);
      acc[15] = (bias_data) ? cfu_op0(20, 0, 0) + bias_data[out_channel + 3] : cfu_op0(20, 0, 0);


      for(int i = 0; i < 16; i++){
        acc[i] = MultiplyByQuantizedMultiplier(acc[i], output_multiplier[out_channel + i%4], output_shift[out_channel + i%4]);
        acc[i] += output_offset;
        acc[i] = std::max(acc[i], output_activation_min);
        acc[i] = std::min(acc[i], output_activation_max);

        output_data[Offset(output_shape, 0, (slide + i/4)/output_height, (slide + i/4)%output_width, out_channel + i%4)] = static_cast<int8_t>(acc[i]);
      }
    }  
  }
}


inline void ConvPerChannelWithPackedInt4Weights(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_input, int8_t* unpacked_filter_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data) {
  TFLITE_DCHECK(unpacked_filter_data != nullptr);
  tflite::tensor_utils::UnpackDenseInt4IntoInt8(
      filter_input, filter_shape.FlatSize(), unpacked_filter_data);
  ConvPerChannel(params, output_multiplier, output_shift, input_shape,
                 input_data, filter_shape, unpacked_filter_data, bias_shape,
                 bias_data, output_shape, output_data);
}

// Fixed-point per-channel-quantization convolution reference kernel.
// 16-bit data and 8-bit filter
template <typename AccumScalar>
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int16_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const AccumScalar* bias_data, const RuntimeShape& output_shape,
    int16_t* output_data) {
  // Get parameters.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          auto group = out_channel / filters_per_group;
          AccumScalar acc = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;

              // Zero padding by omitting the areas outside the image.
              const bool is_point_inside_image =
                  (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                  (in_y < input_height);

              if (!is_point_inside_image) {
                continue;
              }

              for (int in_channel = 0; in_channel < filter_input_depth;
                   ++in_channel) {
                int32_t input_val =
                    input_data[Offset(input_shape, batch, in_y, in_x,
                                      in_channel + group * filter_input_depth)];
                int32_t filter_val = filter_data[Offset(
                    filter_shape, out_channel, filter_y, filter_x, in_channel)];
                // Accumulate with 64 bits accumulator.
                // int64_t += int8_t * int16_t so the highest value we can
                // get from each accumulation is [-127, 127] * ([-32768,
                // 32767] -
                // [-32768, 32767]), which is [-8322945, 8322945].
                // log2(8322945) = 22.99.
                acc += filter_val * input_val;
              }
            }
          }
          if (bias_data) {
            acc += bias_data[out_channel];
          }
          int32_t scaled_acc = MultiplyByQuantizedMultiplier(
              acc, output_multiplier[out_channel], output_shift[out_channel]);
          scaled_acc = std::max(scaled_acc, output_activation_min);
          scaled_acc = std::min(scaled_acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              static_cast<int16_t>(scaled_acc);
        }
      }
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
