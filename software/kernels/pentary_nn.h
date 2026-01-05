/**
 * @file pentary_nn.h
 * @brief Pentary Neural Network Library (PNNL)
 * 
 * This library provides highly optimized neural network primitives for the
 * Pentary Tensor Cores (PTCs), analogous to NVIDIA's cuDNN.
 * 
 * @version 1.0
 * @date 2026-01-03
 */

#ifndef PENTARY_NN_H
#define PENTARY_NN_H

#include <stdbool.h>
#include "pentary_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Types and Constants
// ============================================================================

typedef enum {
    PENTARY_NN_SUCCESS = 0,
    PENTARY_NN_ERROR_INVALID_PARAM = -1,
    PENTARY_NN_ERROR_NOT_SUPPORTED = -2,
    PENTARY_NN_ERROR_OUT_OF_MEMORY = -3,
} pentary_nn_status_t;

typedef enum {
    PENTARY_NN_ACTIVATION_RELU,
    PENTARY_NN_ACTIVATION_GELU,
    PENTARY_NN_ACTIVATION_SIGMOID,
    PENTARY_NN_ACTIVATION_TANH,
} pentary_nn_activation_t;

typedef enum {
    PENTARY_NN_POOLING_MAX,
    PENTARY_NN_POOLING_AVG,
} pentary_nn_pooling_t;

// ============================================================================
// Matrix Operations
// ============================================================================

/**
 * @brief General Matrix Multiplication (GEMM): C = alpha * A * B + beta * C
 * 
 * This operation is optimized for the Pentary Tensor Cores (PTCs) and uses
 * systolic array dataflow for maximum throughput.
 * 
 * @param device Device handle
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param alpha Scalar multiplier for A * B
 * @param A Matrix A (M x K) in device memory
 * @param lda Leading dimension of A
 * @param B Matrix B (K x N) in device memory
 * @param ldb Leading dimension of B
 * @param beta Scalar multiplier for C
 * @param C Matrix C (M x N) in device memory (input and output)
 * @param ldc Leading dimension of C
 * @param stream Stream for asynchronous execution
 * @return Status code
 */
pentary_nn_status_t pentary_nn_gemm(
    pentary_device_t device,
    int M, int N, int K,
    float alpha,
    pentary_ptr_t A, int lda,
    pentary_ptr_t B, int ldb,
    float beta,
    pentary_ptr_t C, int ldc,
    pentary_stream_t stream
);

/**
 * @brief Batched GEMM: Perform GEMM on multiple matrix pairs
 * 
 * @param device Device handle
 * @param batch_count Number of matrix pairs
 * @param M Number of rows in each A and C
 * @param N Number of columns in each B and C
 * @param K Number of columns in each A and rows in each B
 * @param alpha Scalar multiplier
 * @param A_array Array of pointers to A matrices
 * @param lda Leading dimension of A
 * @param B_array Array of pointers to B matrices
 * @param ldb Leading dimension of B
 * @param beta Scalar multiplier
 * @param C_array Array of pointers to C matrices
 * @param ldc Leading dimension of C
 * @param stream Stream for asynchronous execution
 * @return Status code
 */
pentary_nn_status_t pentary_nn_gemm_batched(
    pentary_device_t device,
    int batch_count,
    int M, int N, int K,
    float alpha,
    pentary_ptr_t* A_array, int lda,
    pentary_ptr_t* B_array, int ldb,
    float beta,
    pentary_ptr_t* C_array, int ldc,
    pentary_stream_t stream
);

// ============================================================================
// Convolution Operations
// ============================================================================

/**
 * @brief 2D Convolution: output = conv2d(input, weights) + bias
 * 
 * @param device Device handle
 * @param batch_size Number of images in the batch
 * @param in_channels Number of input channels
 * @param in_height Input height
 * @param in_width Input width
 * @param out_channels Number of output channels (filters)
 * @param kernel_height Kernel height
 * @param kernel_width Kernel width
 * @param stride_h Vertical stride
 * @param stride_w Horizontal stride
 * @param padding_h Vertical padding
 * @param padding_w Horizontal padding
 * @param input Input tensor (N x C_in x H_in x W_in)
 * @param weights Filter weights (C_out x C_in x K_h x K_w)
 * @param bias Bias vector (C_out), can be NULL
 * @param output Output tensor (N x C_out x H_out x W_out)
 * @param stream Stream for asynchronous execution
 * @return Status code
 */
pentary_nn_status_t pentary_nn_conv2d(
    pentary_device_t device,
    int batch_size,
    int in_channels, int in_height, int in_width,
    int out_channels,
    int kernel_height, int kernel_width,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    pentary_ptr_t input,
    pentary_ptr_t weights,
    pentary_ptr_t bias,
    pentary_ptr_t output,
    pentary_stream_t stream
);

// ============================================================================
// Activation Functions
// ============================================================================

/**
 * @brief Apply activation function element-wise
 * 
 * @param device Device handle
 * @param activation Activation function type
 * @param size Number of elements
 * @param input Input tensor
 * @param output Output tensor (can be same as input for in-place)
 * @param stream Stream for asynchronous execution
 * @return Status code
 */
pentary_nn_status_t pentary_nn_activation(
    pentary_device_t device,
    pentary_nn_activation_t activation,
    size_t size,
    pentary_ptr_t input,
    pentary_ptr_t output,
    pentary_stream_t stream
);

// ============================================================================
// Pooling Operations
// ============================================================================

/**
 * @brief 2D Pooling operation
 * 
 * @param device Device handle
 * @param pooling_type Max or average pooling
 * @param batch_size Number of images
 * @param channels Number of channels
 * @param in_height Input height
 * @param in_width Input width
 * @param kernel_height Pooling window height
 * @param kernel_width Pooling window width
 * @param stride_h Vertical stride
 * @param stride_w Horizontal stride
 * @param padding_h Vertical padding
 * @param padding_w Horizontal padding
 * @param input Input tensor (N x C x H_in x W_in)
 * @param output Output tensor (N x C x H_out x W_out)
 * @param stream Stream for asynchronous execution
 * @return Status code
 */
pentary_nn_status_t pentary_nn_pool2d(
    pentary_device_t device,
    pentary_nn_pooling_t pooling_type,
    int batch_size,
    int channels,
    int in_height, int in_width,
    int kernel_height, int kernel_width,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    pentary_ptr_t input,
    pentary_ptr_t output,
    pentary_stream_t stream
);

// ============================================================================
// Normalization Operations
// ============================================================================

/**
 * @brief Layer Normalization
 * 
 * @param device Device handle
 * @param batch_size Number of samples
 * @param normalized_shape Size of the normalized dimension
 * @param input Input tensor
 * @param weight Scale parameter (gamma)
 * @param bias Shift parameter (beta)
 * @param output Output tensor
 * @param eps Small constant for numerical stability
 * @param stream Stream for asynchronous execution
 * @return Status code
 */
pentary_nn_status_t pentary_nn_layer_norm(
    pentary_device_t device,
    int batch_size,
    int normalized_shape,
    pentary_ptr_t input,
    pentary_ptr_t weight,
    pentary_ptr_t bias,
    pentary_ptr_t output,
    float eps,
    pentary_stream_t stream
);

/**
 * @brief Batch Normalization
 * 
 * @param device Device handle
 * @param batch_size Number of samples
 * @param channels Number of channels
 * @param spatial_size Size of spatial dimensions (H * W)
 * @param input Input tensor (N x C x H x W)
 * @param weight Scale parameter
 * @param bias Shift parameter
 * @param running_mean Running mean (updated during training)
 * @param running_var Running variance (updated during training)
 * @param output Output tensor
 * @param eps Small constant for numerical stability
 * @param momentum Momentum for running statistics
 * @param training Training mode flag
 * @param stream Stream for asynchronous execution
 * @return Status code
 */
pentary_nn_status_t pentary_nn_batch_norm(
    pentary_device_t device,
    int batch_size,
    int channels,
    int spatial_size,
    pentary_ptr_t input,
    pentary_ptr_t weight,
    pentary_ptr_t bias,
    pentary_ptr_t running_mean,
    pentary_ptr_t running_var,
    pentary_ptr_t output,
    float eps,
    float momentum,
    bool training,
    pentary_stream_t stream
);

#ifdef __cplusplus
}
#endif

#endif // PENTARY_NN_H
