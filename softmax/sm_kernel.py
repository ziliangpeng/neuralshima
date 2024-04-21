
kernel_code = """
#include <float.h>
__global__ void softmax_kernel(float *input, float *output, int n) {
    extern __shared__ float temp[];
    float max_val = -FLT_MAX;

    // Find the maximum value in the input array
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        max_val = fmaxf(max_val, input[i]);
    }

    // Reduce to find the global maximum value
    temp[threadIdx.x] = max_val;
    __syncthreads();
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            temp[threadIdx.x] = fmaxf(temp[threadIdx.x], temp[threadIdx.x + i]);
        }
        __syncthreads();
    }
    max_val = temp[0];

    // Compute the sum of the exponentials
    float sum_exp = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        sum_exp += expf(input[i] - max_val);
    }

    // Reduce to find the global sum of exponentials
    temp[threadIdx.x] = sum_exp;
    __syncthreads();
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            temp[threadIdx.x] += temp[threadIdx.x + i];
        }
        __syncthreads();
    }
    sum_exp = temp[0];

    // Compute the softmax
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        output[i] = expf(input[i] - max_val) / sum_exp;
    }
}
"""