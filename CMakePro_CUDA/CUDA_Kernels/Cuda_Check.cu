#include "Cuda_Check.cuh"
#include <device_launch_parameters.h>
#include <iostream>
#include <string.h>

// 核函数
__global__ void vectorAddKernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// 辅助函数：根据架构计算核心数
int _ConvertSMVerToCores(int major, int minor) {
    // 定义不同架构下每个 SM 包含的核心数
    switch (major) {
    case 7: return (minor == 0 || minor == 5) ? 64 : 64;  // Volta, Turing
    case 8: return (minor == 0) ? 64 : 128;               // Ampere (8.0=64, 8.6/8.9=128)
    case 9: return 128;                                   // Hopper
    default: return 128;
    }
}

extern "C" GpuInfo GetCudaDeviceInfo() {
    GpuInfo info;
    memset(&info, 0, sizeof(info));
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) return info;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    strncpy_s(info.name, prop.name, sizeof(info.name));
    info.computeCapabilityMajor = prop.major;
    info.computeCapabilityMinor = prop.minor;
    info.totalMem = prop.totalGlobalMem;
    info.multiProcessorCount = prop.multiProcessorCount;
    info.clockRate = prop.clockRate;
    // 计算核心数：SMs * CoresPerSM
    info.cudaCores = prop.multiProcessorCount * _ConvertSMVerToCores(prop.major, prop.minor);
    info.success = true;
    return info;
}


extern "C" bool RunCudaTest(float* h_a, float* h_b, float* h_c, int n) {
    float* d_a = nullptr, * d_b = nullptr, * d_c = nullptr;
    size_t size = n * sizeof(float);

    // 1. 分配显存
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // 2. 拷贝数据到显存
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // 3. 启动核函数 (注意：不要有空格)
    // 如果 n > 1024，建议使用多 block
    int threads = (n > 1024) ? 1024 : n;
    int blocks = (n + threads - 1) / threads;
    vectorAddKernel << <blocks, threads >> > (d_a, d_b, d_c, n);

    // 4. 同步并检查错误
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();

    // 5. 拷贝结果回内存
    if (err == cudaSuccess) {
        cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    }

    // 6. 释放资源
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return err == cudaSuccess;
}