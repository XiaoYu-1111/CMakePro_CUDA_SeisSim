#pragma once
#include <cuda_runtime.h>

// 使用 C 兼容的结构体
struct GpuInfo {
    char name[256];
    int computeCapabilityMajor;
    int computeCapabilityMinor;
    size_t totalMem;
    int multiProcessorCount; // 新增：SM 数量
    int cudaCores;           // 新增：计算出的核心数
    int clockRate;           // 新增：时钟频率
    bool success;
};

#ifdef __cplusplus
extern "C" {
#endif

    // 检查 GPU 硬件信息
    GpuInfo GetCudaDeviceInfo();

    // 运行一个简单的加法核函数测试
    bool RunCudaTest(float* h_a, float* h_b, float* h_c, int n);

#ifdef __cplusplus
}
#endif