#pragma once
#include "Common.h"
#include <cuda_runtime.h>

// =============================================================================
// 1. GPU 硬件基础信息结构体 (C 兼容)
// =============================================================================
struct GpuInfo {
    char   name[256];
    int    computeCapabilityMajor;
    int    computeCapabilityMinor;
    size_t totalMem;
    int    multiProcessorCount; // SM 数量
    int    cudaCores;           // 计算出的 CUDA 核心数
    int    clockRate;           // 时钟频率
    bool   success;
};

// =============================================================================
// 2. C 兼容接口函数声明 (CUDA 硬件检测 / 简单测试 / 生命游戏模块)
// =============================================================================
#ifdef __cplusplus
extern "C" {
#endif

    // --- 硬件环境检测与诊断 ---
    GpuInfo GetCudaDeviceInfo();
    bool    RunCudaTest(float* h_a, float* h_b, float* h_c, int n);

    // --- 生命游戏 CUDA 模块 ---
    void InitCudaLife(int w, int h); // 初始化随机数状态
    void SeedCudaLife(uint8_t* d_world, int w, int h, float density);
    void UpdateLifeCuda(uint8_t* d_current, uint8_t* d_next, float* d_heatMap,
        int w, int h, float deltaTime, bool paused, float decay, int b_mask, int s_mask);
    void MousePaintCuda(uint8_t* d_world, float* d_heat, int w, int h, int mx, int my, int radius, bool erase);
    int  GetPopulationCuda(uint8_t* d_world, int w, int h);

#ifdef __cplusplus
}
#endif

// =============================================================================
// 3. 辅助宏定义
// =============================================================================
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} while (0)

// =============================================================================
// 4. 地震波场模拟 GPU 显存资源管理结构体
// =============================================================================
struct GPUSimData {
    // 基础尺寸与控制参数
    int   NX = 0;
    int   NZ = 0;
    int   total_grid = 0;
    float dt = 0.0f;
    float inv_dt = 0.0f;
    int   flag_type = 1;
    int   npml = 0;
    int   fs_idx = 0;
    bool  upFlag = false;

    // 空间有限差分系数
    float c1_h = 0.0f;
    float c2_h = 0.0f;
    float c3_h = 0.0f;
    float c4_h = 0.0f;

    // [分类指针组 1] 全局 PML 场分裂变量 (Type 1 & Type 3)
    float* d_vx_1 = nullptr; float* d_vx_2 = nullptr;
    float* d_vz_1 = nullptr; float* d_vz_2 = nullptr;
    float* d_sxx_1 = nullptr; float* d_sxx_2 = nullptr;
    float* d_szz_1 = nullptr; float* d_szz_2 = nullptr;
    float* d_sxz_1 = nullptr; float* d_sxz_2 = nullptr;

    // [分类指针组 2] Type 2 专用（非场分裂 PML 变量）
    float* d_vx = nullptr; float* d_vz = nullptr;
    float* d_sigmaxx = nullptr; float* d_sigmazz = nullptr; float* d_sigmaxz = nullptr;
    float* d_mu_x_flat = nullptr; float* d_lambda_x_flat = nullptr;
    float* d_lambda2mu_x_flat = nullptr; float* d_mu_z_flat = nullptr;
    float* d_rho_x_z_flat = nullptr; float* d_rho_orig_flat = nullptr;

    // [分类指针组 3] 基础物理介质参数
    float* d_rho = nullptr;
    float* d_mu = nullptr;
    float* d_lambda = nullptr;
    float* d_lambda2mu = nullptr;

    // [分类指针组 4] PML 边界衰减阻尼轮廓
    float* d_dx = nullptr; float* d_dx_half = nullptr;
    float* d_dz = nullptr; float* d_dz_half = nullptr;
    float* d_dp_flat = nullptr; // Type 3 专用

    // [分类指针组 5] 震源与检波器控制
    float* d_wavelet = nullptr;
    int* d_rcv_grid_idx = nullptr;
    float* d_record_vx_step = nullptr; // 大小：num_rcv * sizeof(float)
    float* d_record_vz_step = nullptr; // 大小：num_rcv * sizeof(float)

    GPUSource* d_active_sources = nullptr; // 活跃震源的显存缓冲区
    int        src_angle = 0;
};

// =============================================================================
// 5. 地震波场 GPU 仿真计算核心接口函数
// =============================================================================
void initGPUSimulation(GPUSimData& gpu, const SimulationContext& ctx);
void freeGPUSimulation(GPUSimData& gpu);
void runGPUStep(GPUSimData& gpu, int current_it, const SimulationContext& ctx, int num_active_sources);
void copyRecordFromGPU(GPUSimData& gpu, int num_rcv, int nt, int current_it, float* h_rec_vx, float* h_rec_vz);
void copyWavefieldToHost(GPUSimData& gpu, float* h_vz_out);
void generateWavefieldTextureCUDA(GPUSimData& gpu, uchar4* d_rgba_out, float color_scale, int show_component);

// --- 【新增：专用于数据采集单步捕获的 C++ 主机端包装接口】 ---
void recordReceiverStepGPU(GPUSimData& gpu, int num_rcv, float* h_out_vx, float* h_out_vz);