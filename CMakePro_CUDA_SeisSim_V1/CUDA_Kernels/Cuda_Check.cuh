#pragma once
#include "Common.h"
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

//生命游戏模块函数
#ifdef __cplusplus
extern "C" {
#endif
    // 传递指针版本
    void InitCudaLife(int w, int h); // 初始化随机数状态
    void SeedCudaLife(uint8_t* d_world, int w, int h, float density);
    // 增加 decay 参数
    void UpdateLifeCuda(uint8_t* d_current, uint8_t* d_next, float* d_heatMap,
        int w, int h, float deltaTime, bool paused, float decay, int b_mask, int s_mask);
    void MousePaintCuda(uint8_t* d_world, float* d_heat, int w, int h, int mx, int my, int radius, bool erase);
    int GetPopulationCuda(uint8_t* d_world, int w, int h);
#ifdef __cplusplus
}
#endif


#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} while (0)

// 管理 GPU 所有显存资源的结构体
struct GPUSimData {
    int NX = 0, NZ = 0;
    int total_grid = 0;
    float dt = 0.0f;
    float inv_dt = 0.0f;
    int flag_type = 1;
    int npml = 0;
    int fs_idx = 0;
    bool upFlag = false;

    // 差分系数
    float c1_h = 0.0f, c2_h = 0.0f, c3_h = 0.0f, c4_h = 0.0f;

    // 1. 全局 PML 场分裂变量 (Type 1 & Type 3)
    float* d_vx_1 = nullptr; float* d_vx_2 = nullptr;
    float* d_vz_1 = nullptr; float* d_vz_2 = nullptr;
    float* d_sxx_1 = nullptr; float* d_sxx_2 = nullptr;
    float* d_szz_1 = nullptr; float* d_szz_2 = nullptr;
    float* d_sxz_1 = nullptr; float* d_sxz_2 = nullptr;

    // 2. Type 2 专用（非场分裂变量）
    float* d_vx = nullptr; float* d_vz = nullptr;
    float* d_sigmaxx = nullptr; float* d_sigmazz = nullptr; float* d_sigmaxz = nullptr;
    float* d_mu_x_flat = nullptr; float* d_lambda_x_flat = nullptr;
    float* d_lambda2mu_x_flat = nullptr; float* d_mu_z_flat = nullptr;
    float* d_rho_x_z_flat = nullptr; float* d_rho_orig_flat = nullptr;

    // 3. 基础物理参数
    float* d_rho = nullptr;
    float* d_mu = nullptr;
    float* d_lambda = nullptr;
    float* d_lambda2mu = nullptr;

    // 4. PML 边界阻尼参数
    float* d_dx = nullptr; float* d_dx_half = nullptr;
    float* d_dz = nullptr; float* d_dz_half = nullptr;
    float* d_dp_flat = nullptr; // Type 3 专用

    // 5. 震源与检波器
    float* d_wavelet = nullptr;
    int* d_rcv_grid_idx = nullptr;
};

// 接口函数
void initGPUSimulation(GPUSimData& gpu, const SimulationContext& ctx);
void freeGPUSimulation(GPUSimData& gpu);
void runGPUStep(GPUSimData& gpu, int current_it, const SimulationContext& ctx);
void copyRecordFromGPU(GPUSimData& gpu, int num_rcv, int nt, int current_it, float* h_rec_vx, float* h_rec_vz);
void copyWavefieldToHost(GPUSimData& gpu, float* h_vz_out);
void generateWavefieldTextureCUDA(GPUSimData& gpu, uchar4* d_rgba_out, float color_scale, int show_component);