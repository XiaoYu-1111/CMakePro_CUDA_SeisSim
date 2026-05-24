#include "Cuda_Check.cuh"
#include <device_launch_parameters.h>


#include "Cuda_Check.cuh"
#include <device_launch_parameters.h>
#include <iostream>
#include <string.h>

#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

#include <thrust/execution_policy.h>

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


#include <cuda_runtime.h>
#include <curand_kernel.h>


static curandState* d_randStates = nullptr;

// 内部使用的随机数状态，只需初始化一次
static curandState* d_states = nullptr;


// --- [核心修复] 定义标准的核函数代替 Lambda ---
__global__ void kInitRand(curandState* states, unsigned long seed, int w, int h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < w * h) {
        // 每个线程初始化自己的随机数状态
        curand_init(seed, idx, 0, &states[idx]);
    }
}

extern "C" void InitCudaLife(int w, int h) {
    if (d_states) cudaFree(d_states); // 如果已存在则释放
    cudaMalloc(&d_states, w * h * sizeof(curandState));

    int threads = 256;
    int blocks = (w * h + threads - 1) / threads;
    kInitRand << <blocks, threads >> > (d_states, (unsigned long)time(0), w, h);
    cudaDeviceSynchronize();
}

// 核心：随机初始化核函数
__global__ void kSeedKernel(uint8_t* world, curandState* states, float density, int w, int h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < w * h) {
        float r = curand_uniform(&states[idx]);
        world[idx] = (r < density) ? 1 : 0;
    }
}

extern "C" void SeedCudaLife(uint8_t* d_world, int w, int h, float density) {
    int n = w * h;
    kSeedKernel << <(n + 255) / 256, 256 >> > (d_world, d_states, density, w, h);
    cudaDeviceSynchronize(); // 确保写完
}

// 核心：逻辑更新核函数///并行卷积算子。
__global__ void kLifeUpdate(const uint8_t* current, uint8_t* next, float* heatMap, int w, int h, float dt,
    bool paused, float decay, int b_mask, int s_mask) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = y * w + x;

    if (!paused) {
        int neighbors = 0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                neighbors += current[((y + dy + h) % h) * w + ((x + dx + w) % w)];
            }
        }
        uint8_t alive = current[idx];
        // 使用位运算检查规则：判断 neighbors 对应的位是否在掩码中
        if (alive) {
            next[idx] = (s_mask & (1 << neighbors)) ? 1 : 0;
        }
        else {
            next[idx] = (b_mask & (1 << neighbors)) ? 1 : 0;
        }
    }
    // 热力图计算
    float hVal = heatMap[idx];
    if (next[idx] == 1) {
        hVal = 1.0f;
    }
    else {
        // 使用传入的参数 decay。建议范围：0.5 (几乎无轨迹) 到 0.99 (长轨迹)
        hVal *= decay;
        if (hVal < 0.005f) hVal = 0.0f; // 稍微调低截断值，让过渡更平滑
    }
    heatMap[idx] = hVal;
}

extern "C" void UpdateLifeCuda(uint8_t* d_current, uint8_t* d_next, float* d_heatMap, int w, int h, float deltaTime, bool paused, float decay, int b_mask, int s_mask) {
    dim3 block(16, 16);
    dim3 grid((w + 15) / 16, (h + 15) / 16);
    kLifeUpdate << <grid, block >> > (d_current, d_next, d_heatMap, w, h, deltaTime, paused, decay, b_mask, s_mask);
}

__global__ void kMousePaint(uint8_t* world, float* heatMap, int w, int h, int mx, int my, int radius, bool erase) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float dist = sqrtf(powf(x - mx, 2) + powf(y - my, 2));
    if (dist < radius) {
        int idx = y * w + x;
        world[idx] = erase ? 0 : 1;
        if (!erase) heatMap[idx] = 1.0f;
    }
}

extern "C" void MousePaintCuda(uint8_t* d_world, float* d_heat, int w, int h, int mx, int my, int radius, bool erase) {
    dim3 block(16, 16);
    dim3 grid((w + 15) / 16, (h + 15) / 16);
    kMousePaint << <grid, block >> > (d_world, d_heat, w, h, mx, my, radius, erase);
}

extern "C" int GetPopulationCuda(uint8_t* d_world, int w, int h) {
    // 将原始显存指针包装为 thrust 迭代器
    thrust::device_ptr<uint8_t> ptr(d_world);
    // 执行规约求和 (1 代表活，0 代表死，求和即为总人口)
    // 使用 thrust::plus<int>() 进行累加
    try {
        return thrust::reduce(thrust::device, ptr, ptr + (w * h), (int)0, thrust::plus<int>());
    }
    catch (...) {
        return 0; // 防止异常
    }
}


// =============================================================================
// CUDA 核心核函数
// =============================================================================

// --- Type 1 PML Stress Update ---
__global__ void update_type1_stress_kernel(GPUSimData g) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; // X
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Z

    if (i >= 4 && i < g.NZ - 4 && j >= 4 && j < g.NX - 4) {
        int k = i * g.NX + j;
        float inv_dt = g.inv_dt;

        float d_z = g.d_dz[i];
        float d_z_h = g.d_dz_half[i];
        float d_x = g.d_dx[j];
        float d_x_h = g.d_dx_half[j];

        float dvx_dx = g.c1_h * (g.d_vx_1[k + 1] + g.d_vx_2[k + 1] - g.d_vx_1[k] - g.d_vx_2[k])
            + g.c2_h * (g.d_vx_1[k + 2] + g.d_vx_2[k + 2] - g.d_vx_1[k - 1] - g.d_vx_2[k - 1])
            + g.c3_h * (g.d_vx_1[k + 3] + g.d_vx_2[k + 3] - g.d_vx_1[k - 2] - g.d_vx_2[k - 2])
            + g.c4_h * (g.d_vx_1[k + 4] + g.d_vx_2[k + 4] - g.d_vx_1[k - 3] - g.d_vx_2[k - 3]);

        float dvz_dz = g.c1_h * (g.d_vz_1[k] + g.d_vz_2[k] - g.d_vz_1[k - g.NX] - g.d_vz_2[k - g.NX])
            + g.c2_h * (g.d_vz_1[k + g.NX] + g.d_vz_2[k + g.NX] - g.d_vz_1[k - 2 * g.NX] - g.d_vz_2[k - 2 * g.NX])
            + g.c3_h * (g.d_vz_1[k + 2 * g.NX] + g.d_vz_2[k + 2 * g.NX] - g.d_vz_1[k - 3 * g.NX] - g.d_vz_2[k - 3 * g.NX])
            + g.c4_h * (g.d_vz_1[k + 3 * g.NX] + g.d_vz_2[k + 3 * g.NX] - g.d_vz_1[k - 4 * g.NX] - g.d_vz_2[k - 4 * g.NX]);

        g.d_sxx_1[k] = (g.d_sxx_1[k] * (inv_dt - d_x_h) + 0.5f * (g.d_lambda2mu[k] + g.d_lambda2mu[k + 1]) * dvx_dx) / (inv_dt + d_x_h);
        g.d_sxx_2[k] = (g.d_sxx_2[k] * (inv_dt - d_z) + 0.5f * (g.d_lambda[k] + g.d_lambda[k + 1]) * dvz_dz) / (inv_dt + d_z);
        g.d_szz_1[k] = (g.d_szz_1[k] * (inv_dt - d_x_h) + 0.5f * (g.d_lambda[k] + g.d_lambda[k + 1]) * dvx_dx) / (inv_dt + d_x_h);
        g.d_szz_2[k] = (g.d_szz_2[k] * (inv_dt - d_z) + 0.5f * (g.d_lambda2mu[k] + g.d_lambda2mu[k + 1]) * dvz_dz) / (inv_dt + d_z);

        float dvz_dx = g.c1_h * (g.d_vz_1[k] + g.d_vz_2[k] - g.d_vz_1[k - 1] - g.d_vz_2[k - 1])
            + g.c2_h * (g.d_vz_1[k + 1] + g.d_vz_2[k + 1] - g.d_vz_1[k - 2] - g.d_vz_2[k - 2])
            + g.c3_h * (g.d_vz_1[k + 2] + g.d_vz_2[k + 2] - g.d_vz_1[k - 3] - g.d_vz_2[k - 3])
            + g.c4_h * (g.d_vz_1[k + 3] + g.d_vz_2[k + 3] - g.d_vz_1[k - 4] - g.d_vz_2[k - 4]);

        float dvx_dz = g.c1_h * (g.d_vx_1[k + g.NX] + g.d_vx_2[k + g.NX] - g.d_vx_1[k] - g.d_vx_2[k])
            + g.c2_h * (g.d_vx_1[k + 2 * g.NX] + g.d_vx_2[k + 2 * g.NX] - g.d_vx_1[k - g.NX] - g.d_vx_2[k - g.NX])
            + g.c3_h * (g.d_vx_1[k + 3 * g.NX] + g.d_vx_2[k + 3 * g.NX] - g.d_vx_1[k - 2 * g.NX] - g.d_vx_2[k - 2 * g.NX])
            + g.c4_h * (g.d_vx_1[k + 4 * g.NX] + g.d_vx_2[k + 4 * g.NX] - g.d_vx_1[k - 3 * g.NX] - g.d_vx_2[k - 3 * g.NX]);

        g.d_sxz_1[k] = (g.d_sxz_1[k] * (inv_dt - d_x) + 0.5f * (g.d_mu[k] + g.d_mu[k + g.NX]) * dvz_dx) / (inv_dt + d_x);
        g.d_sxz_2[k] = (g.d_sxz_2[k] * (inv_dt - d_z_h) + 0.5f * (g.d_mu[k] + g.d_mu[k + g.NX]) * dvx_dz) / (inv_dt + d_z_h);
    }
}

// --- Type 1 PML Velocity Update ---
__global__ void update_type1_velocity_kernel(GPUSimData g) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 4 && i < g.NZ - 4 && j >= 4 && j < g.NX - 4) {
        int k = i * g.NX + j;
        float inv_dt = g.inv_dt;

        float d_z = g.d_dz[i];
        float d_z_h = g.d_dz_half[i];
        float d_x = g.d_dx[j];
        float d_x_h = g.d_dx_half[j];
        float cur_rho = g.d_rho[k];

        float dsigxx_dx = g.c1_h * (g.d_sxx_1[k] + g.d_sxx_2[k] - g.d_sxx_1[k - 1] - g.d_sxx_2[k - 1])
            + g.c2_h * (g.d_sxx_1[k + 1] + g.d_sxx_2[k + 1] - g.d_sxx_1[k - 2] - g.d_sxx_2[k - 2])
            + g.c3_h * (g.d_sxx_1[k + 2] + g.d_sxx_2[k + 2] - g.d_sxx_1[k - 3] - g.d_sxx_2[k - 3])
            + g.c4_h * (g.d_sxx_1[k + 3] + g.d_sxx_2[k + 3] - g.d_sxx_1[k - 4] - g.d_sxx_2[k - 4]);

        float dsigxz_dz = g.c1_h * (g.d_sxz_1[k] + g.d_sxz_2[k] - g.d_sxz_1[k - g.NX] - g.d_sxz_2[k - g.NX])
            + g.c2_h * (g.d_sxz_1[k + g.NX] + g.d_sxz_2[k + g.NX] - g.d_sxz_1[k - 2 * g.NX] - g.d_sxz_2[k - 2 * g.NX])
            + g.c3_h * (g.d_sxz_1[k + 2 * g.NX] + g.d_sxz_2[k + 2 * g.NX] - g.d_sxz_1[k - 3 * g.NX] - g.d_sxz_2[k - 3 * g.NX])
            + g.c4_h * (g.d_sxz_1[k + 3 * g.NX] + g.d_sxz_2[k + 3 * g.NX] - g.d_sxz_1[k - 4 * g.NX] - g.d_sxz_2[k - 4 * g.NX]);

        g.d_vx_1[k] = (g.d_vx_1[k] * (inv_dt - d_x) + dsigxx_dx / cur_rho) / (inv_dt + d_x);
        g.d_vx_2[k] = (g.d_vx_2[k] * (inv_dt - d_z) + dsigxz_dz / cur_rho) / (inv_dt + d_z);

        float dsigxz_dx = g.c1_h * (g.d_sxz_1[k + 1] + g.d_sxz_2[k + 1] - g.d_sxz_1[k] - g.d_sxz_2[k])
            + g.c2_h * (g.d_sxz_1[k + 2] + g.d_sxz_2[k + 2] - g.d_sxz_1[k - 1] - g.d_sxz_2[k - 1])
            + g.c3_h * (g.d_sxz_1[k + 3] + g.d_sxz_2[k + 3] - g.d_sxz_1[k - 2] - g.d_sxz_2[k - 2])
            + g.c4_h * (g.d_sxz_1[k + 4] + g.d_sxz_2[k + 4] - g.d_sxz_1[k - 3] - g.d_sxz_2[k - 3]);

        float dsigzz_dz = g.c1_h * (g.d_szz_1[k + g.NX] + g.d_szz_2[k + g.NX] - g.d_szz_1[k] - g.d_szz_2[k])
            + g.c2_h * (g.d_szz_1[k + 2 * g.NX] + g.d_szz_2[k + 2 * g.NX] - g.d_szz_1[k - g.NX] - g.d_szz_2[k - g.NX])
            + g.c3_h * (g.d_szz_1[k + 3 * g.NX] + g.d_szz_2[k + 3 * g.NX] - g.d_szz_1[k - 2 * g.NX] - g.d_szz_2[k - 2 * g.NX])
            + g.c4_h * (g.d_szz_1[k + 4 * g.NX] + g.d_szz_2[k + 4 * g.NX] - g.d_szz_1[k - 3 * g.NX] - g.d_szz_2[k - 3 * g.NX]);

        float rho_avg = 0.25f * (g.d_rho[k] + g.d_rho[k + 1] + g.d_rho[k + g.NX] + g.d_rho[k + g.NX + 1]);
        g.d_vz_1[k] = (g.d_vz_1[k] * (inv_dt - d_x_h) + dsigxz_dx / rho_avg) / (inv_dt + d_x_h);
        g.d_vz_2[k] = (g.d_vz_2[k] * (inv_dt - d_z_h) + dsigzz_dz / rho_avg) / (inv_dt + d_z_h);
    }
}

// --- Type 2 PML/FDM Hybrid Stress Update ---
__global__ void update_type2_stress_kernel(GPUSimData g) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 4 && i < g.NZ - 4 && j >= 4 && j < g.NX - 4) {
        int k = i * g.NX + j;
        int row_dn1 = (i + 1) * g.NX; int row_dn2 = (i + 2) * g.NX; int row_dn3 = (i + 3) * g.NX; int row_dn4 = (i + 4) * g.NX;
        int row_up1 = (i - 1) * g.NX; int row_up2 = (i - 2) * g.NX; int row_up3 = (i - 3) * g.NX; int row_up4 = (i - 4) * g.NX;

        float dvz_dx = g.c1_h * (g.d_vz[k] - g.d_vz[k - 1]) + g.c2_h * (g.d_vz[k + 1] - g.d_vz[k - 2])
            + g.c3_h * (g.d_vz[k + 2] - g.d_vz[k - 3]) + g.c4_h * (g.d_vz[k + 3] - g.d_vz[k - 4]);

        float dvx_dz = g.c1_h * (g.d_vx[row_dn1 + j] - g.d_vx[k]) + g.c2_h * (g.d_vx[row_dn2 + j] - g.d_vx[row_up1 + j])
            + g.c3_h * (g.d_vx[row_dn3 + j] - g.d_vx[row_up2 + j]) + g.c4_h * (g.d_vx[row_dn4 + j] - g.d_vx[row_up3 + j]);

        float dvx_dx = g.c1_h * (g.d_vx[k + 1] - g.d_vx[k]) + g.c2_h * (g.d_vx[k + 2] - g.d_vx[k - 1])
            + g.c3_h * (g.d_vx[k + 3] - g.d_vx[k - 2]) + g.c4_h * (g.d_vx[k + 4] - g.d_vx[k - 3]);

        float dvz_dz = g.c1_h * (g.d_vz[k] - g.d_vz[row_up1 + j]) + g.c2_h * (g.d_vz[row_dn1 + j] - g.d_vz[row_up2 + j])
            + g.c3_h * (g.d_vz[row_dn2 + j] - g.d_vz[row_up3 + j]) + g.c4_h * (g.d_vz[row_dn3 + j] - g.d_vz[row_up4 + j]);

        bool is_pml = (i < g.npml || i >= g.NZ - g.npml || j < g.npml || j >= g.NX - g.npml);

        if (!is_pml) {
            g.d_sigmaxz[k] += g.dt * g.d_mu_z_flat[k] * (dvz_dx + dvx_dz);
            g.d_sigmaxx[k] += g.dt * (g.d_lambda2mu_x_flat[k] * dvx_dx + g.d_lambda_x_flat[k] * dvz_dz);
            g.d_sigmazz[k] += g.dt * (g.d_lambda_x_flat[k] * dvx_dx + g.d_lambda2mu_x_flat[k] * dvz_dz);
        }
        else {
            float inv_dt = g.inv_dt;
            float d_z_val = g.d_dz[i];
            float d_z_half_val = g.d_dz_half[i];
            float d_x_val = g.d_dx[j];
            float d_x_half_val = g.d_dx_half[j];

            g.d_sxz_1[k] = (g.d_sxz_1[k] * (inv_dt - d_x_val) + g.d_mu_z_flat[k] * dvz_dx) / (inv_dt + d_x_val);
            g.d_sxz_2[k] = (g.d_sxz_2[k] * (inv_dt - d_z_half_val) + g.d_mu_z_flat[k] * dvx_dz) / (inv_dt + d_z_half_val);

            g.d_sxx_1[k] = (g.d_sxx_1[k] * (inv_dt - d_x_half_val) + g.d_lambda2mu_x_flat[k] * dvx_dx) / (inv_dt + d_x_half_val);
            g.d_szz_1[k] = (g.d_szz_1[k] * (inv_dt - d_x_half_val) + g.d_lambda_x_flat[k] * dvx_dx) / (inv_dt + d_x_half_val);

            g.d_sxx_2[k] = (g.d_sxx_2[k] * (inv_dt - d_z_val) + g.d_lambda_x_flat[k] * dvz_dz) / (inv_dt + d_z_val);
            g.d_szz_2[k] = (g.d_szz_2[k] * (inv_dt - d_z_val) + g.d_lambda2mu_x_flat[k] * dvz_dz) / (inv_dt + d_z_val);

            g.d_sigmaxz[k] = g.d_sxz_1[k] + g.d_sxz_2[k];
            g.d_sigmaxx[k] = g.d_sxx_1[k] + g.d_sxx_2[k];
            g.d_sigmazz[k] = g.d_szz_1[k] + g.d_szz_2[k];
        }
    }
}

// --- Type 2 PML/FDM Hybrid Velocity Update ---
__global__ void update_type2_velocity_kernel(GPUSimData g) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 4 && i < g.NZ - 4 && j >= 4 && j < g.NX - 4) {
        int k = i * g.NX + j;
        int row_dn1 = (i + 1) * g.NX; int row_dn2 = (i + 2) * g.NX; int row_dn3 = (i + 3) * g.NX; int row_dn4 = (i + 4) * g.NX;
        int row_up1 = (i - 1) * g.NX; int row_up2 = (i - 2) * g.NX; int row_up3 = (i - 3) * g.NX; int row_up4 = (i - 4) * g.NX;

        float dsigxz_dx = g.c1_h * (g.d_sigmaxz[k + 1] - g.d_sigmaxz[k]) + g.c2_h * (g.d_sigmaxz[k + 2] - g.d_sigmaxz[k - 1])
            + g.c3_h * (g.d_sigmaxz[k + 3] - g.d_sigmaxz[k - 2]) + g.c4_h * (g.d_sigmaxz[k + 4] - g.d_sigmaxz[k - 3]);

        float dsigzz_dz = g.c1_h * (g.d_sigmazz[row_dn1 + j] - g.d_sigmazz[k]) + g.c2_h * (g.d_sigmazz[row_dn2 + j] - g.d_sigmazz[row_up1 + j])
            + g.c3_h * (g.d_sigmazz[row_dn3 + j] - g.d_sigmazz[row_up2 + j]) + g.c4_h * (g.d_sigmazz[row_dn4 + j] - g.d_sigmazz[row_up3 + j]);

        float dsigxx_dx = g.c1_h * (g.d_sigmaxx[k] - g.d_sigmaxx[k - 1]) + g.c2_h * (g.d_sigmaxx[k + 1] - g.d_sigmaxx[k - 2])
            + g.c3_h * (g.d_sigmaxx[k + 2] - g.d_sigmaxx[k - 3]) + g.c4_h * (g.d_sigmaxx[k + 3] - g.d_sigmaxx[k - 4]);

        float dsigxz_dz = g.c1_h * (g.d_sigmaxz[k] - g.d_sigmaxz[row_up1 + j]) + g.c2_h * (g.d_sigmaxz[row_dn1 + j] - g.d_sigmaxz[row_up2 + j])
            + g.c3_h * (g.d_sigmaxz[row_dn2 + j] - g.d_sigmaxz[row_up3 + j]) + g.c4_h * (g.d_sigmaxz[row_dn3 + j] - g.d_sigmaxz[row_up4 + j]);

        bool is_pml = (i < g.npml || i >= g.NZ - g.npml || j < g.npml || j >= g.NX - g.npml);

        if (!is_pml) {
            if (g.d_rho_x_z_flat[k] > 1000.0f) {
                g.d_vz[k] += g.dt * (dsigxz_dx + dsigzz_dz) / g.d_rho_x_z_flat[k];
            }
            g.d_vx[k] += g.dt * (dsigxx_dx + dsigxz_dz) / g.d_rho_orig_flat[k];
        }
        else {
            float inv_dt = g.inv_dt;
            float d_z_val = g.d_dz[i];
            float d_z_half_val = g.d_dz_half[i];
            float d_x_val = g.d_dx[j];
            float d_x_half_val = g.d_dx_half[j];

            g.d_vz_1[k] = (g.d_vz_1[k] * (inv_dt - d_x_half_val) + dsigxz_dx / g.d_rho_x_z_flat[k]) / (inv_dt + d_x_half_val);
            g.d_vz_2[k] = (g.d_vz_2[k] * (inv_dt - d_z_half_val) + dsigzz_dz / g.d_rho_x_z_flat[k]) / (inv_dt + d_z_half_val);

            g.d_vx_1[k] = (g.d_vx_1[k] * (inv_dt - d_x_val) + dsigxx_dx / g.d_rho_orig_flat[k]) / (inv_dt + d_x_val);
            g.d_vx_2[k] = (g.d_vx_2[k] * (inv_dt - d_z_val) + dsigxz_dz / g.d_rho_orig_flat[k]) / (inv_dt + d_z_val);

            g.d_vx[k] = g.d_vx_1[k] + g.d_vx_2[k];
            g.d_vz[k] = g.d_vz_1[k] + g.d_vz_2[k];
        }
    }
}

// --- Type 3 Free Surface Stress Update ---
__global__ void update_type3_stress_kernel(GPUSimData g) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= g.fs_idx && i < g.NZ - 4 && j >= 4 && j < g.NX - 4) {
        int k = i * g.NX + j;
        int row_dn1 = (i + 1) * g.NX; int row_dn2 = (i + 2) * g.NX;
        int row_up1 = (i - 1) * g.NX; int row_up2 = (i - 2) * g.NX; int row_up3 = (i - 3) * g.NX; int row_up4 = (i - 4) * g.NX;

        float d_z_val = (i <= g.fs_idx + 5 && g.upFlag) ? 0.0f : g.d_dz[i];
        float d_z_half_val = (i <= g.fs_idx + 5 && g.upFlag) ? 0.0f : g.d_dz_half[i];
        float d_x_val = g.d_dx[j];
        float d_x_half_val = g.d_dx_half[j];
        float inv_dt = g.inv_dt;

        if (i == g.fs_idx) {
            // 自由边界条件
            g.d_szz_1[k] = 0.0f; g.d_szz_2[k] = 0.0f;

            float dvx_dx = g.c1_h * (g.d_vx_1[k + 1] + g.d_vx_2[k + 1] - g.d_vx_1[k] - g.d_vx_2[k])
                + g.c2_h * (g.d_vx_1[k + 2] + g.d_vx_2[k + 2] - g.d_vx_1[k - 1] - g.d_vx_2[k - 1])
                + g.c3_h * (g.d_vx_1[k + 3] + g.d_vx_2[k + 3] - g.d_vx_1[k - 2] - g.d_vx_2[k - 2])
                + g.c4_h * (g.d_vx_1[k + 4] + g.d_vx_2[k + 4] - g.d_vx_1[k - 3] - g.d_vx_2[k - 3]);

            g.d_sxx_1[k] = g.d_sxx_1[k] + g.dt * g.d_dp_flat[j] * dvx_dx;
            g.d_sxx_2[k] = 0.0f;

            float dvz_dx = g.c1_h * (g.d_vz_1[k] + g.d_vz_2[k] - g.d_vz_1[k - 1] - g.d_vz_2[k - 1])
                + g.c2_h * (g.d_vz_1[k + 1] + g.d_vz_2[k + 1] - g.d_vz_1[k - 2] - g.d_vz_2[k - 2]);

            float dvx_dz = g.c1_h * (g.d_vx_1[row_dn1 + j] + g.d_vx_2[row_dn1 + j] - g.d_vx_1[k] - g.d_vx_2[k])
                + g.c2_h * (g.d_vx_1[row_dn2 + j] + g.d_vx_2[row_dn2 + j] - g.d_vx_1[row_up1 + j] - g.d_vx_2[row_up1 + j]);

            g.d_sxz_1[k] = (g.d_sxz_1[k] * (inv_dt - d_x_val) + 0.5f * (g.d_mu[k] + g.d_mu[row_dn1 + j]) * dvz_dx) / (inv_dt + d_x_val);
            g.d_sxz_2[k] = (g.d_sxz_2[k] * (inv_dt - d_z_half_val) + 0.5f * (g.d_mu[k] + g.d_mu[row_dn1 + j]) * dvx_dz) / (inv_dt + d_z_half_val);
        }
        else {
            // 内域 PML
            float dvx_dx = g.c1_h * (g.d_vx_1[k + 1] + g.d_vx_2[k + 1] - g.d_vx_1[k] - g.d_vx_2[k])
                + g.c2_h * (g.d_vx_1[k + 2] + g.d_vx_2[k + 2] - g.d_vx_1[k - 1] - g.d_vx_2[k - 1])
                + g.c3_h * (g.d_vx_1[k + 3] + g.d_vx_2[k + 3] - g.d_vx_1[k - 2] - g.d_vx_2[k - 2])
                + g.c4_h * (g.d_vx_1[k + 4] + g.d_vx_2[k + 4] - g.d_vx_1[k - 3] - g.d_vx_2[k - 3]);

            float dvz_dz = g.c1_h * (g.d_vz_1[k] + g.d_vz_2[k] - g.d_vz_1[row_up1 + j] - g.d_vz_2[row_up1 + j])
                + g.c2_h * (g.d_vz_1[row_dn1 + j] + g.d_vz_2[row_dn1 + j] - g.d_vz_1[row_up2 + j] - g.d_vz_2[row_up2 + j])
                + g.c3_h * (g.d_vz_1[row_dn1 + g.NX + j] + g.d_vz_2[row_dn1 + g.NX + j] - g.d_vz_1[row_up3 + j] - g.d_vz_2[row_up3 + j])
                + g.c4_h * (g.d_vz_1[row_dn1 + 2 * g.NX + j] + g.d_vz_2[row_dn1 + 2 * g.NX + j] - g.d_vz_1[row_up4 + j] - g.d_vz_2[row_up4 + j]);

            g.d_sxx_1[k] = (g.d_sxx_1[k] * (inv_dt - d_x_half_val) + 0.5f * (g.d_lambda2mu[k] + g.d_lambda2mu[k + 1]) * dvx_dx) / (inv_dt + d_x_half_val);
            g.d_sxx_2[k] = (g.d_sxx_2[k] * (inv_dt - d_z_val) + 0.5f * (g.d_lambda[k] + g.d_lambda[k + 1]) * dvz_dz) / (inv_dt + d_z_val);

            g.d_szz_1[k] = (g.d_szz_1[k] * (inv_dt - d_x_half_val) + 0.5f * (g.d_lambda[k] + g.d_lambda[k + 1]) * dvx_dx) / (inv_dt + d_x_half_val);
            g.d_szz_2[k] = (g.d_szz_2[k] * (inv_dt - d_z_val) + 0.5f * (g.d_lambda2mu[k] + g.d_lambda2mu[k + 1]) * dvz_dz) / (inv_dt + d_z_val);

            float dvz_dx = g.c1_h * (g.d_vz_1[k] + g.d_vz_2[k] - g.d_vz_1[k - 1] - g.d_vz_2[k - 1])
                + g.c2_h * (g.d_vz_1[k + 1] + g.d_vz_2[k + 1] - g.d_vz_1[k - 2] - g.d_vz_2[k - 2])
                + g.c3_h * (g.d_vz_1[k + 2] + g.d_vz_2[k + 2] - g.d_vz_1[k - 3] - g.d_vz_2[k - 3])
                + g.c4_h * (g.d_vz_1[k + 3] + g.d_vz_2[k + 3] - g.d_vz_1[k - 4] - g.d_vz_2[k - 4]);

            float dvx_dz = g.c1_h * (g.d_vx_1[row_dn1 + j] + g.d_vx_2[row_dn1 + j] - g.d_vx_1[k] - g.d_vx_2[k])
                + g.c2_h * (g.d_vx_1[row_dn2 + j] + g.d_vx_2[row_dn2 + j] - g.d_vx_1[row_up1 + j] - g.d_vx_2[row_up1 + j])
                + g.c3_h * (g.d_vx_1[row_dn1 + g.NX + j] + g.d_vx_2[row_dn1 + g.NX + j] - g.d_vx_1[row_up2 + j] - g.d_vx_2[row_up2 + j])
                + g.c4_h * (g.d_vx_1[row_dn1 + 2 * g.NX + j] + g.d_vx_2[row_dn1 + 2 * g.NX + j] - g.d_vx_1[row_up3 + j] - g.d_vx_2[row_up3 + j]);

            g.d_sxz_1[k] = (g.d_sxz_1[k] * (inv_dt - d_x_val) + 0.5f * (g.d_mu[k] + g.d_mu[row_dn1 + j]) * dvz_dx) / (inv_dt + d_x_val);
            g.d_sxz_2[k] = (g.d_sxz_2[k] * (inv_dt - d_z_half_val) + 0.5f * (g.d_mu[k] + g.d_mu[row_dn1 + j]) * dvx_dz) / (inv_dt + d_z_half_val);
        }
    }
}

// --- Type 3 Stress Mirroring (szz/sxz Antisymmetric imaging) ---
__global__ void type3_stress_mirror_kernel(GPUSimData g) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= 4 && j < g.NX - 4) {
        for (int l = 1; l <= 4; ++l) {
            int idx_above = (g.fs_idx - l) * g.NX + j;
            int idx_below = (g.fs_idx + l) * g.NX + j;

            if (idx_above >= 0 && idx_below < g.total_grid) {
                g.d_szz_1[idx_above] = -g.d_szz_1[idx_below];
                g.d_szz_2[idx_above] = -g.d_szz_2[idx_below];

                int idx_below_shear = (g.fs_idx + l - 1) * g.NX + j;
                g.d_sxz_1[idx_above] = -g.d_sxz_1[idx_below_shear];
                g.d_sxz_2[idx_above] = -g.d_sxz_2[idx_below_shear];
            }
        }
    }
}

// --- Type 3 Free Surface Velocity Update ---
__global__ void update_type3_velocity_kernel(GPUSimData g) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 4 && i < g.NZ - 4 && j >= 4 && j < g.NX - 4) {
        int k = i * g.NX + j;

        if (i < g.fs_idx) {
            // 空气层速度设为零
            g.d_vx_1[k] = 0.0f; g.d_vx_2[k] = 0.0f;
            g.d_vz_1[k] = 0.0f; g.d_vz_2[k] = 0.0f;
        }
        else {
            int row_dn1 = (i + 1) * g.NX; int row_dn2 = (i + 2) * g.NX;
            int row_up1 = (i - 1) * g.NX; int row_up2 = (i - 2) * g.NX; int row_up3 = (i - 3) * g.NX; int row_up4 = (i - 4) * g.NX;

            float d_z_val = (i <= g.fs_idx + 5 && g.upFlag) ? 0.0f : g.d_dz[i];
            float d_z_half_val = (i <= g.fs_idx + 5 && g.upFlag) ? 0.0f : g.d_dz_half[i];
            float d_x_val = g.d_dx[j];
            float d_x_half_val = g.d_dx_half[j];
            float cur_rho = g.d_rho[k];
            float inv_dt = g.inv_dt;

            float dsigxx_dx = g.c1_h * (g.d_sxx_1[k] + g.d_sxx_2[k] - g.d_sxx_1[k - 1] - g.d_sxx_2[k - 1])
                + g.c2_h * (g.d_sxx_1[k + 1] + g.d_sxx_2[k + 1] - g.d_sxx_1[k - 2] - g.d_sxx_2[k - 2])
                + g.c3_h * (g.d_sxx_1[k + 2] + g.d_sxx_2[k + 2] - g.d_sxx_1[k - 3] - g.d_sxx_2[k - 3])
                + g.c4_h * (g.d_sxx_1[k + 3] + g.d_sxx_2[k + 3] - g.d_sxx_1[k - 4] - g.d_sxx_2[k - 4]);

            float dsigxz_dz = g.c1_h * (g.d_sxz_1[k] + g.d_sxz_2[k] - g.d_sxz_1[row_up1 + j] - g.d_sxz_2[row_up1 + j])
                + g.c2_h * (g.d_sxz_1[row_dn1 + j] + g.d_sxz_2[row_dn1 + j] - g.d_sxz_1[row_up2 + j] - g.d_sxz_2[row_up2 + j])
                + g.c3_h * (g.d_sxz_1[row_dn1 + g.NX + j] + g.d_sxz_2[row_dn1 + g.NX + j] - g.d_sxz_1[row_up3 + j] - g.d_sxz_2[row_up3 + j])
                + g.c4_h * (g.d_sxz_1[row_dn1 + 2 * g.NX + j] + g.d_sxz_2[row_dn1 + 2 * g.NX + j] - g.d_sxz_1[row_up4 + j] - g.d_sxz_2[row_up4 + j]);

            g.d_vx_1[k] = (g.d_vx_1[k] * (inv_dt - d_x_val) + dsigxx_dx / cur_rho) / (inv_dt + d_x_val);
            g.d_vx_2[k] = (g.d_vx_2[k] * (inv_dt - d_z_val) + dsigxz_dz / cur_rho) / (inv_dt + d_z_val);

            float dsigxz_dx = g.c1_h * (g.d_sxz_1[k + 1] + g.d_sxz_2[k + 1] - g.d_sxz_1[k] - g.d_sxz_2[k])
                + g.c2_h * (g.d_sxz_1[k + 2] + g.d_sxz_2[k + 2] - g.d_sxz_1[k - 1] - g.d_sxz_2[k - 1])
                + g.c3_h * (g.d_sxz_1[k + 3] + g.d_sxz_2[k + 3] - g.d_sxz_1[k - 2] - g.d_sxz_2[k - 2])
                + g.c4_h * (g.d_sxz_1[k + 4] + g.d_sxz_2[k + 4] - g.d_sxz_1[k - 3] - g.d_sxz_2[k - 3]);

            float dsigzz_dz = g.c1_h * (g.d_szz_1[row_dn1 + j] + g.d_szz_2[row_dn1 + j] - g.d_szz_1[k] - g.d_szz_2[k])
                + g.c2_h * (g.d_szz_1[row_dn2 + j] + g.d_szz_2[row_dn2 + j] - g.d_szz_1[row_up1 + j] - g.d_szz_2[row_up1 + j])
                + g.c3_h * (g.d_szz_1[row_dn1 + g.NX + j] + g.d_szz_2[row_dn1 + g.NX + j] - g.d_szz_1[row_up2 + j] - g.d_szz_2[row_up2 + j])
                + g.c4_h * (g.d_szz_1[row_dn1 + 2 * g.NX + j] + g.d_szz_2[row_dn1 + 2 * g.NX + j] - g.d_szz_1[row_up3 + j] - g.d_szz_2[row_up3 + j]);

            float rho_avg = 0.25f * (g.d_rho[k] + g.d_rho[k + 1] + g.d_rho[row_dn1 + j] + g.d_rho[row_dn1 + j + 1]);
            g.d_vz_1[k] = (g.d_vz_1[k] * (inv_dt - d_x_half_val) + dsigxz_dx / rho_avg) / (inv_dt + d_x_half_val);
            g.d_vz_2[k] = (g.d_vz_2[k] * (inv_dt - d_z_half_val) + dsigzz_dz / rho_avg) / (inv_dt + d_z_half_val);
        }
    }
}

// --- Source Injection ---
__global__ void inject_source_kernel(GPUSimData g, float force_x, float force_z, int src_idx) {
    if (g.flag_type == 1 || g.flag_type == 3) {
        g.d_vx_1[src_idx] += force_x * g.dt / g.d_rho[src_idx];
        float rho_avg = 0.25f * (g.d_rho[src_idx] + g.d_rho[src_idx + 1] + g.d_rho[src_idx + g.NX] + g.d_rho[src_idx + g.NX + 1]);
        g.d_vz_1[src_idx] += force_z * g.dt / rho_avg;
    }
    else if (g.flag_type == 2) {
        g.d_vx[src_idx] += force_x * g.dt / g.d_rho_orig_flat[src_idx];
        g.d_vz[src_idx] += force_z * g.dt / g.d_rho_x_z_flat[src_idx];
    }
}

// --- Zero Dirichlet Boundary ---
__global__ void zero_boundary_kernel(GPUSimData g) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < g.NX) {
        int top = idx;
        int btm = (g.NZ - 1) * g.NX + idx;
        if (g.flag_type == 1 || g.flag_type == 3) {
            g.d_vx_1[top] = 0; g.d_vx_2[top] = 0; g.d_vz_1[top] = 0; g.d_vz_2[top] = 0;
            g.d_vx_1[btm] = 0; g.d_vx_2[btm] = 0; g.d_vz_1[btm] = 0; g.d_vz_2[btm] = 0;
        }
        else {
            g.d_vx[top] = 0; g.d_vz[top] = 0; g.d_vx[btm] = 0; g.d_vz[btm] = 0;
        }
    }
    if (idx < g.NZ) {
        int left = idx * g.NX;
        int right = idx * g.NX + (g.NX - 1);
        if (g.flag_type == 1 || g.flag_type == 3) {
            g.d_vx_1[left] = 0; g.d_vx_2[left] = 0; g.d_vz_1[left] = 0; g.d_vz_2[left] = 0;
            g.d_vx_1[right] = 0; g.d_vx_2[right] = 0; g.d_vz_1[right] = 0; g.d_vz_2[right] = 0;
        }
        else {
            g.d_vx[left] = 0; g.d_vz[left] = 0; g.d_vx[right] = 0; g.d_vz[right] = 0;
        }
    }
}

// --- Colormap Generator (For Real-Time rendering to OpenGL texture) ---
__global__ void generate_colormap_kernel(GPUSimData g, uchar4* rgba_out, float scale, int component) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < g.NZ && j < g.NX) {
        int k = i * g.NX + j;
        float val = 0.0f;

        if (component == 0) { // Vz
            val = (g.flag_type == 1 || g.flag_type == 3) ? (g.d_vz_1[k] + g.d_vz_2[k]) : g.d_vz[k];
        }
        else { // Vx
            val = (g.flag_type == 1 || g.flag_type == 3) ? (g.d_vx_1[k] + g.d_vx_2[k]) : g.d_vx[k];
        }

        float norm = val * scale;
        if (norm > 1.0f) norm = 1.0f;
        if (norm < -1.0f) norm = -1.0f;

        // 蓝-白-红 双极地震色谱 (Seismic Bipolar Red-White-Blue)
        unsigned char r = 255, g_val = 255, b = 255;
        if (norm > 0.0f) { // 正振幅 -> 红色
            g_val = static_cast<unsigned char>(255.0f * (1.0f - norm));
            b = static_cast<unsigned char>(255.0f * (1.0f - norm));
        }
        else { // 负振幅 -> 蓝色
            r = static_cast<unsigned char>(255.0f * (1.0f + norm));
            g_val = static_cast<unsigned char>(255.0f * (1.0f + norm));
        }

        rgba_out[k] = make_uchar4(r, g_val, b, 255);
    }
}


// =============================================================================
// 主机驱动接口实现 (Host Drivers)
// =============================================================================

void initGPUSimulation(GPUSimData& gpu, const SimulationContext& ctx) {
    gpu.NX = ctx.NX; gpu.NZ = ctx.NZ; gpu.total_grid = ctx.total_grid;
    gpu.dt = ctx.dt; gpu.inv_dt = 1.0f / ctx.dt;
    gpu.flag_type = ctx.flag_type; gpu.npml = ctx.npml;
    gpu.fs_idx = ctx.npml; gpu.upFlag = ctx.upFlag;
    gpu.c1_h = ctx.c1_h; gpu.c2_h = ctx.c2_h; gpu.c3_h = ctx.c3_h; gpu.c4_h = ctx.c4_h;

    size_t bytes = ctx.total_grid * sizeof(float);

    CUDA_CHECK(cudaMalloc(&gpu.d_vx_1, bytes)); CUDA_CHECK(cudaMemset(gpu.d_vx_1, 0, bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_vx_2, bytes)); CUDA_CHECK(cudaMemset(gpu.d_vx_2, 0, bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_vz_1, bytes)); CUDA_CHECK(cudaMemset(gpu.d_vz_1, 0, bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_vz_2, bytes)); CUDA_CHECK(cudaMemset(gpu.d_vz_2, 0, bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_sxx_1, bytes)); CUDA_CHECK(cudaMemset(gpu.d_sxx_1, 0, bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_sxx_2, bytes)); CUDA_CHECK(cudaMemset(gpu.d_sxx_2, 0, bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_szz_1, bytes)); CUDA_CHECK(cudaMemset(gpu.d_szz_1, 0, bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_szz_2, bytes)); CUDA_CHECK(cudaMemset(gpu.d_szz_2, 0, bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_sxz_1, bytes)); CUDA_CHECK(cudaMemset(gpu.d_sxz_1, 0, bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_sxz_2, bytes)); CUDA_CHECK(cudaMemset(gpu.d_sxz_2, 0, bytes));

    CUDA_CHECK(cudaMalloc(&gpu.d_rho, bytes)); CUDA_CHECK(cudaMemcpy(gpu.d_rho, ctx.rho.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&gpu.d_mu, bytes)); CUDA_CHECK(cudaMemcpy(gpu.d_mu, ctx.mu.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&gpu.d_lambda, bytes)); CUDA_CHECK(cudaMemcpy(gpu.d_lambda, ctx.lambda.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&gpu.d_lambda2mu, bytes)); CUDA_CHECK(cudaMemcpy(gpu.d_lambda2mu, ctx.lambda2mu.data(), bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&gpu.d_dx, ctx.NX * sizeof(float))); CUDA_CHECK(cudaMemcpy(gpu.d_dx, ctx.dx.data(), ctx.NX * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&gpu.d_dx_half, ctx.NX * sizeof(float))); CUDA_CHECK(cudaMemcpy(gpu.d_dx_half, ctx.dx_half.data(), ctx.NX * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&gpu.d_dz, ctx.NZ * sizeof(float))); CUDA_CHECK(cudaMemcpy(gpu.d_dz, ctx.dz.data(), ctx.NZ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&gpu.d_dz_half, ctx.NZ * sizeof(float))); CUDA_CHECK(cudaMemcpy(gpu.d_dz_half, ctx.dz_half.data(), ctx.NZ * sizeof(float), cudaMemcpyHostToDevice));

    if (gpu.flag_type == 2) {
        CUDA_CHECK(cudaMalloc(&gpu.d_vx, bytes)); CUDA_CHECK(cudaMemset(gpu.d_vx, 0, bytes));
        CUDA_CHECK(cudaMalloc(&gpu.d_vz, bytes)); CUDA_CHECK(cudaMemset(gpu.d_vz, 0, bytes));
        CUDA_CHECK(cudaMalloc(&gpu.d_sigmaxx, bytes)); CUDA_CHECK(cudaMemset(gpu.d_sigmaxx, 0, bytes));
        CUDA_CHECK(cudaMalloc(&gpu.d_sigmazz, bytes)); CUDA_CHECK(cudaMemset(gpu.d_sigmazz, 0, bytes));
        CUDA_CHECK(cudaMalloc(&gpu.d_sigmaxz, bytes)); CUDA_CHECK(cudaMemset(gpu.d_sigmaxz, 0, bytes));

        // 临时在Host初始化Type 2参数并拷入GPU
        std::vector<float> mu_x(ctx.total_grid, 0.0f), lambda_x(ctx.total_grid, 0.0f), l2m_x(ctx.total_grid, 0.0f);
        std::vector<float> mu_z(ctx.total_grid, 0.0f), rho_xz(ctx.total_grid, 0.0f), rho_orig(ctx.total_grid, 0.0f);
        for (int i = 0; i < ctx.NZ; ++i) {
            for (int j = 0; j < ctx.NX; ++j) {
                int k = i * ctx.NX + j;
                rho_orig[k] = ctx.rho[k];
                int k_next_x = (j < ctx.NX - 1) ? k + 1 : k;
                int k_next_z = (i < ctx.NZ - 1) ? k + ctx.NX : k;
                mu_x[k] = 0.5f * (ctx.mu[k] + ctx.mu[k_next_x]);
                lambda_x[k] = 0.5f * (ctx.lambda[k] + ctx.lambda[k_next_x]);
                l2m_x[k] = 0.5f * (ctx.lambda2mu[k] + ctx.lambda2mu[k_next_x]);
                mu_z[k] = 0.5f * (ctx.mu[k] + ctx.mu[k_next_z]);
                int k_dr = (i < ctx.NZ - 1 && j < ctx.NX - 1) ? k + ctx.NX + 1 : k;
                rho_xz[k] = 0.25f * (ctx.rho[k] + ctx.rho[k_next_x] + ctx.rho[k_next_z] + ctx.rho[k_dr]);
                if (rho_xz[k] < 1.0f) rho_xz[k] = 1000.0f;
            }
        }
        CUDA_CHECK(cudaMalloc(&gpu.d_mu_x_flat, bytes)); CUDA_CHECK(cudaMemcpy(gpu.d_mu_x_flat, mu_x.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&gpu.d_lambda_x_flat, bytes)); CUDA_CHECK(cudaMemcpy(gpu.d_lambda_x_flat, lambda_x.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&gpu.d_lambda2mu_x_flat, bytes)); CUDA_CHECK(cudaMemcpy(gpu.d_lambda2mu_x_flat, l2m_x.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&gpu.d_mu_z_flat, bytes)); CUDA_CHECK(cudaMemcpy(gpu.d_mu_z_flat, mu_z.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&gpu.d_rho_x_z_flat, bytes)); CUDA_CHECK(cudaMemcpy(gpu.d_rho_x_z_flat, rho_xz.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&gpu.d_rho_orig_flat, bytes)); CUDA_CHECK(cudaMemcpy(gpu.d_rho_orig_flat, rho_orig.data(), bytes, cudaMemcpyHostToDevice));
    }

    if (gpu.flag_type == 3) {
        CUDA_CHECK(cudaMalloc(&gpu.d_dp_flat, ctx.NX * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(gpu.d_dp_flat, ctx.dp_flat.data(), ctx.NX * sizeof(float), cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMalloc(&gpu.d_wavelet, ctx.wavelet.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(gpu.d_wavelet, ctx.wavelet.data(), ctx.wavelet.size() * sizeof(float), cudaMemcpyHostToDevice));

    if (ctx.num_rcv > 0) {
        CUDA_CHECK(cudaMalloc(&gpu.d_rcv_grid_idx, ctx.num_rcv * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(gpu.d_rcv_grid_idx, ctx.rcv_grid_idx.data(), ctx.num_rcv * sizeof(int), cudaMemcpyHostToDevice));
    }
}

void freeGPUSimulation(GPUSimData& gpu) {
    cudaFree(gpu.d_vx_1); cudaFree(gpu.d_vx_2); cudaFree(gpu.d_vz_1); cudaFree(gpu.d_vz_2);
    cudaFree(gpu.d_sxx_1); cudaFree(gpu.d_sxx_2); cudaFree(gpu.d_szz_1); cudaFree(gpu.d_szz_2);
    cudaFree(gpu.d_sxz_1); cudaFree(gpu.d_sxz_2);
    cudaFree(gpu.d_rho); cudaFree(gpu.d_mu); cudaFree(gpu.d_lambda); cudaFree(gpu.d_lambda2mu);
    cudaFree(gpu.d_dx); cudaFree(gpu.d_dx_half); cudaFree(gpu.d_dz); cudaFree(gpu.d_dz_half);
    cudaFree(gpu.d_wavelet); cudaFree(gpu.d_rcv_grid_idx);

    if (gpu.flag_type == 2) {
        cudaFree(gpu.d_vx); cudaFree(gpu.d_vz);
        cudaFree(gpu.d_sigmaxx); cudaFree(gpu.d_sigmazz); cudaFree(gpu.d_sigmaxz);
        cudaFree(gpu.d_mu_x_flat); cudaFree(gpu.d_lambda_x_flat); cudaFree(gpu.d_lambda2mu_x_flat);
        cudaFree(gpu.d_mu_z_flat); cudaFree(gpu.d_rho_x_z_flat); cudaFree(gpu.d_rho_orig_flat);
    }
    if (gpu.flag_type == 3) {
        cudaFree(gpu.d_dp_flat);
    }
}

void runGPUStep(GPUSimData& gpu, int current_it, const SimulationContext& ctx) {
    dim3 block(16, 16);
    dim3 grid((gpu.NX + block.x - 1) / block.x, (gpu.NZ + block.y - 1) / block.y);

    // 1. Stress Update
    if (gpu.flag_type == 1) {
        update_type1_stress_kernel << <grid, block >> > (gpu);
    }
    else if (gpu.flag_type == 2) {
        update_type2_stress_kernel << <grid, block >> > (gpu);
    }
    else if (gpu.flag_type == 3) {
        update_type3_stress_kernel << <grid, block >> > (gpu);
        if (gpu.upFlag) {
            type3_stress_mirror_kernel << <(gpu.NX + 255) / 256, 256 >> > (gpu);
        }
    }

    // 2. Velocity Update
    if (gpu.flag_type == 1) {
        update_type1_velocity_kernel << <grid, block >> > (gpu);
    }
    else if (gpu.flag_type == 2) {
        update_type2_velocity_kernel << <grid, block >> > (gpu);
    }
    else if (gpu.flag_type == 3) {
        update_type3_velocity_kernel << <grid, block >> > (gpu);
    }

    // 3. Source Injection
    float wavelet_val = ctx.wavelet[current_it] * 1e7f;
    float force_x = sinf(ctx.src_angle * M_PI / 180.0f) * wavelet_val;
    float force_z = cosf(ctx.src_angle * M_PI / 180.0f) * wavelet_val;

    int safe_src_z = (gpu.flag_type == 3) ? std::max(ctx.src_z_idx, gpu.fs_idx + 1) : ctx.src_z_idx;
    int src_idx = safe_src_z * gpu.NX + (ctx.src_idx % gpu.NX);

    inject_source_kernel << <1, 1 >> > (gpu, force_x, force_z, src_idx);

    // 4. Dirichlet boundary conditions
    int max_dim = std::max(gpu.NX, gpu.NZ);
    zero_boundary_kernel << <(max_dim + 255) / 256, 256 >> > (gpu);
}

void copyWavefieldToHost(GPUSimData& gpu, float* h_vz_out) {
    size_t bytes = gpu.total_grid * sizeof(float);
    if (gpu.flag_type == 1 || gpu.flag_type == 3) {
        // 合并分裂场 Vx/Vz 传回 Host
        float* temp_v1 = (float*)malloc(bytes);
        float* temp_v2 = (float*)malloc(bytes);
        CUDA_CHECK(cudaMemcpy(temp_v1, gpu.d_vz_1, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(temp_v2, gpu.d_vz_2, bytes, cudaMemcpyDeviceToHost));
        for (int i = 0; i < gpu.total_grid; i++) {
            h_vz_out[i] = temp_v1[i] + temp_v2[i];
        }
        free(temp_v1); free(temp_v2);
    }
    else {
        CUDA_CHECK(cudaMemcpy(h_vz_out, gpu.d_vz, bytes, cudaMemcpyDeviceToHost));
    }
}

// 接收器数据收集：我们直接在 Host 端发起少量单点 D2H 拷贝（或者编写专用内核）
void copyRecordFromGPU(GPUSimData& gpu, int num_rcv, int nt, int current_it, float* h_rec_vx, float* h_rec_vz) {
    if (num_rcv <= 0) return;
    std::vector<int> h_rcv_idx(num_rcv);
    CUDA_CHECK(cudaMemcpy(h_rcv_idx.data(), gpu.d_rcv_grid_idx, num_rcv * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<float> dev_vx(gpu.total_grid), dev_vz(gpu.total_grid);
    if (gpu.flag_type == 1 || gpu.flag_type == 3) {
        std::vector<float> v1(gpu.total_grid), v2(gpu.total_grid);
        CUDA_CHECK(cudaMemcpy(v1.data(), gpu.d_vx_1, gpu.total_grid * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(v2.data(), gpu.d_vx_2, gpu.total_grid * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < gpu.total_grid; i++) dev_vx[i] = v1[i] + v2[i];

        CUDA_CHECK(cudaMemcpy(v1.data(), gpu.d_vz_1, gpu.total_grid * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(v2.data(), gpu.d_vz_2, gpu.total_grid * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < gpu.total_grid; i++) dev_vz[i] = v1[i] + v2[i];
    }
    else {
        CUDA_CHECK(cudaMemcpy(dev_vx.data(), gpu.d_vx, gpu.total_grid * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(dev_vz.data(), gpu.d_vz, gpu.total_grid * sizeof(float), cudaMemcpyDeviceToHost));
    }

    for (int r = 0; r < num_rcv; ++r) {
        int k = h_rcv_idx[r];
        h_rec_vx[r * nt + current_it] = dev_vx[k];
        h_rec_vz[r * nt + current_it] = dev_vz[k];
    }
}

void generateWavefieldTextureCUDA(GPUSimData& gpu, uchar4* d_rgba_out, float color_scale, int show_component) {
    dim3 block(16, 16);
    dim3 grid((gpu.NX + block.x - 1) / block.x, (gpu.NZ + block.y - 1) / block.y);
    generate_colormap_kernel << <grid, block >> > (gpu, d_rgba_out, color_scale, show_component);
}