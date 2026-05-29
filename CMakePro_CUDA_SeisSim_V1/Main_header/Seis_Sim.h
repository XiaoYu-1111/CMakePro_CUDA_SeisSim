#pragma once
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

// =================================================================================
// 1. 平台相关定义与系统头文件 (Platform Specifics)
// =================================================================================
#ifdef _WIN32
#define NOMINMAX          // 禁用 Windows.h 自带的 min/max 宏，避免与 std::min/max 冲突
#include <windows.h>      // Windows API
#include <direct.h>       // _mkdir
#include <io.h>           // _access
#else
#include <sys/stat.h>     // mkdir
#include <sys/types.h>
#include <unistd.h>
#endif

#include "Common.h"
#include "Cuda_Check.cuh" // CUDA 计算接口
#include "SeismicIO.h"

// 声明外部未绑定的 FDM 主机端物理配置函数
extern "C" {
    void setupTestContext(SimulationContext& ctx, const Parameters& par);
    void generateRickerWavelet(std::vector<float>& wavelet, int nt, float dt, float f0, float t0);
}

// =============================================================================
// 2. 全局持久化状态控制变量 (File-level Global State Variables)
// =============================================================================
static SimulationContext ctx;
static GPUSimData        gpu_data;
static bool              is_seis_initialized = false;

// 全局控制信号
inline bool g_resetSimRequested = false; // C 键重置模拟信号
inline bool g_resetViewportRequested = false; // R 键重置视口信号

// --- 物理与网格大小控制 ---
inline int   edit_w = 1000;
inline int   edit_h = 500;
inline float edit_f0 = 100.0f;
inline float edit_t0 = 1.0f / edit_f0;

inline float temp_dt = 0.0002f;
inline float temp_dx = 1.0f;
inline int   temp_nt = 10000;
inline int   temp_pml = 50;

// --- 时间演化与渲染属性 ---
inline int   current_it = 0;
inline float accumulated_compute_time = 0.0f;
inline float color_scale = 20.0f;
inline int   show_component = 2;     // 0: Vz, 1: Vx
inline int   waveStyle = 0;     // 默认采用 Style 7 (Turbo)
inline int   modelStyle = 3; // 0: 钛金灰, 1: 科学地质图, 2: 灰度Vp, 3: 跟随波场, 4: 科学白背景
inline int   steps_per_frame = 20;
inline bool  showHUD = true;
inline bool  show_Monitor_par = false;


// --- 物理震源位置与受力角度 ---
inline int   edit_src_x = 500;
inline int   edit_src_z = 125;
inline float edit_angle = 0.0f;
inline float edit_amp = 1.0f; // 全局共享：激发幅值倍数 (0.1 ~ 10.0x)
inline int   edit_source_type = 0;    // 全局共享：激发震源类型 (0:垂直力, 1:水平力, 2:膨胀源, 3:剪切源, 4:倾斜力)

// --- 视口平移与滚轮缩放 ---
inline float      viewZoom = 1.0f;
inline ImVec2     viewOffset = { 0.0f, 0.0f };
inline bool       first_align_needed = false;
inline Parameters par;

// --- 动态多震源与无限模式 ---
inline bool                  multi_source_mode = false;
inline std::vector<GPUSource> active_sources;
inline bool                  infinite_mode = false;

// --- 鼠标悬停实时监控数据 ---
inline int   hover_mx = 0;
inline int   hover_my = 0;
inline float hover_Vp = 0.0f;
inline float hover_Vs = 0.0f;
inline float hover_Rho = 0.0f;
inline bool  hover_valid = false;

// --- 全局 UI 荧光主题色 (默认绿色荧光) ---
inline ImVec4 uiAccent = ImVec4(0.5f, 1.0f, 0.5f, 1.0f);

inline bool is_mouse_inside = false;

inline int current_scene = SCENE_TWO_LAYER; // 当前加载的物理场景

// --- 共享物性参数 ---
inline float edit_Vp = 2000.0f; // 全局共享：纵波速度
inline float edit_Vs = 1400.0f; // 全局共享：横波速度
inline float edit_Density = 2000.0f; // 全局共享：密度

inline GLuint g_modelTex = 0; // 离屏地质物性纹理句柄

inline bool  showGrid = true;  // 默认开启背景网格与矩阵点显示
inline float ui_gridSpacing = 10.0f; // 推荐默认 10.0f
inline float ui_gridOpacity = 1.0f;  // 推荐默认 1.0f (可在 UI 中无级变暗或变亮)

inline static bool        show_success_popup = false;
inline static bool        show_error_popup = false;
inline static std::string popup_message = "";

// --- 高级物理参数缓存 (避开每帧循环 400 万点导致的 FPS 暴降瓶颈) ---
inline float cached_max_vp = 0.0f;
inline float cached_min_vp = 0.0f;
inline float cached_max_vs = 0.0f;
inline float cached_min_vs = 0.0f;
inline float cached_max_rho = 0.0f;
inline float cached_min_rho = 0.0f;
inline float cached_vram_mb = 0.0f; // 估算显存占用

inline float edit_segy_dx = 1.0f; // 外部导入 SEG-Y 时实际物理道间距 (米)
inline float export_target_dx = 1.0f; // 导出目标网格间距 (米，默认与当前模拟步长对齐)


#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

// =============================================================================
// 【安全硬件监控】：C 兼容的 NVML 结构体与类型定义 (避免引入第三方 .h 产生编译冲突)
// =============================================================================
typedef int nvmlReturn_t;
#define NVML_SUCCESS 0
#define NVML_TEMPERATURE_GPU 0

typedef struct {
    unsigned int gpu;
    unsigned int memory;
} nvmlUtilization_t;

typedef void* nvmlDevice_t;

// 函数指针类型定义
typedef nvmlReturn_t(*pfn_nvmlInit)();
typedef nvmlReturn_t(*pfn_nvmlShutdown)();
typedef nvmlReturn_t(*pfn_nvmlDeviceGetHandleByIndex)(unsigned int, nvmlDevice_t*);
typedef nvmlReturn_t(*pfn_nvmlDeviceGetUtilizationRates)(nvmlDevice_t, nvmlUtilization_t*);
typedef nvmlReturn_t(*pfn_nvmlDeviceGetTemperature)(nvmlDevice_t, int, unsigned int*);
typedef nvmlReturn_t(*pfn_nvmlDeviceGetPowerUsage)(nvmlDevice_t, unsigned int*);

// =============================================================================
// 【安全硬件监控】：动态运行期 NVML 连接器 (Symmetric GPU Loader)
// =============================================================================
struct DynamicNVML {
    bool loaded = false;
#ifdef _WIN32
    HMODULE handle = nullptr;
#else
    void* handle = nullptr;
#endif

    pfn_nvmlInit                     init = nullptr;
    pfn_nvmlShutdown                 shutdown = nullptr;
    pfn_nvmlDeviceGetHandleByIndex   getHandle = nullptr;
    pfn_nvmlDeviceGetUtilizationRates getUtil = nullptr;
    pfn_nvmlDeviceGetTemperature     getTemp = nullptr;
    pfn_nvmlDeviceGetPowerUsage      getPower = nullptr;

    nvmlDevice_t devHandle = nullptr;

    // 动态载入系统层 NVIDIA 管理驱动
    void Load() {
        if (loaded) return;
#ifdef _WIN32
        handle = LoadLibraryA("nvml.dll"); // Windows 驱动目录下自带
#else
        handle = dlopen("libnvidia-ml.so", RTLD_LAZY); // Linux 系统库
#endif
        if (!handle) return;

#ifdef _WIN32
        init = (pfn_nvmlInit)GetProcAddress(handle, "nvmlInit");
        shutdown = (pfn_nvmlShutdown)GetProcAddress(handle, "nvmlShutdown");
        getHandle = (pfn_nvmlDeviceGetHandleByIndex)GetProcAddress(handle, "nvmlDeviceGetHandleByIndex");
        getUtil = (pfn_nvmlDeviceGetUtilizationRates)GetProcAddress(handle, "nvmlDeviceGetUtilizationRates");
        getTemp = (pfn_nvmlDeviceGetTemperature)GetProcAddress(handle, "nvmlDeviceGetTemperature");
        getPower = (pfn_nvmlDeviceGetPowerUsage)GetProcAddress(handle, "nvmlDeviceGetPowerUsage");
#else
        init = (pfn_nvmlInit)dlsym(handle, "nvmlInit");
        shutdown = (pfn_nvmlShutdown)dlsym(handle, "nvmlShutdown");
        getHandle = (pfn_nvmlDeviceGetHandleByIndex)dlsym(handle, "nvmlDeviceGetHandleByIndex");
        getUtil = (pfn_nvmlDeviceGetUtilizationRates)dlsym(handle, "nvmlDeviceGetUtilizationRates");
        getTemp = (pfn_nvmlDeviceGetTemperature)dlsym(handle, "nvmlDeviceGetTemperature");
        getPower = (pfn_nvmlDeviceGetPowerUsage)dlsym(handle, "nvmlDeviceGetPowerUsage");
#endif

        if (init && shutdown && getHandle && getUtil && getTemp && getPower) {
            if (init() == NVML_SUCCESS) {
                if (getHandle(0, &devHandle) == NVML_SUCCESS) {
                    loaded = true;
                }
                else {
                    shutdown();
                }
            }
        }
    }

    void Unload() {
        if (!loaded) return;
        if (shutdown) shutdown();
#ifdef _WIN32
        FreeLibrary(handle);
#else
        dlclose(handle);
#endif
        loaded = false;
    }
};

// 全局唯一的动态硬件监控器实例
inline DynamicNVML g_nvml;

// =============================================================================
// 3. 核心计算状态与极值估算缓存重算
// =============================================================================
inline void RecalculateCachedModelMetrics() {
    if (ctx.total_grid <= 0) return;

    float max_p = -1e20f, min_p = 1e20f;
    float max_s = -1e20f, min_s = 1e20f;
    float max_r = -1e20f, min_r = 1e20f;

    for (int i = 0; i < ctx.total_grid; ++i) {
        float rho = ctx.rho[i];
        float vp = 0.0f;
        float vs = 0.0f;
        if (rho > 0.0f) {
            vp = sqrtf(ctx.lambda2mu[i] / rho);
            vs = sqrtf(ctx.mu[i] / rho);
        }

        // 避开最顶端的低速空气层，以显示真实地下岩层的速度范围
        if (vp > 350.0f) {
            if (vp > max_p) max_p = vp;
            if (vp < min_p) min_p = vp;
            if (vs > max_s) max_s = vs;
            if (vs < min_s) min_s = vs;
            if (rho > max_r) max_r = rho;
            if (rho < min_r) min_r = rho;
        }
    }

    // 防呆保护
    if (max_p == -1e20f) {
        max_p = 2000.0f; min_p = 2000.0f;
        max_s = 1400.0f; min_s = 1400.0f;
        max_r = 2000.0f; min_r = 2000.0f;
    }

    cached_max_vp = max_p;
    cached_min_vp = min_p;
    cached_max_vs = max_s;
    cached_min_vs = min_s;
    cached_max_rho = max_r;
    cached_min_rho = min_r;

    // 估算 GPU 在此尺寸下分配的所有 2D 物理场与系数显存
    // 包含应力、速度、PML 分裂场、物性等共约 20 个单精度二维浮点显存块
    size_t estimated_bytes = ctx.total_grid * sizeof(float) * 20;
    cached_vram_mb = static_cast<float>(estimated_bytes) / (1024.0f * 1024.0f);
}


// =============================================================================
// 11. 离屏非均匀地质物性纹理打包上传器
// =============================================================================
inline void UpdateModelTexture() {
    if (g_modelTex == 0) {
        glGenTextures(1, &g_modelTex);
    }

    std::vector<float> temp_data(ctx.total_grid * 4, 0.0f);

    for (int i = 0; i < ctx.total_grid; ++i) {
        float rho = ctx.rho[i];
        float vp = 0.0f;
        float vs = 0.0f;
        if (rho > 0.0f) {
            vp = sqrtf(ctx.lambda2mu[i] / rho);
            vs = sqrtf(ctx.mu[i] / rho);
        }

        // 归一化后存入对应通道
        temp_data[i * 4 + 0] = vp / 6000.0f;  // R 通道: 纵波速度 Vp (映射至 6000m/s)
        temp_data[i * 4 + 1] = vs / 3500.0f;  // G 通道: 横波速度 Vs
        temp_data[i * 4 + 2] = rho / 3000.0f; // B 通道: 密度 Rho
        temp_data[i * 4 + 3] = 1.0f;          // A 通道
    }

    glBindTexture(GL_TEXTURE_2D, g_modelTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, ctx.NX, ctx.NZ, 0, GL_RGBA, GL_FLOAT, temp_data.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
}
// =============================================================================
// 【核心新增】：一键同步 CPU 地层修改至 GPU 并刷新地质背景纹理图层
// =============================================================================
inline void UploadMaterialToGPU() {
    if (ctx.total_grid <= 0) return;

    size_t bytes = ctx.total_grid * sizeof(float);

    // 1. 将修改后的基础弹性拉梅常数与密度矩阵重新拷入显存
    CUDA_CHECK(cudaMemcpy(gpu_data.d_rho, ctx.rho.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_data.d_mu, ctx.mu.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_data.d_lambda, ctx.lambda.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_data.d_lambda2mu, ctx.lambda2mu.data(), bytes, cudaMemcpyHostToDevice));

    // 2. 如果当前运行在 Type 2 PML (非分裂混合) 下，必须重新重组并拷入平展的半步长各向异性介质场 [3]
    if (gpu_data.flag_type == 2) {
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
        CUDA_CHECK(cudaMemcpy(gpu_data.d_mu_x_flat, mu_x.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu_data.d_lambda_x_flat, lambda_x.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu_data.d_lambda2mu_x_flat, l2m_x.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu_data.d_mu_z_flat, mu_z.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu_data.d_rho_x_z_flat, rho_xz.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu_data.d_rho_orig_flat, rho_orig.data(), bytes, cudaMemcpyHostToDevice));
    }

    // 3. 如果当前在 Type 3 PML (自由表面) 下，重新计算自由表面 dp_flat 阻抗
    if (gpu_data.flag_type == 3) {
        ctx.dp_flat.assign(ctx.NX, 0.0f);
        int fs_idx = ctx.npml;
        for (int j = 0; j < ctx.NX; ++j) {
            int k = fs_idx * ctx.NX + j;
            if (k < ctx.total_grid && ctx.lambda2mu[k] > 0.0f) {
                float l2m = ctx.lambda2mu[k];
                float lam = ctx.lambda[k];
                ctx.dp_flat[j] = (l2m * l2m - lam * lam) / l2m;
            }
        }
        CUDA_CHECK(cudaMemcpy(gpu_data.d_dp_flat, ctx.dp_flat.data(), ctx.NX * sizeof(float), cudaMemcpyHostToDevice));
    }

    // 4. 重算 CPU 侧物理极值与显存估算缓存 [3]
    RecalculateCachedModelMetrics();

    // 5. 重新上传并刷新离屏背景地层纹理图层，使 OpenGL 视口立即显示出画笔划过的层位颜色！
    UpdateModelTexture();
}

// =============================================================================
// 【重构版】：高保真物理标尺渲染器 (基于屏幕目标像素间距反向自适应解算)
//  支持在任意超大网格尺度下完美、均匀、永不重叠地呈现整百/整千标定数字
// =============================================================================
inline void RenderGridRulerOnBackbuffer(
    const SimulationContext& ctx,
    const SimState& state,
    int                      winW,
    int                      winH,
    float                    barHeight,
    const ImVec2& viewOffset,
    float                    viewZoom
) {
    ImDrawList* draw_list = ImGui::GetBackgroundDrawList();
    ImU32       textCol = IM_COL32(220, 220, 220, 225); // 荧光白
    ImU32       tickCol = IM_COL32(0, 255, 200, 100);   // 青色刻度线

    float  winAspect = (float)winW / (float)winH;
    float  simAspect = (float)ctx.NX / (float)ctx.NZ;
    ImVec2 aspectCorr = { 1.0f, 1.0f };

    if (winAspect > simAspect) {
        aspectCorr.x = winAspect / simAspect;
    }
    else {
        aspectCorr.y = simAspect / winAspect;
    }

    // =========================================================================
    // 【核心新增】：基于 100 像素目标间距的 X / Y 轴物理步长自适应解算器
    // =========================================================================
    auto calcNiceInterval = [&](float target_pixel_dist, float total_cells, float aspect_corr, float win_dim) -> float {
        // 1. 反向换算 100 像素在当前缩放率下对应的理论物理距离 (米)
        float target_phys = (target_pixel_dist * total_cells * aspect_corr * ctx.h) / (win_dim * viewZoom);
        if (target_phys <= 0.0f) return 10.0f;

        // 2. 利用 Log10 将其舍入到 10 的 N 次幂基准
        float log_val = log10f(target_phys);
        float base_power = powf(10.0f, floorf(log_val));
        float ratio = target_phys / base_power;

        // 3. 按照标准地学绘图网格比例 (1, 2, 5 划分法) 匹配最美观的步长
        float nice_interval = base_power;
        if (ratio > 5.0f)      nice_interval = base_power * 5.0f;
        else if (ratio > 2.0f) nice_interval = base_power * 2.0f;

        return (nice_interval < 1.0f) ? 1.0f : nice_interval;
        };

    // 分别解算符合 X 轴和 Y 轴像素分辨率的完美物理间距 (单位：米)
    float nice_interval_x = calcNiceInterval(100.0f, ctx.NX, aspectCorr.x, (float)winW);
    float nice_interval_y = calcNiceInterval(100.0f, ctx.NZ, aspectCorr.y, (float)winH);

    // =========================================================================
    // A. 绘制顶部 X 轴自适应物理刻度 (以物理“米”为循环单位，保证刻度值永远是整百/整十)
    // =========================================================================
    float total_width_m = ctx.NX * ctx.h;
    for (float physX = 0.0f; physX <= total_width_m; physX += nice_interval_x) {
        // 将物理位置反向折算为网格格数，进行 NDC 变换与像素投影
        float simX = physX / ctx.h;
        float aPos_x = (simX / (float)ctx.NX) * 2.0f - 1.0f;
        float Pndc_x = (aPos_x - viewOffset.x) * viewZoom / aspectCorr.x;
        float screenX = (Pndc_x * 0.5f + 0.5f) * winW;

        if (screenX >= 0.0f && screenX <= (float)winW) {
            float screenY = barHeight + 2.0f;
            draw_list->AddLine(ImVec2(screenX, screenY), ImVec2(screenX, screenY + 6.0f), tickCol, 1.5f);

            char buf[32];
            snprintf(buf, 32, "%.0fm", physX);

            ImVec2 txtSize = ImGui::CalcTextSize(buf);
            float  offsetX = -txtSize.x * 0.5f;

            ImVec2 pos(screenX + offsetX, screenY + 8.0f);
            draw_list->AddText(ImVec2(pos.x + 1, pos.y + 1), IM_COL32(0, 0, 0, 200), buf);
            draw_list->AddText(pos, textCol, buf);
        }
    }

    // =========================================================================
    // B. 绘制左侧 Y 轴自适应深度刻度 (以物理“米”为循环单位)
    // =============================================================================
    float total_depth_m = ctx.NZ * ctx.h;
    for (float physY = 0.0f; physY <= total_depth_m; physY += nice_interval_y) {
        float simY = physY / ctx.h;
        float aPos_y = 1.0f - (simY / (float)ctx.NZ) * 2.0f;
        float Pndc_y = (aPos_y - viewOffset.y) * viewZoom / aspectCorr.y;
        float screenY = (0.5f - Pndc_y * 0.5f) * winH;

        if (screenY >= barHeight && screenY <= (float)winH) {
            float screenX = 4.0f;
            draw_list->AddLine(ImVec2(screenX, screenY), ImVec2(screenX + 6.0f, screenY), tickCol, 1.5f);

            char buf[32];
            snprintf(buf, 32, "%.0fm", physY);

            ImVec2 txtSize = ImGui::CalcTextSize(buf);
            ImVec2 pos(screenX + 8.0f, screenY - txtSize.y * 0.5f);

            draw_list->AddText(ImVec2(pos.x + 1, pos.y + 1), IM_COL32(0, 0, 0, 200), buf);
            draw_list->AddText(pos, textCol, buf);
        }
    }
}

// =============================================================================
// 5. 鼠标画笔光标渲染 (星空/科技主题自适应)
// =============================================================================
inline void RenderBrushCursor(const SimState& state, const ViewportInfo& vp) {
    ImVec2 mousePos = ImGui::GetMousePos();
    bool   isInside = (mousePos.x >= vp.x && mousePos.x < (vp.x + vp.w) &&
        mousePos.y >= vp.y && mousePos.y < (vp.y + vp.h));

    if (ImGui::GetIO().WantCaptureMouse) return;

    if (isInside) {
        ImDrawList* fg = ImGui::GetForegroundDrawList();

        if (state.brushType == TOOL_NONE) {
            float t = (float)ImGui::GetTime();
            const float crosshairSize = 18.0f;
            const float outerRingRadius = 35.0f;

            // -----------------------------------------------------------------
            // 【自适应色彩系统】：解构全局 uiAccent 主题色进行高拟真重映射
            // -----------------------------------------------------------------
            float r = uiAccent.x;
            float g = uiAccent.y;
            float b = uiAccent.z;

            // 主色调：与前台主题 UI 100% 物理对齐
            ImU32 col_main = IM_COL32(r * 255, g * 255, b * 255, 255);

            // 动态扫描指针色：在主题色基础上进行高亮推高，形成耀眼的发光点
            ImU32 col_dynamic = IM_COL32(
                (int)std::min(255.0f, r * 1.15f * 255.0f),
                (int)std::min(255.0f, g * 1.15f * 255.0f),
                (int)std::min(255.0f, b * 1.15f * 255.0f),
                255
            );

            // 投影阴影：修改为更协调的深黑/暗绿暗角，使绿光边缘对比更柔和
            ImU32 col_shadow = IM_COL32(10, 18, 12, 160);

            // -----------------------------------------------------------------
            // 图层绘制逻辑 (保持原有高性能几何结构)
            // -----------------------------------------------------------------
            // 1. 底层高对比投影十字线
            fg->AddLine(ImVec2(mousePos.x - crosshairSize, mousePos.y), ImVec2(mousePos.x + crosshairSize, mousePos.y), col_shadow, 4.0f);
            fg->AddLine(ImVec2(mousePos.x, mousePos.y - crosshairSize), ImVec2(mousePos.x, mousePos.y + crosshairSize), col_shadow, 4.0f);

            // 2. 顶层主题色十字准星
            fg->AddLine(ImVec2(mousePos.x - crosshairSize, mousePos.y), ImVec2(mousePos.x + crosshairSize, mousePos.y), col_main, 2.0f);
            fg->AddLine(ImVec2(mousePos.x, mousePos.y - crosshairSize), ImVec2(mousePos.x, mousePos.y + crosshairSize), col_main, 2.0f);

            // 3. 科技准星外包圆环
            fg->AddCircle(mousePos, outerRingRadius, col_main, 32, 1.5f);

            // 4. 雷达旋转扫描扫描针
            float  angle = t * 4.0f;
            ImVec2 scanEnd = ImVec2(mousePos.x + cos(angle) * outerRingRadius,
                mousePos.y + sin(angle) * outerRingRadius);
            fg->AddLine(mousePos, scanEnd, col_dynamic, 2.0f);
            fg->AddCircleFilled(scanEnd, 3.5f, col_dynamic);

            // 5. 阻尼正弦收缩脉冲波
            float pulse_t = fmodf(t, 1.5f) / 1.5f;
            float pulse_radius = pulse_t * outerRingRadius;
            int   pulse_alpha = (int)(sin(pulse_t * 3.14159f) * 120);

            // 脉冲波颜色同步使用带透明度的主题色
            ImU32 col_pulse = IM_COL32(r * 255, g * 255, b * 255, pulse_alpha);
            fg->AddCircle(mousePos, pulse_radius, col_pulse, 32, 3.0f);
        }
        else {
            ImU32 brushColor;
            switch (state.brushType) {
            case TOOL_HIGH:   brushColor = IM_COL32(255, 100, 100, 200); break;
            case TOOL_LOW:    brushColor = IM_COL32(100, 150, 255, 200); break;
            case TOOL_WALL:   brushColor = IM_COL32(255, 255, 0, 200);   break;
            case TOOL_ERASER: brushColor = IM_COL32(200, 200, 200, 200); break;
            default:          brushColor = IM_COL32(255, 255, 255, 200); break;
            }

            float screenRadius = state.brushRadius * vp.scaleX;
            fg->AddCircle(mousePos, screenRadius, brushColor, 0, 2.0f);
            fg->AddCircle(mousePos, screenRadius - 1.0f, IM_COL32(0, 0, 0, 150), 0, 1.0f);
            fg->AddCircleFilled(mousePos, 3.0f, IM_COL32(255, 255, 255, 255));
        }
    }
}

// =============================================================================
// 6. 物理震源准星渲染 (高精度随背景平移缩放、无漂移)
// =============================================================================
inline void RenderSourceMarker(
    const SimulationContext& ctx,
    int                      src_x,
    int                      src_z,
    int                      winW,
    int                      winH,
    float                    barHeight,
    const ImVec2& viewOffset,
    float                    viewZoom
) {
    float  winAspect = (float)winW / (float)winH;
    float  simAspect = (float)ctx.NX / (float)ctx.NZ;
    ImVec2 aspectCorr = { 1.0f, 1.0f };

    if (winAspect > simAspect) {
        aspectCorr.x = winAspect / simAspect;
    }
    else {
        aspectCorr.y = simAspect / winAspect;
    }

    float aPos_x = ((float)src_x / (float)ctx.NX) * 2.0f - 1.0f;
    float aPos_y = 1.0f - ((float)src_z / (float)ctx.NZ) * 2.0f;

    float Pndc_x = (aPos_x - viewOffset.x) * viewZoom / aspectCorr.x;
    float screenX = (Pndc_x * 0.5f + 0.5f) * winW;

    float Pndc_y = (aPos_y - viewOffset.y) * viewZoom / aspectCorr.y;
    float screenY = (0.5f - Pndc_y * 0.5f) * winH;

    if (screenX >= 0.0f && screenX <= winW && screenY >= barHeight && screenY <= winH) {
        ImDrawList* fg = ImGui::GetForegroundDrawList();
        ImVec2      pos(screenX, screenY);

        ImU32 colMain = IM_COL32(125, 255, 125, 255);
        ImU32 colBlack = IM_COL32(0, 0, 0, 180);
        float lineLen = 20.0f;
        float gap = 3.0f;

        fg->AddLine(ImVec2(pos.x - lineLen, pos.y), ImVec2(pos.x + lineLen, pos.y), colBlack, 3.5f);
        fg->AddLine(ImVec2(pos.x, pos.y - lineLen), ImVec2(pos.x, pos.y + lineLen), colBlack, 3.5f);

        fg->AddLine(ImVec2(pos.x - lineLen, pos.y), ImVec2(pos.x - gap, pos.y), colMain, 1.5f);
        fg->AddLine(ImVec2(pos.x + gap, pos.y), ImVec2(pos.x + lineLen, pos.y), colMain, 1.5f);
        fg->AddLine(ImVec2(pos.x, pos.y - lineLen), ImVec2(pos.x, pos.y - gap), colMain, 1.5f);
        fg->AddLine(ImVec2(pos.x, pos.y + gap), ImVec2(pos.x, pos.y + lineLen), colMain, 1.5f);

        char buf[64];
        snprintf(buf, 64, "%d, %d (%.0fm, %.0fm)", src_x, src_z, src_x * ctx.h, src_z * ctx.h);

        ImVec2 textPos = ImVec2(pos.x + 8, pos.y + 8);
        fg->AddText(ImVec2(textPos.x + 1, textPos.y + 1), colBlack, buf);
        fg->AddText(textPos, colMain, buf);

        float t = (float)ImGui::GetTime();
        float pulse = (sin(t * 3.0f) * 0.5f + 0.5f) * 5.0f;
        fg->AddCircle(pos, 5.0f + pulse, IM_COL32(125, 255, 125, 155), 16, 1.0f);
    }
}
// =============================================================================
// 【新增】：全局悬浮模态弹窗渲染器 (处理加载成功/失败的 UI 弹窗反馈)
// =============================================================================
inline void RenderSeisPopups() {
    // -------------------------------------------------------------------------
    // A. 成功模态弹窗 (绿框白字)
    // -------------------------------------------------------------------------
    if (show_success_popup) {
        ImGui::OpenPopup("Simulation Success");
        show_success_popup = false; // 消费单次激发信号，避免重复 Open
    }

    // 为成功弹窗绑定绿色荧光边框
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.2f, 0.8f, 0.2f, 1.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.5f);

    if (ImGui::BeginPopupModal("Simulation Success", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.4f, 1.0f), "[ SUCCESS ]");
        ImGui::Separator();
        ImGui::Spacing();

        // 渲染我们在后台写入的字符串消息
        ImGui::Text("%s", popup_message.c_str());

        ImGui::Spacing();
        ImGui::Separator();

        if (ImGui::Button("OK", ImVec2(120, 0))) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();

    // -------------------------------------------------------------------------
    // B. 失败模态弹窗 (红框白字)
    // -------------------------------------------------------------------------
    if (show_error_popup) {
        ImGui::OpenPopup("Simulation Error");
        show_error_popup = false; // 消费单次激发信号
    }

    // 为失败弹窗绑定高警示度红框
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.9f, 0.2f, 0.2f, 1.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.5f);

    if (ImGui::BeginPopupModal("Simulation Error", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextColored(ImVec4(1.0f, 0.2f, 0.2f, 1.0f), "[ ERROR ]");
        ImGui::Separator();
        ImGui::Spacing();

        // 渲染后台写入的错误诊断信息
        ImGui::Text("%s", popup_message.c_str());

        ImGui::Spacing();
        ImGui::Separator();

        if (ImGui::Button("OK", ImVec2(120, 0))) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}

// 简单的雷克子波产生器
void generateRickerWavelet(std::vector<float>& wavelet, int nt, float dt, float f0, float t0) {
    wavelet.resize(nt);
    for (int i = 0; i < nt; ++i) {
        float t = i * dt - t0;
        float pi2_f0 = M_PI * M_PI * f0 * f0;
        wavelet[i] = (1.0f - 2.0f * pi2_f0 * t * t) * expf(-pi2_f0 * t * t);
    }
}


/// //////////////////////////////////////////第二部分
// =============================================================================
// 【新增】：根据当前地层物理极限，一键自动对齐并锁定到最大安全时间步长 (CFL = 0.480)
//  每当切换预设、导入 SEG-Y/TXT 自定义模型时，一键调用此函数即可完成物理对齐
// =============================================================================
// =============================================================================
// 【修正后】：一键自动对齐并锁定到最大安全时间步长 (CFL = 0.480)
// =============================================================================
inline void AutoAlignTimeStep() {
    if (ctx.total_grid <= 0) return;

    // 1. 自动提取地层最大纵波速度
    float max_vp = 0.0f;
    for (int k = 0; k < ctx.total_grid; ++k) {
        float vp = sqrtf(ctx.lambda2mu[k] / ctx.rho[k]);
        if (vp > max_vp) max_vp = vp;
    }
    if (max_vp <= 350.0f) max_vp = 2000.0f;

    // 2. 自动锁定在极限安全步长
    ctx.dt = 0.480f * ctx.h / max_vp;
    gpu_data.inv_dt = 1.0f / ctx.dt;
    temp_dt = ctx.dt;

    // =========================================================================
    // 【核心修复】：必须使用最新、对齐后的时间步长 ctx.dt 重新生成 Ricker 子波！
    //  否则用旧 dt 生成的子波在新时步下播放会发生严重的频率畸变 (缩放拉伸)
    // =========================================================================
    generateRickerWavelet(ctx.wavelet, ctx.nt, ctx.dt, edit_f0, edit_t0);
}
// =============================================================================
// 7. PML 吸收层阻尼衰减系数更新 (专注于差分系数建立，不越权修改系统全局 dt)
// =============================================================================
inline void UpdatePmlDampingArrays() {
    float max_vp = 0.0f;
    for (int k = 0; k < ctx.total_grid; ++k) {
        float vp = sqrtf(ctx.lambda2mu[k] / ctx.rho[k]);
        if (vp > max_vp) max_vp = vp;
    }
    if (max_vp <= 350.0f) max_vp = 2000.0f;

    ctx.dx.assign(ctx.NX, 0.0f);
    ctx.dx_half.assign(ctx.NX, 0.0f);
    ctx.dz.assign(ctx.NZ, 0.0f);
    ctx.dz_half.assign(ctx.NZ, 0.0f);

    if (ctx.npml > 0) {
        float L = ctx.npml * ctx.h;
        float d_max = (3.0f * max_vp * 16.12f * 3.5f) / (2.0f * L);
        float pml_power = 2.5f;

        for (int i = 0; i < ctx.npml; ++i) {
            float x_thick = (ctx.npml - i) / (float)ctx.npml;
            float x_thick_half = (ctx.npml - (i + 0.5f)) / (float)ctx.npml;
            if (x_thick_half < 0.0f) x_thick_half = 0.0f;

            float val = d_max * powf(x_thick, pml_power);
            float val_half = d_max * powf(x_thick_half, pml_power);

            ctx.dx[i] = val;
            ctx.dx[ctx.NX - 1 - i] = val;
            ctx.dx_half[i] = val_half;
            ctx.dx_half[ctx.NX - 1 - i] = val_half;
        }

        for (int i = 0; i < ctx.npml; ++i) {
            float z_thick = (ctx.npml - i) / (float)ctx.npml;
            float z_thick_half = (ctx.npml - (i + 0.5f)) / (float)ctx.npml;
            if (z_thick_half < 0.0f) z_thick_half = 0.0f;

            float val = d_max * powf(z_thick, pml_power);
            float val_half = d_max * powf(z_thick_half, pml_power);

            ctx.dz[i] = val;
            ctx.dz[ctx.NZ - 1 - i] = val;
            ctx.dz_half[i] = val_half;
            ctx.dz_half[ctx.NZ - 1 - i] = val_half;
        }
    }
}

// =============================================================================
// 8. 物理网格与测试上下文参数分配
// =============================================================================
void setupTestContext(SimulationContext& ctx, const Parameters& par) {
    ctx.NX = par.model.xnum;
    ctx.NZ = par.model.znum;
    ctx.total_grid = ctx.NX * ctx.NZ;
    ctx.dt = par.FDM.dt;
    ctx.nt = static_cast<int>(par.FDM.nt);
    ctx.npml = static_cast<int>(par.FDM.npml);

    // 仅在首次启动未初始化时设为默认值 3；其余情况继承用户在 UI 上的选择
    if (ctx.flag_type == 0) {
        ctx.flag_type = 3;
    }

    ctx.upFlag = (par.FDM.upFlag > 0.5f);

    float h = par.model.dx;
    if (h <= 0.0f) h = 1.0f; // 防呆保护
    ctx.h = h;

    temp_dx = h;
    temp_dt = ctx.dt;
    temp_nt = ctx.nt;
    temp_pml = ctx.npml;

    // 将无单位的 8 阶有限差分系数除以空间步长 h 
    ctx.c1_h = 1.125022f / h;
    ctx.c2_h = -0.04687594f / h;
    ctx.c3_h = 0.00416669f / h;
    ctx.c4_h = -0.00019234f / h;

    // 直接读取定义在头文件最顶部的、与 UI 强绑定的全局共享物性变量
    float mu_val = edit_Density * edit_Vs * edit_Vs;
    float lambda2mu_val = edit_Density * edit_Vp * edit_Vp;
    float lambda_val = lambda2mu_val - 2.0f * mu_val;

    // 重新填充本地 Context
    ctx.rho.assign(ctx.total_grid, edit_Density);
    ctx.mu.assign(ctx.total_grid, mu_val);
    ctx.lambda.assign(ctx.total_grid, lambda_val);
    ctx.lambda2mu.assign(ctx.total_grid, lambda2mu_val);

    // 重新计算 PML 阻尼吸收线
    UpdatePmlDampingArrays();

    // 自由表面 dp_flat 计算
    {
        ctx.dp_flat.assign(ctx.NX, 0.0f);
        int fs_idx = ctx.npml;
        for (int j = 0; j < ctx.NX; ++j) {
            int k = fs_idx * ctx.NX + j;
            if (k < ctx.total_grid && ctx.lambda2mu[k] > 0.0f) {
                float l2m = ctx.lambda2mu[k];
                float lam = ctx.lambda[k];
                ctx.dp_flat[j] = (l2m * l2m - lam * lam) / l2m;
            }
        }
    }

    // 震源初始点设定
    ctx.src_z_idx = ctx.NZ / 4;                                      // Z 轴深度设定在 1/4 处
    ctx.src_idx = ctx.src_z_idx * ctx.NX + (ctx.NX / 2);             // X 轴定位在中心 1/2 处
    ctx.src_angle = par.FDM.angle;
    ctx.src_type = edit_source_type; // 绑定类型
    ctx.src_amp = edit_amp;         // 绑定幅值
    generateRickerWavelet(ctx.wavelet, ctx.nt, ctx.dt, par.FDM.f0, par.FDM.t0);

    // 检波器 (中间水平放一行检波器)
    ctx.num_rcv = 100;
    ctx.rcv_grid_idx.resize(ctx.num_rcv);
    int rcv_z = ctx.NZ / 2 + 30; // 震源下方 30 采样点
    int step = ctx.NX / ctx.num_rcv;
    for (int r = 0; r < ctx.num_rcv; ++r) {
        ctx.rcv_grid_idx[r] = rcv_z * ctx.NX + (r * step);
    }

    ctx.record_vx.assign(ctx.num_rcv * ctx.nt, 0.0f);
    ctx.record_vz.assign(ctx.num_rcv * ctx.nt, 0.0f);

    RecalculateCachedModelMetrics();
}

// =============================================================================
// 9. 核心物性注入器：将速度、密度转换为 CPU 端弹性拉梅常数 (对齐 Y 轴)
// =============================================================================
inline void SetMaterialAt(int x, int y, float vp, float vs, float rho) {
    // 垂直镜像对齐：使 LoadScenario 的 y=0 对应 CUDA 网格的最底层 (NZ - 1 - y)
    int k = (ctx.NZ - 1 - y) * ctx.NX + x;
    if (k >= 0 && k < ctx.total_grid) {
        ctx.rho[k] = rho;
        ctx.mu[k] = rho * vs * vs;
        ctx.lambda2mu[k] = rho * vp * vp;
        ctx.lambda[k] = ctx.lambda2mu[k] - 2.0f * ctx.mu[k];
    }
}

// =============================================================================
// 10. 场景装载器：完美适配 11 套高维地球物理仿真场景
// =============================================================================
inline void LoadScenario(int type) {
    int W = ctx.NX;
    int H = ctx.NZ;

    // 默认背景常数 (硬砂岩)
    float defVp = 3000.0f, defVs = 1732.0f, defRho = 2200.0f;

    // 默认背景填充
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            SetMaterialAt(x, y, defVp, defVs, defRho);
        }
    }

    // 默认激发位置重定位
    edit_src_x = W / 2;
    edit_src_z = H / 2;

    if (type == SCENE_EARTH) {
        // 地核地幔分层模型 (展示液态铁外核横波 Vs=0 消失与 P 波透射物理现象)
        int   cx = W / 2;
        int   cy = H / 2;
        float maxRadius = (std::min(W, H) / 2.0f) * 0.95f;
        float r_core = maxRadius * 0.54f;
        float r_mantle = maxRadius * 0.92f;

        float r_core_sq = r_core * r_core;
        float r_mantle_sq = r_mantle * r_mantle;
        float max_sq = maxRadius * maxRadius;

        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float dx = (float)(x - cx);
                float dy = (float)(y - cy);
                float distSq = dx * dx + dy * dy;

                if (distSq < r_core_sq) {
                    SetMaterialAt(x, y, 4000.0f, 0.0f, 4000.0f);    // 液态地核：Vs=0
                }
                else if (distSq < r_mantle_sq) {
                    SetMaterialAt(x, y, 6000.0f, 3400.0f, 3000.0f); // 固体地幔
                }
                else if (distSq < max_sq) {
                    SetMaterialAt(x, y, 3000.0f, 1700.0f, 2500.0f); // 地壳
                }
                else {
                    SetMaterialAt(x, y, 340.0f, 10.0f, 100.0f);     // 外部低速介质
                }
            }
        }
        edit_src_x = W / 2;
        edit_src_z = H / 2;
    }
    else if (type == SCENE_DOUBLE_SLIT) {
        // 水中高阻抗双缝挡板干涉场景
        int cy = H / 2;
        int cx = W / 2;
        int wallX = cx;

        // 水介质背景 (Vp=1500, Vs=0)
        for (int i = 0; i < W * H; i++) {
            SetMaterialAt(i % W, i / W, 1500.0f, 0.0f, 1000.0f);
        }

        // 高阻抗墙体屏障 (厚10格) 与双缝
        for (int y = 0; y < H; ++y) {
            bool isSlit = (abs(y - cy) > 30 && abs(y - cy) < 60);
            if (!isSlit) {
                for (int w = 0; w < 10; w++) {
                    SetMaterialAt(wallX + w, y, 5500.0f, 3200.0f, 8000.0f); // 高速致密钢质屏障
                }
            }
        }
        edit_src_x = W / 4;
        edit_src_z = cy;
    }
    else if (type == SCENE_TWO_LAYER) {
        // 典型双层层状介质模型
        int interfaceY = H / 2;
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                if (y < interfaceY) {
                    SetMaterialAt(x, y, 4000.0f, 2300.0f, 2500.0f); // 深部硬岩
                }
                else {
                    SetMaterialAt(x, y, 2000.0f, 1100.0f, 1500.0f); // 浅部软岩
                }
            }
        }
        edit_src_x = W / 2;
        edit_src_z = H / 2 + 50;
    }
    else if (type == SCENE_WAVEGUIDE) {
        // 直条状低速波导通道 (能量在其中全反射限制传播)
        int cy = H / 2;
        int guideHalfWidth = 60;
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                if (abs(y - cy) < guideHalfWidth) {
                    SetMaterialAt(x, y, 2000.0f, 1000.0f, 1500.0f); // 低速芯层
                }
                else {
                    SetMaterialAt(x, y, 100.0f, 60.0f, 10000.0f);   // 高阻抗包层
                }
            }
        }
        edit_src_x = 100;
        edit_src_z = cy;
    }
    else if (type == SCENE_WAVEGUIDE_CURVED) {
        // 正弦曲线型弯曲高速波导
        int cy = H / 2;
        int guideHalfWidth = 60;
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float currentCenterY = (H / 2) + (H / 4) * sin((float)x / W * 2.0f * 3.14159f);
                if (abs(y - currentCenterY) < guideHalfWidth) {
                    SetMaterialAt(x, y, 2000.0f, 1000.0f, 1500.0f);
                }
                else {
                    SetMaterialAt(x, y, 5500.0f, 3200.0f, 3000.0f);
                }
            }
        }
        edit_src_x = 100;
        edit_src_z = H / 2;
    }
    else if (type == SCENE_CRYSTAL) {
        // 晶体交错六角声子晶格 (钢球格栅阻碍与干涉波前)
        float bgVp = 1500.0f, bgVs = 0.0f, bgRho = 1000.0f;   // 水背景
        float scVp = 5800.0f, scVs = 3200.0f, scRho = 7800.0f;// 钢球
        int   radius = 12, spacingX = 50, spacingY = 50;
        int   startX = 250, endX = W - 100, startY = 100, endY = H - 100;

        for (int i = 0; i < W * H; ++i) SetMaterialAt(i % W, i / W, bgVp, bgVs, bgRho);

        int   cols = (endX - startX) / spacingX;
        int   rows = (endY - startY) / spacingY;
        float r2 = (float)(radius * radius);

        for (int j = 0; j < rows; ++j) {
            for (int i = 0; i < cols; ++i) {
                int cx = startX + i * spacingX;
                int cy = startY + j * spacingY;
                if (j % 2 == 1) cx += spacingX / 2; // 蜂窝交错

                for (int y = cy - radius; y <= cy + radius; ++y) {
                    for (int x = cx - radius; x <= cx + radius; ++x) {
                        if (x >= 0 && x < W && y >= 0 && y < H) {
                            float dx = (float)(x - cx);
                            float dy = (float)(y - cy);
                            if (dx * dx + dy * dy <= r2) {
                                SetMaterialAt(x, y, scVp, scVs, scRho);
                            }
                        }
                    }
                }
            }
        }
        edit_src_x = 100;
        edit_src_z = H / 2;
    }
    else if (type == SCENE_RANDOM_SCATTER) {
        // 随机泡泡介质强散射场
        float bgVp = 2500.0f, bgVs = 1200.0f, bgRho = 2000.0f;
        float scVp = 4500.0f, scVs = 2500.0f, scRho = 2800.0f;

        for (int i = 0; i < W * H; ++i) SetMaterialAt(i % W, i / W, bgVp, bgVs, bgRho);

        int numScatterers = 300;
        int minR = 5, maxR = 15;
        for (int k = 0; k < numScatterers; ++k) {
            int   cx = rand() % (W - 100) + 50;
            int   cy = rand() % (H - 100) + 50;
            int   r = rand() % (maxR - minR) + minR;
            float r2 = (float)(r * r);

            for (int y = cy - r; y <= cy + r; ++y) {
                for (int x = cx - r; x <= cx + r; ++x) {
                    if (x >= 0 && x < W && y >= 0 && y < H) {
                        float dx = x - cx;
                        float dy = y - cy;
                        if (dx * dx + dy * dy <= r2) {
                            SetMaterialAt(x, y, scVp, scVs, scRho);
                        }
                    }
                }
            }
        }
        edit_src_x = W / 2;
        edit_src_z = H / 2;
    }
    else if (type == SCENE_CURVED) {
        // 起伏正弦两层介质分界面
        float l1_Vp = 2000.0f, l1_Vs = 1100.0f, l1_Rho = 1800.0f;
        float l2_Vp = 3500.0f, l2_Vs = 2000.0f, l2_Rho = 2400.0f;
        float l3_Vp = 5500.0f, l3_Vs = 3200.0f, l3_Rho = 2800.0f;

        float baseH1 = H * 0.35f, amp1 = H * 0.08f, freq1 = 2.0f * 3.14159f / W * 2.5f;
        float baseH2 = H * 0.65f, amp2 = H * 0.05f, freq2 = 2.0f * 3.14159f / W * 1.5f, phase2 = 1.0f;

        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float h1 = baseH1 + amp1 * sin(x * freq1);
                float h2 = baseH2 + amp2 * sin(x * freq2 + phase2);
                if (y < h1)       SetMaterialAt(x, y, l3_Vp, l3_Vs, l3_Rho);
                else if (y < h2)  SetMaterialAt(x, y, l2_Vp, l2_Vs, l2_Rho);
                else              SetMaterialAt(x, y, l1_Vp, l1_Vs, l1_Rho);
            }
        }
        edit_src_x = W / 2;
        edit_src_z = H - 50;
    }
    else if (type == SCENE_REFRACTION) {
        // 速度连续线性增加的梯度介质 (折射和回转回弹波)
        float vTop = 2000.0f, vBot = 5500.0f;
        float rhoTop = 1800.0f, rhoBot = 3000.0f;
        for (int y = 0; y < H; ++y) {
            float ratio = (float)y / (float)H;
            float vp = vBot + (vTop - vBot) * ratio;
            float vs = vp / 1.732f;
            if (vp < 1600.0f) vs = 0.0f;
            float rho = rhoBot + (rhoTop - rhoBot) * ratio;

            for (int x = 0; x < W; ++x) SetMaterialAt(x, y, vp, vs, rho);
        }
        edit_src_x = 100;
        edit_src_z = H - 50; // 浅层激发
    }
    else if (type == SCENE_PENROSE_ROOM) {
        // 彭罗斯椭圆房间 (展示波形双焦点汇聚透镜效应)
        float rockVp = 3000.0f, rockVs = 1732.0f, rockRho = 2500.0f;
        float airVp = 0.0f, airVs = 0.0f, airRho = 20.0f;
        float cx = W * 0.5f, cy = H * 0.5f;
        float s = std::min(W, H) * 0.38f;

        auto isInsidePenrose = [&](float px, float py) -> bool {
            float x = (px - cx) / s;
            float y = (py - cy) / s;
            float room_half_width = 1.5f;
            bool  top = (y < -0.4f) && ((x * x) / (room_half_width * room_half_width) + pow((y + 0.4f) / 0.7f, 2) <= 1.0f);
            bool  bottom = (y > 0.4f) && ((x * x) / (room_half_width * room_half_width) + pow((y - 0.4f) / 0.7f, 2) <= 1.0f);
            bool  middle_rect = (abs(y) <= 0.4f) && (abs(x) <= room_half_width);
            bool  room_base = top || bottom || middle_rect;

            float mush_pos_x = 1.3f;
            auto  is_mushroom = [&](float mx, float my) -> bool {
                bool cap = (mx > 0.0f) && (my * my / 0.16f + mx * mx / 0.09f <= 1.0f);
                bool stem = (mx <= 0.0f) && (abs(my) <= 0.08f) && (mx > -0.6f);
                return cap || stem;
                };
            bool left_mushroom = is_mushroom(x - (-mush_pos_x), y);
            bool right_mushroom = is_mushroom(-(x - mush_pos_x), y);
            return room_base && !left_mushroom && !right_mushroom;
            };

        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                if (isInsidePenrose((float)x, (float)y)) SetMaterialAt(x, y, rockVp, rockVs, rockRho);
                else                                     SetMaterialAt(x, y, airVp, airVs, airRho);
            }
        }
        edit_src_x = static_cast<int>(cx);
        edit_src_z = static_cast<int>(cy);
    }
}


// =============================================================================
// 12. 弹性场景重装一键应用驱动
// =============================================================================
inline void ApplyScenario(int type, SimState& state) {
    state.running = false;
    current_it = 0;
    accumulated_compute_time = 0.0f;
    active_sources.clear(); // 清空多点激发队列

    // 1. 重新构建 CPU 物理场几何框架
    setupTestContext(ctx, par);

    // 2. 装载目标地层模型
    LoadScenario(type);

    // 3. 实时重新计算物理极限所匹配的自适应 PML 阻尼系数
    UpdatePmlDampingArrays();
    AutoAlignTimeStep();
    // 4. 重算自由表面 dp_flat 阻抗
    ctx.dp_flat.assign(ctx.NX, 0.0f);
    int fs_idx = ctx.npml;
    for (int j = 0; j < ctx.NX; ++j) {
        int k = fs_idx * ctx.NX + j;
        if (k < ctx.total_grid && ctx.lambda2mu[k] > 0.0f) {
            float l2m = ctx.lambda2mu[k];
            float lam = ctx.lambda[k];
            ctx.dp_flat[j] = (l2m * l2m - lam * lam) / l2m;
        }
    }

    // 5. 注销、重装并初始化 GPU 显存
    freeGPUSimulation(gpu_data);
    initGPUSimulation(gpu_data, ctx);

    // 6. 同步上传离屏二维物性纹理并更新系统极值缓存
    UpdateModelTexture();
    RecalculateCachedModelMetrics();
}

/// ////////////////////////////////////////////////////////////////////////第三部分
// =============================================================================
// 13. 系统原生打开文件选择对话框 (Windows/Linux 跨平台双模)
// =============================================================================
inline bool OpenSystemFileDialog(char* buffer, int bufferSize) {
    memset(buffer, 0, bufferSize);

#ifdef _WIN32
    // Windows 原生底层实现
    OPENFILENAMEA ofn;
    memset(&ofn, 0, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFile = buffer;
    ofn.nMaxFile = bufferSize;
    ofn.lpstrFilter = "Model Data (*.txt;*.sgy;*.csv)\0*.txt;*.sgy;*.segy;*.csv\0All Files (*.*)\0*.*\0";
    ofn.nFilterIndex = 1; // 默认选中第一个 (Model Data)
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
    return GetOpenFileNameA(&ofn);
#else
    // Linux 原生实现 (需要终端预装 zenity 命令组件)
    const char* cmd = "zenity --file-selection --file-filter='*.txt' --title='Select Model Data'";
    FILE* pipe = popen(cmd, "r");
    if (!pipe) {
        std::cerr << "[IO Error] Zenity not found. Please install it to use the file dialog." << std::endl;
        return false;
    }
    bool success = false;
    if (fgets(buffer, bufferSize, pipe) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // 裁剪末尾换行符
        }
        success = (strlen(buffer) > 0);
    }
    pclose(pipe);
    return success;
#endif
}

// =============================================================================
// 14. 系统原生保存文件选择对话框 (Windows/Linux 跨平台双模)
// =============================================================================
#ifdef _WIN32
inline bool SaveSystemFileDialog(char* outPath, DWORD maxPathLen) {
    OPENFILENAMEA ofn;
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFile = outPath;
    ofn.lpstrFile[0] = '\0';
    ofn.nMaxFile = maxPathLen;
    ofn.lpstrFilter = "Seismic SEGY (*.sgy)\0*.sgy;*.segy\0All Files (*.*)\0*.*\0";
    ofn.nFilterIndex = 1;
    ofn.lpstrFileTitle = NULL;
    ofn.nMaxFileTitle = 0;
    ofn.lpstrInitialDir = NULL;

    // OFN_NOCHANGEDIR 极其关键，防止更改程序运行目录导致相对路径着色器失效
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT | OFN_NOCHANGEDIR;

    return GetSaveFileNameA(&ofn) == TRUE;
}
#else
inline bool SaveSystemFileDialog(char* outPath, size_t maxPathLen) {
    return false;
}
#endif

// =============================================================================
// 15. 列优先 (Column-Major) 2D 物理模型网格重采样算法 (最近邻插值)
// =============================================================================
inline std::vector<std::vector<float>> ResampleGrid2D(
    const std::vector<std::vector<float>>& input,
    float                                  target_dx,
    float                                  original_dx
) {
    if (input.empty() || input[0].empty()) return {};

    int old_nx = static_cast<int>(input.size());    // 原始 X 点数 (Traces)
    int old_nz = static_cast<int>(input[0].size()); // 原始 Z 点数 (Samples)

    // 计算实际地质体的物理宽高
    float physical_width = (old_nx - 1) * original_dx;
    float physical_depth = (old_nz - 1) * original_dx;

    // 计算对应 target_dx 步长下的新网格点数
    int new_nx = static_cast<int>(physical_width / target_dx) + 1;
    int new_nz = static_cast<int>(physical_depth / target_dx) + 1;

    std::cout << "[Resample] Resampling grid: " << old_nx << "x" << old_nz
        << " -> " << new_nx << "x" << new_nz
        << " (spacing: " << original_dx << "m -> " << target_dx << "m)" << std::endl;

    std::vector<std::vector<float>> output(new_nx, std::vector<float>(new_nz, 0.0f));
    float ratio = target_dx / original_dx;

    for (int x = 0; x < new_nx; ++x) {
        int old_x = static_cast<int>(x * ratio + 0.5f);
        if (old_x >= old_nx) old_x = old_nx - 1;

        for (int z = 0; z < new_nz; ++z) {
            int old_z = static_cast<int>(z * ratio + 0.5f);
            if (old_z >= old_nz) old_z = old_nz - 1;

            output[x][z] = input[old_x][old_z];
        }
    }
    return output;
}

// =============================================================================
// 16. 列优先 2D 物理地层边界外推扩边算子 (带流体安全固化机制)
// =============================================================================
inline std::vector<std::vector<float>> PadPmlToGrid2D(
    const std::vector<std::vector<float>>& grid_vp,
    const std::vector<std::vector<float>>& grid_vs,
    const std::vector<std::vector<float>>& grid_rho,
    int                                    npml,
    int                                    component_type // 0: Vp, 1: Vs, 2: Rho
) {
    if (grid_vp.empty() || grid_vp[0].empty()) return {};

    int old_nx = static_cast<int>(grid_vp.size());
    int old_nz = static_cast<int>(grid_vp[0].size());

    int new_nx = old_nx + 2 * npml;
    int new_nz = old_nz + 2 * npml;

    std::vector<std::vector<float>> output(new_nx, std::vector<float>(new_nz, 0.0f));

    for (int x = 0; x < new_nx; ++x) {
        int  src_x = std::max(0, std::min(x - npml, old_nx - 1));
        bool in_x_pml = (x < npml) || (x >= new_nx - npml);

        for (int z = 0; z < new_nz; ++z) {
            int  src_z = std::max(0, std::min(z - npml, old_nz - 1));
            bool in_z_pml = (z < npml) || (z >= new_nz - npml);

            // 获取外推对齐后的原始物理量
            float vp = grid_vp[src_x][src_z];
            float vs = grid_vs[src_x][src_z];
            float rho = grid_rho[src_x][src_z];

            // 智能量纲转换 (g/cm3 转 kg/m3)
            if (rho > 0.0f && rho < 10.0f) rho *= 1000.0f;
            if (rho < 10.0f) rho = 1000.0f;

            // 如果该点处于 PML 内部，且原先是液态地层（Vs == 0）
            // 在 PML 内部强行将其“固化”为剪切波速为 Vp / 2 的稳定固体介质，消灭 PML 奇点
            if (in_x_pml || in_z_pml) {
                if (vs <= 0.0f && vp > 350.0f) {
                    vs = vp / 2.0f; // 稳定剪切速度
                }
            }

            // 根据请求的分量返回物理值
            if (component_type == 0)      output[x][z] = vp;
            else if (component_type == 1) output[x][z] = vs;
            else                          output[x][z] = rho;
        }
    }
    return output;
}

// =============================================================================
// 17. 升级版 ASCII 格式 (TXT) 物理模型导入器
// =============================================================================
inline bool LoadModelFromTxt(const std::string& filename, bool flipVertically, GLHandles& gl, SimState& state) {
    std::cout << "[IO] Loading custom ASCII model: " << filename << " (Flip Y: " << (flipVertically ? "Yes" : "No") << ")" << std::endl;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[IO] Error: Could not open model file: " << filename << std::endl;
        return false;
    }

    // 1. 解析 Dimensions 头部尺寸
    std::string line;
    int         file_w = 0, file_h = 0;
    bool        dimensionsFound = false;

    while (std::getline(file, line)) {
        if (line.rfind("# Dimensions", 0) == 0) {
            sscanf_s(line.c_str(), "# Dimensions (Width x Height): %d %d", &file_w, &file_h);
            if (file_w > 0 && file_h > 0) dimensionsFound = true;
            break;
        }
    }

    if (!dimensionsFound) {
        std::cerr << "[IO] Error: Missing '# Dimensions' header." << std::endl;
        return false;
    }

    // 创建 CPU 临时 2D 原始物性数组
    std::vector<std::vector<float>> raw_vp(file_w, std::vector<float>(file_h, 0.0f));
    std::vector<std::vector<float>> raw_vs(file_w, std::vector<float>(file_h, 0.0f));
    std::vector<std::vector<float>> raw_rho(file_w, std::vector<float>(file_h, 0.0f));

    file.clear();
    file.seekg(0, std::ios::beg);

    // 2. 逐行解析并装入临时数组
    int pixelCount = 0;
    int total_pixels = file_w * file_h;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::stringstream ss(line);
        float vp = 0.0f, vs = 0.0f, rho = 0.0f;
        if (!(ss >> vp >> vs >> rho)) continue;

        if (pixelCount >= total_pixels) break;

        int currentX = pixelCount % file_w;
        int currentY = pixelCount / file_w;

        raw_vp[currentX][currentY] = vp;
        raw_vs[currentX][currentY] = vs;
        raw_rho[currentX][currentY] = rho;

        pixelCount++;
    }
    file.close();

    if (pixelCount < total_pixels) {
        std::cerr << "[IO] Error: Custom model data is incomplete." << std::endl;
        return false;
    }

    // 动态重采样 (对齐系统当前的物理空间步长 par.model.dx)
    float original_txt_dx = 1.0f;
    float target_system_dx = par.model.dx;

    auto grid_vp = ResampleGrid2D(raw_vp, target_system_dx, original_txt_dx);
    auto grid_vs = ResampleGrid2D(raw_vs, target_system_dx, original_txt_dx);
    auto grid_rho = ResampleGrid2D(raw_rho, target_system_dx, original_txt_dx);

    // PML 扩边与流体安全固化 (一气呵成)
    int  current_npml = static_cast<int>(par.FDM.npml);
    auto vp_pad = PadPmlToGrid2D(grid_vp, grid_vs, grid_rho, current_npml, 0); // 传 0 导 Vp
    auto vs_pad = PadPmlToGrid2D(grid_vp, grid_vs, grid_rho, current_npml, 1); // 传 1 导 Vs
    auto rho_pad = PadPmlToGrid2D(grid_vp, grid_vs, grid_rho, current_npml, 2); // 传 2 导 Rho

    // 3. 重置模拟演化参数与最终网格宽高
    state.running = false;
    current_it = 0;
    accumulated_compute_time = 0.0f;
    active_sources.clear();

    edit_w = static_cast<int>(vp_pad.size());
    edit_h = static_cast<int>(vp_pad[0].size());
    par.model.xnum = edit_w;
    par.model.znum = edit_h;

    // 重新在 CPU 端重组基础物理场大小
    setupTestContext(ctx, par);

    // 4. 将完美扩边与对齐的数据注入 CPU 内存
    for (int x = 0; x < ctx.NX; ++x) {
        for (int z = 0; z < ctx.NZ; ++z) {
            float vp = vp_pad[x][z];
            float vs = vs_pad[x][z];
            float rho = rho_pad[x][z];
            int   tgtX = x;
            int   tgtZ = z;

            if (flipVertically) {
                tgtZ = ctx.NZ - 1 - z; // 垂直地表对齐
            }
            SetMaterialAt(tgtX, tgtZ, vp, vs, rho);
        }
    }

    // 5. 重置震源位置与表面波 dp_flat
    ctx.src_z_idx = ctx.NZ / 4;
    ctx.src_idx = ctx.src_z_idx * ctx.NX + (ctx.NX / 2);
    edit_src_x = ctx.NX / 2;
    edit_src_z = ctx.NZ / 4;

    // 重新计算并标定最高波速下的 PML 阻尼系数
    UpdatePmlDampingArrays();
    AutoAlignTimeStep();

    ctx.dp_flat.assign(ctx.NX, 0.0f);
    int fs_idx = ctx.npml;
    for (int j = 0; j < ctx.NX; ++j) {
        int k = fs_idx * ctx.NX + j;
        if (k < ctx.total_grid && ctx.lambda2mu[k] > 0.0f) {
            float l2m = ctx.lambda2mu[k];
            float lam = ctx.lambda[k];
            ctx.dp_flat[j] = (l2m * l2m - lam * lam) / l2m;
        }
    }

    // 6. 释放并重新注册 FBO 纹理尺寸，防止显存崩溃
    freeGPUSimulation(gpu_data);
    cudaGraphicsUnregisterResource(gl.cudaSeisRes);
    glBindTexture(GL_TEXTURE_2D, gl.seisTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, ctx.NX, ctx.NZ, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    cudaGraphicsGLRegisterImage(&gl.cudaSeisRes, gl.seisTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

    // 7. 重新初始化新显存
    initGPUSimulation(gpu_data, ctx);

    // 8. 同步更新地质背景纹理，重新标定极值和视口对齐
    UpdateModelTexture();
    RecalculateCachedModelMetrics();
    first_align_needed = true;

    std::cout << "[IO] Custom ASCII Model Loaded, Padded and Stabilized Successfully: " << ctx.NX << "x" << ctx.NZ << std::endl;
    return true;
}

// =============================================================================
// 18. 升级版 SEG-Y 多道物理模型加载器 (支持双向翻转、对齐重采样)
// =============================================================================
inline bool LoadModelFromSegy(
    const std::string& vp_path,
    const std::string& vs_path,
    const std::string& rho_path,
    bool               flipVertically,
    bool               flipHorizontally,
    GLHandles& gl,
    SimState& state
) {
    std::cout << "[IO] Loading SEGY model files..." << std::endl;

    auto raw_vp = SeismicIO::readSegyFile2D(vp_path);
    auto raw_vs = SeismicIO::readSegyFile2D(vs_path);
    auto raw_rho = SeismicIO::readSegyFile2D(rho_path);

    if (raw_vp.empty() || raw_vs.empty() || raw_rho.empty()) {
        std::cerr << "[IO] Error: One or more SEGY files failed to load." << std::endl;
        return false;
    }

    // 1. 动态重采样 (对齐我们当前系统的物理空间步长 ctx.h / par.model.dx)
    float original_segy_dx = edit_segy_dx;
    float target_system_dx = par.model.dx;

    auto grid_vp = ResampleGrid2D(raw_vp, target_system_dx, original_segy_dx);
    auto grid_vs = ResampleGrid2D(raw_vs, target_system_dx, original_segy_dx);
    auto grid_rho = ResampleGrid2D(raw_rho, target_system_dx, original_segy_dx);

    // 2. PML 扩边 + 100% P波阻抗匹配 + 水层安全固化
    int  current_npml = static_cast<int>(par.FDM.npml);
    auto vp_pad = PadPmlToGrid2D(grid_vp, grid_vs, grid_rho, current_npml, 0); // 传 0 导 Vp
    auto vs_pad = PadPmlToGrid2D(grid_vp, grid_vs, grid_rho, current_npml, 1); // 传 1 导 Vs
    auto rho_pad = PadPmlToGrid2D(grid_vp, grid_vs, grid_rho, current_npml, 2); // 传 2 导 Rho

    // 确定最终尺寸
    int file_w = static_cast<int>(vp_pad.size());
    int file_h = static_cast<int>(vp_pad[0].size());

    state.running = false;
    current_it = 0;
    accumulated_compute_time = 0.0f;
    active_sources.clear();

    edit_w = file_w;
    edit_h = file_h;
    par.model.xnum = edit_w;
    par.model.znum = edit_h;

    // 重新在 CPU 侧初始化上下文尺寸
    setupTestContext(ctx, par);

    // 3. 将完美对齐、完全稳定的数据写入内存 (支持垂直/水平翻转)
    for (int x = 0; x < ctx.NX; ++x) {
        for (int z = 0; z < ctx.NZ; ++z) {
            float vp = vp_pad[x][z];
            float vs = vs_pad[x][z];
            float rho = rho_pad[x][z];

            int tgtX = x;
            if (flipHorizontally) tgtX = ctx.NX - 1 - x;

            int tgtZ = z;
            if (flipVertically)   tgtZ = ctx.NZ - 1 - z;

            SetMaterialAt(tgtX, tgtZ, vp, vs, rho);
        }
    }

    // 4. 重置激发位置与 PML 表面波 dp_flat
    ctx.src_z_idx = ctx.NZ / 4;
    ctx.src_idx = ctx.src_z_idx * ctx.NX + (ctx.NX / 2);
    edit_src_x = ctx.NX / 2;
    edit_src_z = ctx.NZ / 4;

    ctx.dp_flat.assign(ctx.NX, 0.0f);
    int fs_idx = ctx.npml;
    for (int j = 0; j < ctx.NX; ++j) {
        int k = fs_idx * ctx.NX + j;
        if (k < ctx.total_grid && ctx.lambda2mu[k] > 0.0f) {
            float l2m = ctx.lambda2mu[k];
            float lam = ctx.lambda[k];
            ctx.dp_flat[j] = (l2m * l2m - lam * lam) / l2m;
        }
    }
    UpdatePmlDampingArrays();
    AutoAlignTimeStep();
    // 5. 释放并重新注册 FBO 纹理尺寸，防止显存崩溃
    freeGPUSimulation(gpu_data);
    cudaGraphicsUnregisterResource(gl.cudaSeisRes);
    glBindTexture(GL_TEXTURE_2D, gl.seisTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, ctx.NX, ctx.NZ, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    cudaGraphicsGLRegisterImage(&gl.cudaSeisRes, gl.seisTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

    // 6. 重新初始化新大小的 GPU 显存
    initGPUSimulation(gpu_data, ctx);

    // 7. 同步重绘地质背景纹理，重新对齐视口标尺
    UpdateModelTexture();
    RecalculateCachedModelMetrics();
    first_align_needed = true;

    return true;
}

// =============================================================================
// 19. 终极版地层模型导出器 (支持空间局部裁剪 Cropping 与水平翻转导出为 SGY 记录)
// =============================================================================
// =============================================================================
// 【重构后】：支持空间局部裁剪、双向翻转以及动态重采样的 SEGY 模型导出驱动
// =============================================================================
inline bool ExportModelToSegy(
    const std::string& base_filepath,
    bool               density_gcm3,
    bool               flipHorizontally,
    int                x_min,
    int                z_min,
    int                x_max,
    int                z_max,
    float              target_dx // <-- 【核心新增】：目标导出网格步长 (米)
) {
    if (ctx.total_grid <= 0) return false;

    // 1. 安全边界检查与自适应纠错
    x_min = std::clamp(x_min, 0, ctx.NX - 1);
    x_max = std::clamp(x_max, 0, ctx.NX - 1);
    z_min = std::clamp(z_min, 0, ctx.NZ - 1);
    z_max = std::clamp(z_max, 0, ctx.NZ - 1);

    if (x_min > x_max) std::swap(x_min, x_max);
    if (z_min > z_max) std::swap(z_min, z_max);

    // 计算裁剪后的新模型尺寸 (新宽度、新深度)
    int new_NX = x_max - x_min + 1;
    int new_NZ = z_max - z_min + 1;

    std::cout << "[IO] Preparing SEGY Export with Cropping..." << std::endl;
    std::cout << "     - Crop Range: X[" << x_min << " ~ " << x_max << "], Z[" << z_min << " ~ " << z_max << "]" << std::endl;
    std::cout << "     - Raw Crop Size: " << new_NX << "x" << new_NZ << std::endl;

    // 2. 提取裁剪区域到 CPU 2D 临时缓冲区
    std::vector<std::vector<float>> cropped_vp(new_NX, std::vector<float>(new_NZ, 0.0f));
    std::vector<std::vector<float>> cropped_vs(new_NX, std::vector<float>(new_NZ, 0.0f));
    std::vector<std::vector<float>> cropped_rho(new_NX, std::vector<float>(new_NZ, 0.0f));

    for (int x = 0; x < new_NX; ++x) {
        // 如果开启了水平翻转，则对裁剪出的子区域进行左右镜像
        int targetX = flipHorizontally ? (x_max - x) : (x_min + x);

        for (int z = 0; z < new_NZ; ++z) {
            int targetZ = z_min + z; // 对齐深度

            int k_cuda = targetZ * ctx.NX + targetX; // 读取全网格中对应的位置

            float rho_val = ctx.rho[k_cuda];
            float vp_val = 0.0f;
            float vs_val = 0.0f;

            if (rho_val > 0.0f) {
                vp_val = sqrtf(ctx.lambda2mu[k_cuda] / rho_val);
                vs_val = sqrtf(ctx.mu[k_cuda] / rho_val);
            }

            cropped_vp[x][z] = vp_val;
            cropped_vs[x][z] = vs_val;

            // 物理单位换算
            if (density_gcm3) cropped_rho[x][z] = rho_val / 1000.0f;
            else              cropped_rho[x][z] = rho_val;
        }
    }

    // 3. 【核心新增】：物理重采样管线对齐
    std::vector<std::vector<float>> final_vp, final_vs, final_rho;
    int export_NX = new_NX;
    int export_NZ = new_NZ;

    // 只有当目标步长大于 0 且与当前步长不同时，才触发二维重采样插值
    if (target_dx > 0.0f && std::abs(target_dx - ctx.h) > 1e-4f) {
        std::cout << "     - [Trigger] Resampling sub-grid: " << ctx.h << "m -> " << target_dx << "m" << std::endl;
        final_vp = ResampleGrid2D(cropped_vp, target_dx, ctx.h);
        final_vs = ResampleGrid2D(cropped_vs, target_dx, ctx.h);
        final_rho = ResampleGrid2D(cropped_rho, target_dx, ctx.h);

        // 刷新重采样后的真实尺寸
        export_NX = static_cast<int>(final_vp.size());
        export_NZ = static_cast<int>(final_vp[0].size());
    }
    else {
        final_vp = cropped_vp;
        final_vs = cropped_vs;
        final_rho = cropped_rho;
    }

    // 4. 将 2D 物理网格展平为 1D 导出向量
    size_t             export_total_grid = static_cast<size_t>(export_NX) * export_NZ;
    std::vector<float> flat_vp(export_total_grid);
    std::vector<float> flat_vs(export_total_grid);
    std::vector<float> flat_rho(export_total_grid);

    for (int x = 0; x < export_NX; ++x) {
        for (int z = 0; z < export_NZ; ++z) {
            size_t k_segy = (size_t)x * export_NZ + z; // SEGY 要求的道内排列次序
            flat_vp[k_segy] = final_vp[x][z];
            flat_vs[k_segy] = final_vs[x][z];
            flat_rho[k_segy] = final_rho[x][z];
        }
    }

    // 5. 自动格式化文件路径并写出二进制文件
    std::string clean_path = base_filepath;
    size_t      dot_pos = clean_path.find_last_of('.');
    if (dot_pos != std::string::npos) {
        clean_path = clean_path.substr(0, dot_pos);
    }
    std::string out_vp_path = clean_path + "_vp.sgy";
    std::string out_vs_path = clean_path + "_vs.sgy";
    std::string out_rho_path = clean_path + "_rho.sgy";

    float dummy_dt = 0.001f; // 模型文件的标称垂直间隔
    SeismicIO::writeSegyFile2D(flat_vp, export_NX, export_NZ, out_vp_path, dummy_dt);
    SeismicIO::writeSegyFile2D(flat_vs, export_NX, export_NZ, out_vs_path, dummy_dt);
    SeismicIO::writeSegyFile2D(flat_rho, export_NX, export_NZ, out_rho_path, dummy_dt);

    std::cout << "[IO] Resampled SEGY Model Group Export Completed successfully." << std::endl;
    std::cout << "     - Final Output Size: " << export_NX << " traces x " << export_NZ << " samples" << std::endl;
    return true;
}

// =============================================================================
// 20. 标准二进制 SEG-Y 地震波记录文件数据加载器
// =============================================================================
inline bool LoadSeismicSegy(const std::string& filepath, AnalysisState& state) {
    // 1. 调用底层的标准 SEGY 2D 接口读取
    state.traces = SeismicIO::readSegyFile2D(filepath);
    if (state.traces.empty()) {
        return false;
    }

    state.numChannels = static_cast<int>(state.traces.size());
    state.numSamples = static_cast<int>(state.traces[0].size());

    // 2. 自动从二进制卷头的 Bytes 17-18 (即文件第 3216 字节) 提取采样间隔
    state.samplingInterval = 0.001f; // 默认防呆值 (1ms)
    std::ifstream file(filepath, std::ios::binary);

    if (file.is_open()) {
        file.seekg(3200 + 16, std::ios::beg); // 定位至 3216 字节
        uint16_t raw_dt = 0;
        file.read(reinterpret_cast<char*>(&raw_dt), 2);

        // 大端序转小端序
        uint16_t swapped_dt = ((raw_dt & 0xFF00) >> 8) | ((raw_dt & 0x00FF) << 8);
        if (swapped_dt > 0) {
            state.samplingInterval = static_cast<float>(swapped_dt) / 1000000.0f; // 微秒换算为秒
        }
        file.close();
    }

    // 3. 计算全局最大振幅用于 Wiggle/Heatmap 动态边界自适应
    state.globalMaxAmp = 0.0f;
    for (const auto& trace : state.traces) {
        for (float val : trace) {
            float abs_val = std::abs(val);
            if (abs_val > state.globalMaxAmp) {
                state.globalMaxAmp = abs_val;
            }
        }
    }
    return true;
}

// =============================================================================
// 【数据采集核心】：检波器单点位置
// =============================================================================
struct ReceiverPos {
    int x;
    int y;
};

// =============================================================================
// 【数据采集核心】：数据采集与道集录制管理器
// =============================================================================
struct AcquisitionManager {
    float samplingRateHz = 10000.0f; // 采样率
    float totalDurationSec = 0.5f;    // 录制总时长 (秒)
    float exportIntervalSec = 0.5f;
    int   numChannels = 100;     // 道数
    int   startX = 100;     // 起点 X 网格
    int   endX = 900;     // 终点 X 网格
    int   receiverDepth = 150;     // 排列所在深度 Z

    std::vector<ReceiverPos> receivers; // 检波器物理网格坐标数组
    bool isShowarray = true;            // 是否在波场上图层高亮显示检波器
    bool isRecording = false;           // 是否处于录制状态

    float totalTimer = 0.0f;      // 录制计时器 (秒)
    float exportTimer = 0.0f;
    float timer = 0.0f;
    int   currentFileIndex = 1;

    // 二维采集数据缓冲区 [channel][sample]
    std::vector<std::vector<float>> recorded_vx;
    std::vector<std::vector<float>> recorded_vz;

    // 初始化/清空采集缓冲区
    void initBuffer() {
        int expectedPoints = static_cast<int>(samplingRateHz * totalDurationSec);
        if (expectedPoints <= 0) expectedPoints = 1000;
        recorded_vx.assign(numChannels, std::vector<float>(expectedPoints, 0.0f));
        recorded_vz.assign(numChannels, std::vector<float>(expectedPoints, 0.0f));
    }
};

// 全局唯一的采集控制器实例
inline AcquisitionManager rec;

// =============================================================================
// 检波器排列数据同步至 GPU 显存的核心接口
// =============================================================================
inline void SyncReceiversToGPU() {
    if (rec.receivers.empty()) return;

    ctx.num_rcv = rec.numChannels;
    ctx.rcv_grid_idx.resize(ctx.num_rcv);
    for (int i = 0; i < ctx.num_rcv; ++i) {
        ctx.rcv_grid_idx[i] = rec.receivers[i].y * ctx.NX + rec.receivers[i].x;
    }

    // 重新在显卡端分配检波器索引及临时数据缓冲区，防止越界
    cudaFree(gpu_data.d_rcv_grid_idx);
    cudaMalloc(&gpu_data.d_rcv_grid_idx, ctx.num_rcv * sizeof(int));
    cudaMemcpy(gpu_data.d_rcv_grid_idx, ctx.rcv_grid_idx.data(), ctx.num_rcv * sizeof(int), cudaMemcpyHostToDevice);

    cudaFree(gpu_data.d_record_vx_step);
    cudaFree(gpu_data.d_record_vz_step);
    cudaMalloc(&gpu_data.d_record_vx_step, ctx.num_rcv * sizeof(float));
    cudaMalloc(&gpu_data.d_record_vz_step, ctx.num_rcv * sizeof(float));
}

// =============================================================================
// 数据采集一键重置与单点激发辅助驱动
// =============================================================================
inline void TriggerSingleShot(GLHandles& gl, SimState& state) {
    current_it = 0;
    state.running = true;
    accumulated_compute_time = 0.0f;
    active_sources.clear();

    freeGPUSimulation(gpu_data);
    initGPUSimulation(gpu_data, ctx);

    // 强制执行一次检波器显存同步，防止显存地址失效
    SyncReceiversToGPU();

    std::fill(ctx.record_vx.begin(), ctx.record_vx.end(), 0.0f);
    std::fill(ctx.record_vz.begin(), ctx.record_vz.end(), 0.0f);
}

// =============================================================================
// 将采集的数据导出为带有物理道头坐标的标准 SEG-Y 格式文件
// =============================================================================
inline void ExportToSegy(const AcquisitionManager& recorder, const std::string& filename) {
    if (recorder.recorded_vz.empty() || recorder.recorded_vz[0].empty()) return;

    std::vector<SeismicIO::TraceMetadata> metadata(recorder.numChannels);
    for (int i = 0; i < recorder.numChannels; ++i) {
        metadata[i].fieldRecordNum = recorder.currentFileIndex;
        metadata[i].traceInRecord = i + 1;
        metadata[i].sourceX = static_cast<int32_t>(edit_src_x * ctx.h); // 写入真实的物理炮点坐标 (米)
        metadata[i].sourceY = static_cast<int32_t>(edit_src_z * ctx.h);
        metadata[i].groupX = static_cast<int32_t>(recorder.receivers[i].x * ctx.h); // 写入检波点物理坐标
        metadata[i].groupY = static_cast<int32_t>(recorder.receivers[i].y * ctx.h);
    }

    float sample_interval_dt = 1.0f / recorder.samplingRateHz;
    // 调用上一节扩展的第三个重载函数
    SeismicIO::writeSegyFile2D(recorder.recorded_vz, metadata, filename, sample_interval_dt);
}

// CSV 简单格式备份导出
inline void ExportToCSV(const AcquisitionManager& recorder, const std::string& filename) {
    if (recorder.recorded_vz.empty()) return;
    std::ofstream file(filename);
    if (!file.is_open()) return;

    int samples = recorder.recorded_vz[0].size();
    for (int j = 0; j < samples; ++j) {
        for (int i = 0; i < recorder.numChannels; ++i) {
            file << recorder.recorded_vz[i][j];
            if (i < recorder.numChannels - 1) file << ",";
        }
        file << "\n";
    }
    file.close();
}

/// ////////////////////////////////////////////////////////////////////////第四部分

// =============================================================================
// 21. 惰性更新一维 Heatmap 矩阵的辅助函数
// =============================================================================
inline void UpdateHeatmapData(AnalysisState& state) {
    state.heatmapData.resize(state.numSamples * state.numChannels);
    for (int r = 0; r < state.numSamples; ++r) {
        for (int c = 0; c < state.numChannels; ++c) {
            // 将 [channel_idx][sample_idx] 映射为行优先矩阵，完美适配 ImPlot::PlotHeatmap
            state.heatmapData[r * state.numChannels + c] = state.traces[c][r];
        }
    }
}

// =============================================================================
// 22. 视口 LOD 高效回调提取器
// =============================================================================
inline ImPlotPoint SeismicDataGetter(int idx, void* data) {
    PlotContext* ctx = reinterpret_cast<PlotContext*>(data);
    int          sampleIdx = ctx->startSample + idx * ctx->step;
    float        val = (*ctx->traceData)[sampleIdx];

    // 振幅横向展开偏移与时间深度转化
    double x = ctx->offsetX + val * ctx->gain;
    double y = ctx->useTime ? (sampleIdx * ctx->dt) : (double)sampleIdx;

    return ImPlotPoint(x, y);
}

// 全局唯一的分析器状态实例
inline AnalysisState g_analyzerState;

// =============================================================================
// 23. 高保真 SEG-Y 探查数据分析器主窗口 (Seismic Data Analyzer Window)
// =============================================================================
void RenderAnalysisWindow(AnalysisState& state) {
    if (!state.isOpen) return;

    ImGui::SetNextWindowSize(ImVec2(1000, 700), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Seismic Data Analyzer", &state.isOpen)) {
        static bool useTimeAxis = true;

        // ------------------------------------------------------------
        // A. 顶部工具栏 (SEG-Y 加载与数据描述)
        // ------------------------------------------------------------
        if (ImGui::Button("Load SEG-Y Data")) {
            char filePath[1024] = { 0 };
            if (OpenSystemFileDialog(filePath, sizeof(filePath))) {
                // 一键载入标准的二进制 SEG-Y 数据组
                if (LoadSeismicSegy(filePath, state)) {
                    // 自动计算 Wiggle 增益与范围
                    if (state.globalMaxAmp > 1e-20f) {
                        state.displayGain = 0.8f / state.globalMaxAmp;
                        state.heatmapMin = -state.globalMaxAmp;
                        state.heatmapMax = state.globalMaxAmp;
                    }
                    else {
                        state.displayGain = 1.0f;
                    }
                    state.heatmapData.clear(); // 标记脏数据，触发重新拼装
                    state.fitRequest = true;   // 一键自适应缩放全屏
                }
            }
        }

        ImGui::SameLine();
        if (ImGui::Button("Reset View")) {
            state.fitRequest = true;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Reset axes to full data range.");
        }

        ImGui::SameLine();
        ImGui::TextDisabled("| %d Traces x %d Samples | MaxAmp: %.2e | dt: %.5f s",
            state.numChannels, state.numSamples, state.globalMaxAmp, state.samplingInterval);

        // ------------------------------------------------------------
        // B. 参数控制面板
        // ------------------------------------------------------------
        if (ImGui::CollapsingHeader("Display Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Columns(2, "viz_cols", false);

            // --- 左列：Wiggle 波形细线绘制设置 ---
            ImGui::TextDisabled("[ Wiggle Trace ]");
            ImGui::Checkbox("Show Wiggle", &state.showWiggle);
            if (state.showWiggle) {
                ImGui::SameLine();
                ImGui::Checkbox("Time Axis (Y)", &useTimeAxis);

                float baseGain = (state.globalMaxAmp > 0) ? (1.0f / state.globalMaxAmp) : 1.0f;
                float minGain = baseGain * 0.01f;
                float maxGain = baseGain * 1000.0f;

                ImGui::SetNextItemWidth(150);
                if (ImGui::DragFloat("Gain", &state.displayGain, baseGain * 0.01f, minGain, maxGain, "%.2e", ImGuiSliderFlags_Logarithmic)) {
                    if (state.displayGain < minGain) state.displayGain = minGain;
                }

                ImGui::SameLine();
                ImGui::ColorEdit4("Color", (float*)&state.colorLine, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel);
            }

            // --- 右列：Heatmap 密度着色背景设置 ---
            ImGui::NextColumn();
            ImGui::TextDisabled("[ Heatmap Background ]");
            ImGui::Checkbox("Show Heatmap", &state.showHeatmap);
            if (state.showHeatmap) {
                ImGui::SameLine();
                const char* cmaps[] = { "RdBu", "PiYG", "Spectral", "Greys" };
                ImGui::SetNextItemWidth(100);
                ImGui::Combo("##cmap", &state.colormapIndex, cmaps, IM_ARRAYSIZE(cmaps));

                ImGui::SetNextItemWidth(180);
                float dragSpeed = (state.globalMaxAmp > 0) ? state.globalMaxAmp * 0.01f : 0.01f;
                ImGui::DragFloatRange2("Range", &state.heatmapMin, &state.heatmapMax, dragSpeed, -FLT_MAX, FLT_MAX, "Min: %.2e", "Max: %.2e");
            }
            ImGui::Columns(1);

            // --- 只读物理采样率显示 ---
            ImGui::Separator();
            ImGui::Text("Physical Parameters:");
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.8f, 1.0f), "Sampling Interval (dt): %.5f s  (Auto-Parsed from SEGY Header)", state.samplingInterval);
        }

        ImGui::Separator();

        // ------------------------------------------------------------
        // C. 核心绘图展现区 (基于 ImPlot)
        // ------------------------------------------------------------
        if (state.numChannels > 0 && state.numSamples > 0) {
            if (state.showHeatmap && state.heatmapData.empty()) {
                UpdateHeatmapData(state);
            }

            // 经典学术白底绘图纸风格色板
            ImPlot::PushStyleColor(ImPlotCol_FrameBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
            ImPlot::PushStyleColor(ImPlotCol_PlotBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
            ImPlot::PushStyleColor(ImPlotCol_AxisGrid, ImVec4(0.9f, 0.9f, 0.9f, 1.0f));
            ImPlot::PushStyleColor(ImPlotCol_AxisText, ImVec4(0.2f, 0.2f, 0.2f, 1.0f));
            ImPlot::PushStyleColor(ImPlotCol_TitleText, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));

            if (ImPlot::BeginPlot("Seismic Record", ImVec2(-1, -1))) {
                double maxTime = state.numSamples * state.samplingInterval;
                double yEnd = useTimeAxis ? maxTime : (double)state.numSamples;

                const char* yLabel = useTimeAxis ? "Time (s)" : "Sample Index";
                ImPlot::SetupAxes("Trace Number", yLabel);
                ImPlot::SetupAxis(ImAxis_Y1, NULL, ImPlotAxisFlags_Invert); // 标杆：浅层在地表居顶对齐

                if (state.fitRequest) {
                    ImPlot::SetupAxisLimits(ImAxis_X1, -2.0, state.numChannels + 2.0, ImGuiCond_Always);
                    ImPlot::SetupAxisLimits(ImAxis_Y1, -yEnd * 0.05, yEnd * 1.05, ImGuiCond_Always);
                    state.fitRequest = false;
                }
                else {
                    ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, -10.0, state.numChannels + 10.0);
                    ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, -yEnd * 0.5, yEnd * 2.0);
                }

                // --- Layer 1: Heatmap 密度剖面背景 ---
                if (state.showHeatmap && !state.heatmapData.empty()) {
                    ImPlotColormap mapID = ImPlotColormap_RdBu;
                    if (state.colormapIndex == 1)      mapID = ImPlotColormap_PiYG;
                    else if (state.colormapIndex == 2) mapID = ImPlotColormap_Spectral;
                    else if (state.colormapIndex == 3) mapID = ImPlotColormap_Greys;

                    ImPlot::PushColormap(mapID);
                    ImPlot::PlotHeatmap("##HM",
                        state.heatmapData.data(),
                        state.numSamples,
                        state.numChannels,
                        state.heatmapMin, state.heatmapMax,
                        NULL,
                        { 0.5, yEnd },
                        { (double)state.numChannels + 0.5, 0.0 }
                    );
                    ImPlot::PopColormap();
                }

                // --- Layer 2: Wiggle Trace 地震道曲线 ---
                if (state.showWiggle) {
                    ImPlotRect limits = ImPlot::GetPlotLimits();
                    ImVec2     plotSize = ImPlot::GetPlotSize();

                    // 空间水平视口剔除，避免渲染视野外的道
                    int startCh = (int)floor(limits.X.Min) - 1;
                    int endCh = (int)ceil(limits.X.Max) + 1;
                    if (startCh < 0)                  startCh = 0;
                    if (endCh > state.numChannels) endCh = state.numChannels;

                    // 深度纵向视口剔除 (兼容 Invert 轴)
                    double valMin = (limits.Y.Min < limits.Y.Max) ? limits.Y.Min : limits.Y.Max;
                    double valMax = (limits.Y.Min > limits.Y.Max) ? limits.Y.Min : limits.Y.Max;

                    int startSample, endSample;
                    if (useTimeAxis) {
                        startSample = (int)(valMin / state.samplingInterval);
                        endSample = (int)(valMax / state.samplingInterval);
                    }
                    else {
                        startSample = (int)valMin;
                        endSample = (int)valMax;
                    }
                    if (startSample < 0)                 startSample = 0;
                    if (endSample > state.numSamples) endSample = state.numSamples;

                    if (startCh < endCh && startSample < endSample) {
                        // 动态 LOD 采样滤波，防止极高分辨率道图因像素过多导致抗锯齿失效/卡顿
                        double visibleSamples = endSample - startSample;
                        int    step = 1;
                        if (visibleSamples > 0 && plotSize.y > 0) {
                            double samplesPerPixel = visibleSamples / plotSize.y;
                            if (samplesPerPixel > 1.0) step = (int)samplesPerPixel;
                        }

                        PlotContext ctx;
                        ctx.gain = state.displayGain;
                        ctx.dt = state.samplingInterval;
                        ctx.useTime = useTimeAxis;
                        ctx.startSample = startSample;
                        ctx.step = step;
                        int pointsToDraw = (endSample - startSample) / step;

                        for (int i = startCh; i < endCh; ++i) {
                            ImPlot::SetNextLineStyle(state.colorLine);
                            ctx.traceData = &state.traces[i];
                            ctx.offsetX = (float)(i + 1);

                            ImGui::PushID(i);
                            if (pointsToDraw > 0) {
                                ImPlot::PlotLineG("##Trace", SeismicDataGetter, &ctx, pointsToDraw);
                            }
                            ImGui::PopID(); // <-- 修正：恢复为平衡的 PopID()
                        }
                    }
                }
                ImPlot::EndPlot();
            }
            ImPlot::PopStyleColor(5);
        }
        else {
            ImVec2 avail = ImGui::GetContentRegionAvail();
            ImGui::GetWindowDrawList()->AddText(
                ImVec2(ImGui::GetCursorScreenPos().x + avail.x / 2 - 80, ImGui::GetCursorScreenPos().y + avail.y / 2),
                IM_COL32(128, 128, 128, 255), "No SEG-Y Data Loaded"
            );
        }
    }
    ImGui::End();
}

// =============================================================================
// 24. 初始化显存与数值模拟物理网格资源
// =============================================================================
inline void InitializeSeismicSimulation(GLHandles& gl) {
    if (!is_seis_initialized) {
        edit_src_x = edit_w / 2;
        edit_src_z = edit_h / 4; // 震源默认深度设为 1/4

        par.model.xnum = edit_w;
        par.model.znum = edit_h;
        par.model.dx = 1.0f;
        par.model.dz = 1.0f;
        par.FDM.f0 = edit_f0;
        par.FDM.t0 = edit_t0;

        setupTestContext(ctx, par);
        initGPUSimulation(gpu_data, ctx);

        glGenTextures(1, &gl.seisTex);
        glBindTexture(GL_TEXTURE_2D, gl.seisTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, ctx.NX, ctx.NZ, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        cudaGraphicsGLRegisterImage(&gl.cudaSeisRes, gl.seisTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

        is_seis_initialized = true;
        first_align_needed = true;
        UpdateModelTexture();
    }
}

// =============================================================================
// 25. 物理标尺自适应平移相机矩阵对齐
// =============================================================================
inline void ApplyCameraAutoAlignment(int winW, int winH, float barHeight) {
    if (first_align_needed && winH > 0 && ctx.NX > 0) {
        float  winAspect = (float)winW / (float)winH;
        float  simAspect = (float)ctx.NX / (float)ctx.NZ;
        ImVec2 aspectCorr = { 1.0f, 1.0f };

        if (winAspect > simAspect) {
            aspectCorr.x = winAspect / simAspect;
        }
        else {
            aspectCorr.y = simAspect / winAspect;
        }

        viewOffset.x = aspectCorr.x - 1.0f;
        viewOffset.y = 1.0f - aspectCorr.y * (1.0f - 2.0f * barHeight / (float)winH);
        first_align_needed = false;
    }
}

// =============================================================================
// 26. 交互逻辑处理 (鼠标点击/拖拽，按键触发重置)
// =============================================================================
inline void HandleSeismicInteractions(SimState& state, int winW, int winH, float barHeight) {
    ImGuiIO& io = ImGui::GetIO();

    // A. 响应键盘 C 键一键清空与重置模拟信号
    if (g_resetSimRequested) {
        current_it = 0;
        state.running = false;
        accumulated_compute_time = 0.0f;
        active_sources.clear();
        ctx.dp_flat.assign(ctx.NX, 0.0f);

        int fs_idx = ctx.npml;
        for (int j = 0; j < ctx.NX; ++j) {
            int k = fs_idx * ctx.NX + j;
            if (k < ctx.total_grid && ctx.lambda2mu[k] > 0.0f) {
                float l2m = ctx.lambda2mu[k];
                float lam = ctx.lambda[k];
                ctx.dp_flat[j] = (l2m * l2m - lam * lam) / l2m;
            }
        }

        freeGPUSimulation(gpu_data);
        initGPUSimulation(gpu_data, ctx);
        UpdateModelTexture();
        std::fill(ctx.record_vx.begin(), ctx.record_vx.end(), 0.0f);
        std::fill(ctx.record_vz.begin(), ctx.record_vz.end(), 0.0f);
        g_resetSimRequested = false;
    }

    // B. 响应 R 键重置相机视角与缩放
    if (g_resetViewportRequested) {
        viewZoom = 1.0f;

        float  winAspect = (float)winW / (float)winH;
        float  simAspect = (float)ctx.NX / (float)ctx.NZ;
        ImVec2 aspectCorr = { 1.0f, 1.0f };

        if (winAspect > simAspect) {
            aspectCorr.x = winAspect / simAspect;
        }
        else {
            aspectCorr.y = simAspect / winAspect;
        }

        // 重新对齐左上角，确保视角完美复位
        viewOffset.x = aspectCorr.x - 1.0f;
        viewOffset.y = 1.0f - aspectCorr.y * (1.0f - 2.0f * barHeight / (float)winH);

        g_resetViewportRequested = false;
        active_sources.clear();
    }

    if (!io.WantCaptureKeyboard && ImGui::IsKeyPressed(ImGuiKey_Q, false)) {
        // 总共有 7 套物理分量 (Index 0 ~ 6)
        show_component = (show_component + 1) % 7;
    }
    // =============================================================================
    // F. 【新增】：响应键盘 W 键一键循环切换 13 套科学色谱波形样式
    // =============================================================================
    if (!io.WantCaptureKeyboard && ImGui::IsKeyPressed(ImGuiKey_W, false)) {
        // 总共有 13 套波场样式 (Index 0 ~ 12)
        waveStyle = (waveStyle + 1) % 13;
    }
    if (!io.WantCaptureKeyboard && ImGui::IsKeyPressed(ImGuiKey_E, false)) {
        // 总共有 5 套背景风格 (Index 0 ~ 4)
        modelStyle = (modelStyle + 1) % 5;
    }

    // 【核心新增】：键盘数字键 1 - 5 快速一键切换当前鼠标画笔工具模式
    if (!io.WantCaptureKeyboard) {
        if (ImGui::IsKeyPressed(ImGuiKey_1)) state.brushType = TOOL_NONE;   // 1 键：恢复默认单点震源
        if (ImGui::IsKeyPressed(ImGuiKey_2)) state.brushType = TOOL_HIGH;   // 2 键：切换为硬岩高速体刷
        if (ImGui::IsKeyPressed(ImGuiKey_3)) state.brushType = TOOL_LOW;    // 3 键：切换为松散层低速体刷
        if (ImGui::IsKeyPressed(ImGuiKey_4)) state.brushType = TOOL_WALL;   // 4 键：切换为自定义物性刷
        if (ImGui::IsKeyPressed(ImGuiKey_5)) state.brushType = TOOL_ERASER; // 5 键：切换为橡皮擦
    }

    // 计算屏幕缩放比例系数
    float  winAspect = (float)winW / (float)winH;
    float  simAspect = (float)ctx.NX / (float)ctx.NZ;
    ImVec2 aspectCorr = { 1.0f, 1.0f };
    if (winAspect > simAspect) aspectCorr.x = winAspect / simAspect;
    else                         aspectCorr.y = simAspect / winAspect;

    // C. 鼠标左键激发 (多震源连续或单震源瞬时注射)
    static double last_inject_time = 0.0;
    double        current_time = glfwGetTime();
    bool          trigger_active = false;

    if (multi_source_mode) {
        if (ImGui::IsMouseDown(0) && (current_time - last_inject_time >= 0.02)) {
            trigger_active = true;
            last_inject_time = current_time;
        }
    }
    else {
        if (ImGui::IsMouseClicked(0)) {
            trigger_active = true;
        }
    }

    if (!io.WantCaptureMouse) {
        float normX = io.MousePos.x / (float)winW;
        float normY = io.MousePos.y / (float)winH;

        float worldX = (normX - 0.5f) * (aspectCorr.x / viewZoom) + 0.5f + (0.5f * viewOffset.x);
        float worldY = (normY - 0.5f) * (aspectCorr.y / viewZoom) + 0.5f - (0.5f * viewOffset.y);

        int mx = (int)(worldX * (float)ctx.NX);
        int my = (int)(worldY * (float)ctx.NZ);

        if (state.brushType == TOOL_NONE) {
            // -----------------------------------------------------------------
            // 1. 【默认单点激发模式】：放置单点震源并重新计算
            // -----------------------------------------------------------------
            if (trigger_active) {
                int min_valid = ctx.npml + 0;
                int max_valid_x = ctx.NX - ctx.npml - 0;
                int max_valid_z = ctx.NZ - ctx.npml - 0;

                if (mx >= min_valid && mx <= max_valid_x && my >= min_valid && my <= max_valid_z) {
                    int clicked_idx = my * ctx.NX + mx;

                    if (multi_source_mode) {
                        GPUSource new_src;
                        new_src.idx = clicked_idx;
                        new_src.t = 0.0f;
                        new_src.f_peak = edit_f0;
                        new_src.amp = 1.0f;

                        if (active_sources.size() < 150) {
                            active_sources.push_back(new_src);
                        }
                        edit_src_x = mx;
                        edit_src_z = my;
                    }
                    else {
                        ctx.src_z_idx = my;
                        ctx.src_idx = clicked_idx;
                        edit_src_x = mx;
                        edit_src_z = my;
                        current_it = 0;
                        state.running = true;
                        accumulated_compute_time = 0.0f;

                        freeGPUSimulation(gpu_data);
                        initGPUSimulation(gpu_data, ctx);
                        std::fill(ctx.record_vx.begin(), ctx.record_vx.end(), 0.0f);
                        std::fill(ctx.record_vz.begin(), ctx.record_vz.end(), 0.0f);
                    }
                }
            }
        }
        else {
            // -----------------------------------------------------------------
            // 2. 【模型涂画编辑模式】：按住左键不放即可在当前位置执行物性涂抹
            // -----------------------------------------------------------------
            if (ImGui::IsMouseDown(0)) {
                int  r_cells = static_cast<int>(state.brushRadius);
                bool model_changed = false;

                // 2.1 设定不同画笔形态的目标物理参数
                float p_vp = 3000.0f, p_vs = 1732.0f, p_rho = 2200.0f;
                if (state.brushType == TOOL_HIGH) {
                    p_vp = 4800.0f; p_vs = 2800.0f; p_rho = 2500.0f; // 高速体 (硬岩)
                }
                else if (state.brushType == TOOL_LOW) {
                    p_vp = 1500.0f; p_vs = 800.0f;  p_rho = 1800.0f; // 低速体 (松散泥质/砂岩)
                }
                else if (state.brushType == TOOL_WALL) {
                    p_vp = edit_Vp; p_vs = edit_Vs;  p_rho = edit_Density; // 自定义物性笔 (与 UI 共享)
                }
                else if (state.brushType == TOOL_ERASER) {
                    p_vp = 3000.0f; p_vs = 1732.0f; p_rho = 2200.0f; // 橡皮擦：恢复为背景标准层
                }

                // 2.2 空间圆形邻域涂刷算法
                for (int dy = -r_cells; dy <= r_cells; ++dy) {
                    for (int dx = -r_cells; dx <= r_cells; ++dx) {
                        if (dx * dx + dy * dy <= r_cells * r_cells) {
                            int target_x = mx + dx;
                            int target_y = my + dy; // 屏幕深度 (0为地表, NZ-1为底部)

                            if (target_x >= 0 && target_x < ctx.NX && target_y >= 0 && target_y < ctx.NZ) {
                                // 【核心物理转换】：SetMaterialAt 内部 y=0 对应地底，
                                // 因此需在此处进行一次垂直对齐转换：y_cpu = NZ - 1 - y_screen
                                int cpu_y = ctx.NZ - 1 - target_y;
                                SetMaterialAt(target_x, cpu_y, p_vp, p_vs, p_rho);
                                model_changed = true;
                            }
                        }
                    }
                }

                // 2.3 触发高速显存同步与纹理重绘
                if (model_changed) {
                    UploadMaterialToGPU();
                }
            }
        }
        // =============================================================================
        // D. 鼠标中键拖拽平移与滚轮视口缩放 (修正版：引入 aspectCorr 彻底消除拖拽粘滞感)
        // =============================================================================
        if (!io.WantCaptureMouse) {
            // 1. 动态计算当前视口的自适应长宽比修正系数
            float winAspect = (float)winW / (float)winH;
            float simAspect = (float)ctx.NX / (float)ctx.NZ;
            ImVec2 aspectCorr = { 1.0f, 1.0f };
            if (winAspect > simAspect) {
                aspectCorr.x = winAspect / simAspect;
            }
            else {
                aspectCorr.y = simAspect / winAspect;
            }

            // 2. 滚轮无级缩放
            if (io.MouseWheel != 0) {
                float mouseSpeed = 0.1f * viewZoom;
                viewZoom += io.MouseWheel * mouseSpeed;
                if (viewZoom < 0.1f)   viewZoom = 0.1f;
                if (viewZoom > 100.0f) viewZoom = 100.0f;
            }

            // 3. 鼠标中键平移 (核心修复：乘上 aspectCorr 修正，使模型位移与鼠标位移实现 1:1 精确对齐)
            if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
                viewOffset.x -= 2.0f * io.MouseDelta.x * aspectCorr.x / (winW * viewZoom);
                viewOffset.y += 2.0f * io.MouseDelta.y * aspectCorr.y / (winH * viewZoom);
            }
        }

        ImVec2 mousePos = ImGui::GetMousePos();

        // E. 鼠标悬停位置逆向映射与物性监控
        is_mouse_inside = (mousePos.x >= 0.0f && mousePos.x < (float)winW &&
            mousePos.y >= barHeight && mousePos.y < (float)(winH - barHeight));

        if (io.WantCaptureMouse) {
            is_mouse_inside = false;
        }

        hover_valid = false;
        if (is_mouse_inside) {
            float normX = mousePos.x / (float)winW;
            float normY = mousePos.y / (float)winH;

            float worldX = (normX - 0.5f) * (aspectCorr.x / viewZoom) + 0.5f + (0.5f * viewOffset.x);
            float worldY = (normY - 0.5f) * (aspectCorr.y / viewZoom) + 0.5f - (0.5f * viewOffset.y);

            hover_mx = (int)(worldX * (float)ctx.NX);
            hover_my = (int)(worldY * (float)ctx.NZ);

            if (hover_mx >= 0 && hover_mx < ctx.NX && hover_my >= 0 && hover_my < ctx.NZ) {
                int k = hover_my * ctx.NX + hover_mx;
                hover_Rho = ctx.rho[k];
                if (hover_Rho > 0.0f) {
                    hover_Vp = sqrtf(ctx.lambda2mu[k] / hover_Rho);
                    hover_Vs = sqrtf(ctx.mu[k] / hover_Rho);
                    hover_valid = true;
                }
            }
        }
    }
}

/// ////////////////////////////////////////////////////////////////////////第五部分
// =============================================================================
// 27. 执行 CUDA 有限差分时演更新 (CUDA Finite-Difference Wavefield Update)
// =============================================================================
inline void UpdateWavefieldSimulation(SimState& state) {
    ImGuiIO& io = ImGui::GetIO();
    bool     can_run = state.running && (infinite_mode || current_it < ctx.nt);

    if (can_run) {
        accumulated_compute_time += io.DeltaTime;

        for (int s = 0; s < steps_per_frame; ++s) {
            if (infinite_mode || current_it < ctx.nt) {
                // A. 多点激发寿命更新与自动消亡
                if (multi_source_mode && !active_sources.empty()) {
                    for (auto it = active_sources.begin(); it != active_sources.end(); ) {
                        it->t += ctx.dt;
                        float t_peak = 1.0f / it->f_peak;
                        if (it->t > 2.5f * t_peak) {
                            it = active_sources.erase(it);
                        }
                        else {
                            ++it;
                        }
                    }
                    if (!active_sources.empty()) {
                        cudaMemcpy(gpu_data.d_active_sources, active_sources.data(),
                            active_sources.size() * sizeof(GPUSource), cudaMemcpyHostToDevice);
                    }
                }

                // B. 执行 GPU 计算核函数推进
                runGPUStep(gpu_data, current_it, ctx, multi_source_mode ? (int)active_sources.size() : -1);

                // =====================================================================
                // 【核心新增】：高保真地震数据采集采样逻辑
                // =====================================================================
                if (rec.isRecording) {
                    // 计算时演步长与目标采集采样率的换算因子
                    int steps_per_sample = static_cast<int>(round((1.0f / rec.samplingRateHz) / ctx.dt));
                    if (steps_per_sample <= 0) steps_per_sample = 1;

                    if (current_it % steps_per_sample == 0) {
                        int sample_idx = current_it / steps_per_sample;
                        int max_samples = rec.recorded_vz[0].size();

                        if (sample_idx < max_samples) {
                            // 1. 调用 GPU 高并发提取核函数，提取当前时间切片上所有接收器的振幅
                            // 1. 创建 CPU 临时缓冲区，并调用 GPU 包装接口提取当前时刻所有接收器的 Vx 和 Vz 振幅
                            std::vector<float> h_step_vx(rec.numChannels);
                            std::vector<float> h_step_vz(rec.numChannels);
                            recordReceiverStepGPU(gpu_data, rec.numChannels, h_step_vx.data(), h_step_vz.data());
                            // 3. 填入录制缓冲区对应道的 Sample 索引中
                            for (int c = 0; c < rec.numChannels; ++c) {
                                rec.recorded_vx[c][sample_idx] = h_step_vx[c];
                                rec.recorded_vz[c][sample_idx] = h_step_vz[c];
                            }
                        }
                    }

                    // 更新录制物理时间，达到设定时长后自动结束并写出物理记录文件
                    rec.totalTimer = current_it * ctx.dt;
                    if (rec.totalTimer >= rec.totalDurationSec) {
                        rec.isRecording = false;
                        state.running = false;

                        // 自动写出到标准 SEG-Y 文件中
                        ExportToSegy(rec, "auto_export_gather.sgy");
                        popup_message = "Simulation recording completed!\nExported to 'auto_export_gather.sgy'";
                        show_success_popup = true;
                    }
                }

                current_it++;
            }
        }
    }
}

// =============================================================================
// 28. OpenGL 渲染全屏背景波场纹理
// =============================================================================
inline void RenderFullBackbufferWavefield(int winW, int winH, float barHeight, GLHandles& gl) {
    // 1. 显存色彩映射与 2D copy 写入纹理
    cudaGraphicsMapResources(1, &gl.cudaSeisRes, 0);
    cudaGraphicsSubResourceGetMappedArray(&gl.cudaSeisArray, gl.cudaSeisRes, 0, 0);

    static uchar4* d_rgba_out = nullptr;
    static int     last_size = 0;

    if (last_size != ctx.total_grid) {
        if (d_rgba_out) cudaFree(d_rgba_out);
        cudaMalloc(&d_rgba_out, ctx.total_grid * sizeof(uchar4));
        last_size = ctx.total_grid;
    }
    generateWavefieldTextureCUDA(gpu_data, d_rgba_out, color_scale, show_component);

    cudaMemcpy2DToArray(
        gl.cudaSeisArray, 0, 0,
        d_rgba_out,
        ctx.NX * sizeof(uchar4),
        ctx.NX * sizeof(uchar4),
        ctx.NZ,
        cudaMemcpyDeviceToDevice
    );
    cudaGraphicsUnmapResources(1, &gl.cudaSeisRes, 0);

    // 2. 着色器渲染管线
    glViewport(0, 0, winW, winH);
    glUseProgram(gl.seisProg);

    // 绑定并载入地震波场纹理到单元 0
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, gl.seisTex);
    glUniform1i(glGetUniformLocation(gl.seisProg, "seisTexture"), 0);

    // 激活单元 1 并绑定离屏地质层位物性纹理
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, g_modelTex);
    glUniform1i(glGetUniformLocation(gl.seisProg, "modelTexture"), 1);

    // 传递地震及地质背景相关的 Uniform 变量
    glUniform1i(glGetUniformLocation(gl.seisProg, "waveStyle"), waveStyle);
    glUniform1i(glGetUniformLocation(gl.seisProg, "modelStyle"), modelStyle); // 传递背景风格
    glUniform1i(glGetUniformLocation(gl.seisProg, "showGrid"), showGrid ? 1 : 0); // 网格开关状态

    glUniform1f(glGetUniformLocation(gl.seisProg, "gridSpacing"), ui_gridSpacing);
    glUniform1f(glGetUniformLocation(gl.seisProg, "gridOpacity"), ui_gridOpacity);

    // 无缝向下传递当前滑块设置的 Vp 和 密度 (Rho)，建立物理场关联
    glUniform1f(glGetUniformLocation(gl.seisProg, "uniformVp"), edit_Vp);
    glUniform1f(glGetUniformLocation(gl.seisProg, "uniformRho"), edit_Density);
    glUniform1i(glGetUniformLocation(gl.seisProg, "useModelTexture"), 1); // 启用非均匀速度模型纹理

    // 绑定常规参数 Uniform 变量
    glUniform2f(glGetUniformLocation(gl.seisProg, "winSize"), (float)winW, (float)winH);
    glUniform2f(glGetUniformLocation(gl.seisProg, "simSize"), (float)ctx.NX, (float)ctx.NZ);
    glUniform2f(glGetUniformLocation(gl.seisProg, "viewOffset"), viewOffset.x, viewOffset.y);
    glUniform1f(glGetUniformLocation(gl.seisProg, "viewZoom"), viewZoom);
    glUniform1f(glGetUniformLocation(gl.seisProg, "totalTime"), (float)glfwGetTime());
    glUniform1f(glGetUniformLocation(gl.seisProg, "npml"), (float)ctx.npml);

    // 再次绑定并载入纹理
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, gl.seisTex);
    glUniform1i(glGetUniformLocation(gl.seisProg, "seisTexture"), 0);

    glBindVertexArray(gl.quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}

/// ////////////////////////////////////////////////////////////////////////第六部分
// =============================================================================
// 29. 升级版：地震模拟多维图形用户界面与 HUD 交互框架 (ImGui / ImPlot)
// =============================================================================
inline void RenderSeisHUD(SimState& state, int winW, int winH, float barHeight, float scale, GLHandles& gl, const GpuInfo& info) {
    ImGuiIO& io = ImGui::GetIO();

    // A. 绘制主缓冲区自适应标尺 (在波场之上，ImGui 控制面板之下)
    RenderGridRulerOnBackbuffer(ctx, state, winW, winH, barHeight, viewOffset, viewZoom);

    // B. 绘制高亮科技鼠标光标 (Fg 图层)
    ViewportInfo vp;
    vp.x = 0.0f;
    vp.y = 0.0f;
    vp.w = (float)winW;
    vp.h = (float)winH;
    vp.scaleX = vp.w / (float)ctx.NX;
    vp.scaleY = vp.h / (float)ctx.NZ;
    RenderBrushCursor(state, vp);

    // C. 绘制当前编辑/激发的物理震源绿色/橙色准星
    RenderSourceMarker(ctx, edit_src_x, edit_src_z, winW, winH, barHeight, viewOffset, viewZoom);

    // --- 【新增：绘制排列图层】 ---
    if (rec.isShowarray && !rec.receivers.empty()) {
        ImDrawList* fg = ImGui::GetForegroundDrawList();

        float  winAspect = (float)winW / (float)winH;
        float  simAspect = (float)ctx.NX / (float)ctx.NZ;
        ImVec2 aspectCorr = { 1.0f, 1.0f };
        if (winAspect > simAspect) {
            aspectCorr.x = winAspect / simAspect;
        }
        else {
            aspectCorr.y = simAspect / winAspect;
        }

        for (size_t i = 0; i < rec.receivers.size(); ++i) {
            float aPos_x = ((float)rec.receivers[i].x / (float)ctx.NX) * 2.0f - 1.0f;
            float aPos_y = 1.0f - ((float)rec.receivers[i].y / (float)ctx.NZ) * 2.0f;

            float Pndc_x = (aPos_x - viewOffset.x) * viewZoom / aspectCorr.x;
            float screenX = (Pndc_x * 0.5f + 0.5f) * winW;

            float Pndc_y = (aPos_y - viewOffset.y) * viewZoom / aspectCorr.y;
            float screenY = (0.5f - Pndc_y * 0.5f) * winH;

            if (screenX >= 0.0f && screenX <= winW && screenY >= barHeight && screenY <= winH) {
                // 采用学术界通用的黄色倒三角形高亮标志检波器
                ImVec2 p1(screenX, screenY - 5.0f);
                ImVec2 p2(screenX - 4.0f, screenY + 3.0f);
                ImVec2 p3(screenX + 4.0f, screenY + 3.0f);
                fg->AddTriangleFilled(p1, p2, p3, IM_COL32(255, 230, 51, 230)); // 荧光黄
                fg->AddTriangle(p1, p2, p3, IM_COL32(0, 0, 0, 255), 1.0f);      // 描黑边
            }
        }
    }

    // -------------------------------------------------------------------------
    // D. 渲染顶部 TopBar 状态遥测栏
    // -------------------------------------------------------------------------
    {
        ImGui::SetNextWindowPos({ 0, 0 });
        ImGui::SetNextWindowSize({ (float)winW, barHeight });

        ImVec4 topBarBg = ImVec4(0.95f, 1.0f, 0.98f, 0.2f);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, topBarBg);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10 * scale, 0));

        ImGui::PushStyleColor(ImGuiCol_CheckMark, uiAccent);
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.0f, 0.0f, 0.0f, 0.35f));
        ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(uiAccent.x * 0.15f, uiAccent.y * 0.15f, uiAccent.z * 0.15f, 0.4f));
        ImGui::PushStyleColor(ImGuiCol_FrameBgActive, ImVec4(uiAccent.x * 0.3f, uiAccent.y * 0.3f, uiAccent.z * 0.3f, 0.6f));

        ImGui::Begin("TopBar_Seis", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar);
        {
            float centerY = 14.0f * scale;

            ImGui::SetCursorPos({ 20 * scale, centerY });
            ImGui::TextDisabled("ACTIVE GPU:"); ImGui::SameLine();
            ImGui::TextColored(uiAccent, "%s", info.name);

            float physicalTime = current_it * ctx.dt;

            char computeTimeStr[64];
            if (accumulated_compute_time < 60.0f) {
                sprintf(computeTimeStr, "%.6f s", accumulated_compute_time);
            }
            else {
                int   minutes = (int)(accumulated_compute_time) / 60;
                float seconds = fmodf(accumulated_compute_time, 60.0f);
                sprintf(computeTimeStr, "%02dm %.2fs", minutes, seconds);
            }

            char stepStr[128];
            if (infinite_mode) {
                sprintf(stepStr, "%d / INF", current_it);
            }
            else {
                sprintf(stepStr, "%d / %d", current_it, ctx.nt);
            }

            char statusStr[512];
            sprintf(statusStr, "FORMULATION: TYPE %d  |  STEP: %s  |  PHYS TIME: %.4f s  |  COMPUTE TIME: %s",
                ctx.flag_type, stepStr, physicalTime, computeTimeStr);

            float textWidth = ImGui::CalcTextSize(statusStr).x;
            ImGui::SetCursorPos({ (winW - textWidth) * 0.5f, centerY });
            ImGui::TextColored(uiAccent, "%s", statusStr);

            ImGui::SetCursorPos({ (float)winW - 260.0f * scale, centerY - 2.0f * scale });

            static float displayFPS = 60.0f;
            displayFPS = displayFPS * 0.95f + io.Framerate * 0.05f;
            ImU32 fpsColor = IM_COL32(uiAccent.x * 255, uiAccent.y * 255, uiAccent.z * 255, 220);
            ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(fpsColor), "[ %.0f FPS ]", displayFPS);

            ImGui::SameLine(0, 20 * scale);
            ImGui::SetCursorPosY(centerY - 5.0f * scale);
            ImGui::Checkbox("OSD_PANEL", &showHUD);
        }
        ImGui::End();
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(5);

        ImDrawList* topDraw = ImGui::GetForegroundDrawList();
        topDraw->AddLine(
            { 0, barHeight },
            { (float)winW, barHeight },
            IM_COL32(uiAccent.x * 255, uiAccent.y * 255, uiAccent.z * 255, 60),
            1.0f
        );
    }

    // -------------------------------------------------------------------------
    // E. 渲染左侧悬浮控制面板 (LAB_CONTROLS)
    // -------------------------------------------------------------------------
    if (showHUD) {
        ImGui::SetNextWindowPos({ 30 * scale, 80 * scale }, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize({ 420 * scale, 780 * scale }, ImGuiCond_FirstUseEver);

        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.95f, 1.0f, 0.98f, 0.3f));
        ImGui::PushStyleColor(ImGuiCol_Border, uiAccent);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.5f);

        ImGui::PushStyleColor(ImGuiCol_SliderGrab, uiAccent);
        ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, ImVec4(uiAccent.x * 1.2f, uiAccent.y * 1.2f, uiAccent.z * 1.2f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_TabActive, uiAccent);
        ImGui::PushStyleColor(ImGuiCol_TabHovered, ImVec4(uiAccent.x * 0.8f, uiAccent.y * 0.8f, uiAccent.z * 0.8f, 0.8f));
        ImGui::PushStyleColor(ImGuiCol_HeaderActive, uiAccent);
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(uiAccent.x * 0.8f, uiAccent.y * 0.8f, uiAccent.z * 0.8f, 0.8f));

        if (ImGui::Begin("Control Panel", &showHUD)) {
            static float displayFPS = 60.0f;
            displayFPS = displayFPS * 0.95f + io.Framerate * 0.05f;
            ImGui::TextColored(uiAccent, "FPS: %.1f", displayFPS);
            ImGui::Separator();

            if (ImGui::BeginTabBar("ControlTabs")) {

                // ==========================================
                // Tab 1: Simulation 运行设置
                // ==========================================
                if (ImGui::BeginTabItem("Simulation")) {
                    ImGui::Spacing();
                    ImGui::TextColored(uiAccent, "Simulation Protocol");
                    const char* sim_types[] = { "Type 1: Whole-domain PML", "Type 2: Interior FDM + Boundary PML", "Type 3: Free Surface + PML" };
                    int temp_type = ctx.flag_type - 1;
                    if (ImGui::Combo("Formulation", &temp_type, sim_types, IM_ARRAYSIZE(sim_types))) {
                        ctx.flag_type = temp_type + 1;
                        g_resetSimRequested = true;
                    }
                    ImGui::Spacing();

                    ImGui::TextColored(uiAccent, "Time Step, Grid Spacing & Iteration Control");
                    float max_vp = 0.0f;
                    for (int k = 0; k < ctx.total_grid; ++k) {
                        float vp = sqrtf(ctx.lambda2mu[k] / ctx.rho[k]);
                        if (vp > max_vp) max_vp = vp;
                    }

                    ImGui::SliderFloat("Time Step (dt)", &temp_dt, 0.00001f, 0.003f, "%.6f s");
                    ImGui::SliderFloat("Grid Spacing (dx)", &temp_dx, 0.1f, 20.0f, "%.2f m");
                    // =========================================================================
                    // 【优化】：开放 PML 宽度下限至 0，实现一键无缝切换至“物理全反射模式”
                    // =========================================================================
                    if (temp_pml == 0) {
                        ImGui::SliderInt("PML Layer Width", &temp_pml, 0, 100, "0 (Total Reflection)");
                    }
                    else {
                        ImGui::SliderInt("PML Layer Width", &temp_pml, 0, 100, "%d px");
                    }
                    ImGui::SliderInt("Max Steps (nt)", &temp_nt, 500, 50000);
                    ImGui::SliderInt("Steps / Frame", &steps_per_frame, 1, 100);

                    // =========================================================================
                    // 【重构】：双库朗数（CFL）多维物理稳定性分析系统
                    // =========================================================================
                    // 1. 理论绝对失稳极限值（CFL 理论库朗数 = 0.601）
                    float cfl_theory = 0.601f;
                    float dt_theory_limit = cfl_theory * temp_dx / max_vp;

                    // 2. 实际应用处理后的安全限制（CFL 安全库朗数 = 0.480，留出 20% 余量防 PML 边界发散）
                    float cfl_safe = 0.480f;
                    float dt_safe_limit = cfl_safe * temp_dx / max_vp;

                    ImGui::TextDisabled("CFL Stability Analysis:");

                    // 理论极限显示
                    ImGui::Text("- Theoretical Limit (CFL = %.3f):", cfl_theory); ImGui::SameLine();
                    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%.6f s", dt_theory_limit);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Absolute mathematical threshold.\nGoing above this will cause instant numerical divergence (NaN).");
                    }

                    // 实际处理后（安全）限制显示
                    ImGui::Text("- Processed Safe Limit (CFL = %.3f):", cfl_safe); ImGui::SameLine();
                    ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.8f, 1.0f), "%.6f s", dt_safe_limit);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Engineering standard threshold with a 20%% safety margin.\nThis coefficient is practically applied to absorb PML boundary stiffness safely.");
                    }

                    ImGui::Text("- Current Draft Step (dt): %.6f s", temp_dt);

                    // 3. 三色高阶稳定状态监控指示灯
                    if (temp_dt <= dt_safe_limit) {
                        // 处于安全区域
                        ImGui::TextColored({ 0.2f, 1.0f, 0.4f, 1.0f }, "[ STATUS: CFL STABLE (Highly Safe) ]");
                    }
                    else if (temp_dt <= dt_theory_limit) {
                        // 处于临界高风险区域
                        ImGui::TextColored({ 1.0f, 0.62f, 0.0f, 1.0f }, "[ STATUS: CFL MARGINAL (Risk of PML boundary explosion!) ]");
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Warning: The requested dt is mathematically stable for homogeneous grids,\nbut may diverge inside PML damping layers or free surfaces.");
                        }
                    }
                    else {
                        // 已经越过生死线，必定溢出发散
                        ImGui::TextColored({ 1.0f, 0.2f, 0.2f, 1.0f }, "[ STATUS: CFL DIVERGENT (Will blow up!) ]");
                    }

                    ImGui::Spacing();
                    if (ImGui::Button("APPLY TIME & SPACING CONTROLS", { -1, 30 * scale })) {
                        // 1. 提取当前物理最大波速，用于安全边界钳位检测
                        float max_vp = 0.0f;
                        for (int k = 0; k < ctx.total_grid; ++k) {
                            float vp = sqrtf(ctx.lambda2mu[k] / ctx.rho[k]);
                            if (vp > max_vp) max_vp = vp;
                        }
                        if (max_vp <= 350.0f) max_vp = 2000.0f;

                        // 2. 算理论生死限 (CFL = 0.601)
                        float dt_theory_limit = 0.601f * temp_dx / max_vp;

                        // 3. 【安全保护】：若用户输入的 dt 大于理论极限，强制将其钳位重置为 0.480 的安全值
                        if (temp_dt > dt_theory_limit) {
                            temp_dt = 0.480f * temp_dx / max_vp;
                            popup_message = "Warning: Input dt exceeded CFL stability limit.\nAutomatically clamped to safe value.";
                            show_error_popup = true;
                        }

                        // 4. 应用经过安全拦截的步长与空间参数
                        ctx.dt = temp_dt;
                        gpu_data.inv_dt = 1.0f / ctx.dt;
                        ctx.nt = temp_nt;
                        ctx.h = temp_dx;

                        par.model.dx = temp_dx;
                        par.model.dz = temp_dx;
                        ctx.npml = temp_pml;
                        par.FDM.npml = (float)temp_pml;

                        // 5. 极简化差分系数更新
                        ctx.c1_h = 1.125022f / temp_dx;
                        ctx.c2_h = -0.04687594f / temp_dx;
                        ctx.c3_h = 0.00416669f / temp_dx;
                        ctx.c4_h = -0.00019234f / temp_dx;

                        // 6. 重新执行高能效阻尼剖面重算与自由表面自适应对齐
                        UpdatePmlDampingArrays();

                        ctx.dp_flat.assign(ctx.NX, 0.0f);
                        int fs_idx = ctx.npml;
                        for (int j = 0; j < ctx.NX; ++j) {
                            int k = fs_idx * ctx.NX + j;
                            if (k < ctx.total_grid && ctx.lambda2mu[k] > 0.0f) {
                                float l2m = ctx.lambda2mu[k];
                                float lam = ctx.lambda[k];
                                ctx.dp_flat[j] = (l2m * l2m - lam * lam) / l2m;
                            }
                        }

                        // 7. 重新分管道记录表并重绘震源子波
                        ctx.record_vx.assign(ctx.num_rcv * ctx.nt, 0.0f);
                        ctx.record_vz.assign(ctx.num_rcv * ctx.nt, 0.0f);
                        generateRickerWavelet(ctx.wavelet, ctx.nt, ctx.dt, par.FDM.f0, par.FDM.t0);

                        g_resetSimRequested = true;
                    }
                    ImGui::Spacing();

                    ImGui::Separator();
                    ImGui::TextColored(uiAccent, "GRID DIMENSIONS (NX, NZ)");
                    ImGui::PushItemWidth(100 * scale);
                    ImGui::InputInt("##WidthVal", &edit_w, 0, 0); ImGui::SameLine();
                    ImGui::PopItemWidth();
                    if (ImGui::Button("-##WDec")) { edit_w = std::max(64, edit_w - 64); } ImGui::SameLine();
                    if (ImGui::Button("+##WInc")) { edit_w = std::min(8192, edit_w + 64); } ImGui::SameLine();
                    ImGui::Text("Grid Width (NX)");
                    ImGui::Spacing();

                    ImGui::PushItemWidth(100 * scale);
                    ImGui::InputInt("##HeightVal", &edit_h, 0, 0); ImGui::SameLine();
                    ImGui::PopItemWidth();
                    if (ImGui::Button("-##HDec")) { edit_h = std::max(64, edit_h - 64); } ImGui::SameLine();
                    if (ImGui::Button("+##HInc")) { edit_h = std::min(8192, edit_h + 64); } ImGui::SameLine();
                    ImGui::Text("Grid Height (NZ)");

                    ImGui::Spacing();
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.75f, 0.15f, 0.15f, 0.85f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.9f, 0.2f, 0.2f, 1.0f));
                    if (ImGui::Button("APPLY NEW GRID SIZE & CLEAR MODEL", { -1, 35 * scale })) {
                        state.running = false;
                        current_it = 0;
                        accumulated_compute_time = 0.0f;
                        active_sources.clear();

                        par.model.xnum = edit_w;
                        par.model.znum = edit_h;

                        setupTestContext(ctx, par);
                        freeGPUSimulation(gpu_data);

                        cudaGraphicsUnregisterResource(gl.cudaSeisRes);
                        glBindTexture(GL_TEXTURE_2D, gl.seisTex);
                        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, ctx.NX, ctx.NZ, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
                        cudaGraphicsGLRegisterImage(&gl.cudaSeisRes, gl.seisTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

                        initGPUSimulation(gpu_data, ctx);
                        first_align_needed = true;
                    }
                    ImGui::PopStyleColor(2);
                    ImGui::Spacing();
                    ImGui::Separator();

                    ImGui::Checkbox("V-Sync Control", &state.vsyncEnabled);
                    if (ImGui::IsItemDeactivatedAfterEdit()) glfwSwapInterval(state.vsyncEnabled ? 1 : 0);
                    ImGui::Spacing();

                    ImGui::Checkbox("Continuous Simulation (Infinite Mode)", &infinite_mode);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Check to run the wavefield tank indefinitely.\nIdeal for interactive sandbox clicking.");
                    }
                    ImGui::Spacing();

                    ImGui::Checkbox("Enable Multi-Source Real-time Clicks", &multi_source_mode);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Check to click and spawn multiple intersecting waves simultaneously\nUncheck for standard single-point reset mode.");
                    }
                    ImGui::Checkbox("Enable Show Monitor par", &show_Monitor_par);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Show additional diagnostic window for PML parameters and geophysical range.");
                    }

                    ImGui::Separator();
                    ImGui::TextColored(uiAccent, "EXECUTION CONTROLS");

                    if (state.running) {
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.85f, 0.6f, 0.1f, 0.85f));
                        if (ImGui::Button("PAUSE SIMULATION", { -1, 30 * scale })) {
                            state.running = false;
                        }
                        ImGui::PopStyleColor();
                    }
                    else {
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.65f, 0.3f, 0.85f));
                        if (ImGui::Button("RUN SIMULATION", { -1, 30 * scale })) {
                            state.running = true;
                        }
                        ImGui::PopStyleColor();
                    }

                    if (ImGui::Button("RESET SIMULATION", { -1, 28 * scale })) {
                        g_resetSimRequested = true;
                    }

                    if (ImGui::Button("RESET VIEWPORT", { -1, 28 * scale })) {
                        g_resetViewportRequested = true;
                    }

                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(uiAccent.x * 0.12f, uiAccent.y * 0.32f, uiAccent.z * 0.42f, 0.8f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(uiAccent.x * 0.18f, uiAccent.y * 0.52f, uiAccent.z * 0.62f, 1.0f));
                    if (ImGui::Button("OPEN DATA ANALYZER", { -1, 28 * scale })) {
                        g_analyzerState.isOpen = true; // 打开数据分析剖面视窗
                    }
                    ImGui::PopStyleColor(2);
                    ImGui::EndTabItem();
                }

                // ==========================================
                // Tab 2: Source 激发震源与角度调节
                // ==========================================
                if (ImGui::BeginTabItem("Source")) {
                    ImGui::Spacing();
                    ImGui::TextColored(uiAccent, "Source Position, Type & Amplitude");

                    int min_valid_grid = ctx.npml + 5;
                    int max_valid_x = ctx.NX - ctx.npml - 5;
                    int max_valid_z = ctx.NZ - ctx.npml - 5;

                    // 1. 坐标位置调节
                    ImGui::SliderInt("Source X (Grid)", &edit_src_x, min_valid_grid, max_valid_x);
                    ImGui::SliderInt("Source Z (Grid)", &edit_src_z, min_valid_grid, max_valid_z);
                    ImGui::Spacing();

                    // 2. 新增：激发幅值调节滑动条
                    ImGui::SliderFloat("Excitation Amp", &edit_amp, 0.1f, 10.0f, "%.1f x");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Adjust the source intensity in real-time.");
                    }

                    // 3. 新增：物理震源激发类型选择 Combo (与您给出的图片完美对应)
                    const char* source_types[] = {
                        "Vertical Force (Tzz)",
                        "Horizontal Force (Txx)",
                        "Explosive (Txx + Tzz)",
                        "Shear (Txz)",
                        "Rotated Force"
                    };
                    ImGui::Combo("Source Type", &edit_source_type, source_types, IM_ARRAYSIZE(source_types));
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Select seismic source mechanism (e.g. Pure P-Wave or Shear S-Wave).");
                    }

                    // 4. 优化：仅在选择“倾斜力 (Rotated Force)”时，才显示受力角度调节，保持面板整洁
                    if (edit_source_type == 4) {
                        ImGui::SliderFloat("Force Angle", &edit_angle, -180.0f, 180.0f, "%.1f deg");
                    }

                    ImGui::Separator();
                    ImGui::TextColored(uiAccent, "Ricker Wavelet Properties");
                    ImGui::SliderFloat("Peak Freq (f0)", &edit_f0, 5.0f, 300.0f, "%.1f Hz");
                    ImGui::SliderFloat("Time Delay (t0)", &edit_t0, 0.001f, 0.2f, "%.3f s");

                    ImGui::TextDisabled("Waveform Real-Time Preview:");
                    float              preview_dt = 0.0005f;
                    std::vector<float> preview_x(200);
                    std::vector<float> preview_y(200);
                    for (int i = 0; i < 200; ++i) {
                        float t = i * preview_dt - edit_t0;
                        float pi2_f0 = M_PI * M_PI * edit_f0 * edit_f0;
                        preview_x[i] = i * preview_dt;
                        preview_y[i] = (1.0f - 2.0f * pi2_f0 * t * t) * expf(-pi2_f0 * t * t) * edit_amp; // 预览同样应用幅值
                    }

                    ImPlot::PushStyleColor(ImPlotCol_FrameBg, IM_COL32(125, 125, 125, 255));
                    ImPlot::PushStyleColor(ImPlotCol_PlotBg, IM_COL32(125, 125, 125, 255));
                    ImPlot::PushStyleColor(ImPlotCol_Line, IM_COL32(0, 255, 120, 255));
                    ImPlot::PushStyleColor(ImPlotCol_AxisGrid, IM_COL32(35, 35, 35, 120));

                    if (ImPlot::BeginPlot("##WaveletPreview", ImVec2(-1, 95 * scale), ImPlotFlags_NoLegend)) {
                        ImPlot::SetupAxes("Time", "Amp", ImPlotAxisFlags_NoTickLabels, ImPlotAxisFlags_NoTickLabels);
                        ImPlot::SetupAxesLimits(0.0f, 200 * preview_dt, -1.1f * edit_amp, 1.1f * edit_amp);
                        ImPlot::PlotLine("Ricker", preview_x.data(), preview_y.data(), 200);
                        ImPlot::EndPlot();
                    }
                    ImPlot::PopStyleColor(4);

                    ImGui::Spacing();
                    if (ImGui::Button("APPLY SOURCE CONFIG", { -1, 30 * scale })) {
                        ctx.src_z_idx = edit_src_z;
                        ctx.src_idx = edit_src_z * ctx.NX + edit_src_x;
                        ctx.src_angle = edit_angle;
                        ctx.src_type = edit_source_type; // 更新激发类型
                        ctx.src_amp = edit_amp;         // 更新激发幅值

                        par.FDM.f0 = edit_f0;
                        par.FDM.t0 = edit_t0;
                        par.FDM.angle = edit_angle;

                        generateRickerWavelet(ctx.wavelet, ctx.nt, ctx.dt, edit_f0, edit_t0);
                        current_it = 0;
                        state.running = false;
                        accumulated_compute_time = 0.0f;

                        freeGPUSimulation(gpu_data);
                        initGPUSimulation(gpu_data, ctx);
                    }
                    ImGui::Spacing();
                    if (ImGui::Button("TRIGGER SINGLE SOURCE SHOT (C)", { -1, 30 * scale })) {
                        multi_source_mode = false;
                        ctx.src_z_idx = edit_src_z;
                        ctx.src_idx = edit_src_z * ctx.NX + edit_src_x;
                        ctx.src_type = edit_source_type; // 一键激发时应用当前类型
                        ctx.src_amp = edit_amp;         // 一键激发时应用当前幅值
                        g_resetSimRequested = true;
                    }
                    ImGui::Spacing();
                    ImGui::EndTabItem();
                }

                // ==========================================
                // Tab 3: Model_INPUT 外部导入与物理耦合
                // ==========================================
                if (ImGui::BeginTabItem("Model Edit")) {
                    ImGui::Spacing();
                    ImGui::TextColored(uiAccent, "Rock Physics Coupling");

                    ImGui::SliderFloat("P-Wave Velocity (Vp)", &edit_Vp, 500.0f, 6000.0f, "%.1f m/s");
                    ImGui::SliderFloat("S-Wave Velocity (Vs)", &edit_Vs, 200.0f, 3500.0f, "%.1f m/s");
                    ImGui::SliderFloat("Density (Rho)", &edit_Density, 1000.0f, 3000.0f, "%.1f kg/m^3");

                    if (edit_Vp < edit_Vs * 1.5f) edit_Vp = edit_Vs * 1.5f;

                    ImGui::Spacing();
                    if (ImGui::Button("APPLY PHYSICAL MEDIUM", { -1, 30 * scale })) {
                        float mu_val = edit_Density * edit_Vs * edit_Vs;
                        float lambda2mu_val = edit_Density * edit_Vp * edit_Vp;
                        float lambda_val = lambda2mu_val - 2.0f * mu_val;

                        ctx.rho.assign(ctx.total_grid, edit_Density);
                        ctx.mu.assign(ctx.total_grid, mu_val);
                        ctx.lambda.assign(ctx.total_grid, lambda_val);
                        ctx.lambda2mu.assign(ctx.total_grid, lambda2mu_val);
                        g_resetSimRequested = true;
                        UpdatePmlDampingArrays();
                        AutoAlignTimeStep();
                    }
                    ImGui::Spacing();
                    ImGui::Separator();
                    ImGui::TextColored(uiAccent, "GEOPHYSICAL SCENARIO PRESETS");

                    const char* scene_names[] = {
                        "Uniform Medium",
                        "Earth Shell & Core",
                        "Double Slit Interference",
                        "2-Layered Medium",
                        "Straight Waveguide",
                        "Curved Waveguide",
                        "Phononic Crystal Hex",
                        "Random Scattering",
                        "Sinusoidal Interface",
                        "Linear Velocity Gradient",
                        "Penrose Room"
                    };

                    static int selected_scene = current_scene;
                    ImGui::Combo("Select Preset", &selected_scene, scene_names, IM_ARRAYSIZE(scene_names));

                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(uiAccent.x * 0.15f, uiAccent.y * 0.5f, uiAccent.z * 0.4f, 0.75f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(uiAccent.x * 0.2f, uiAccent.y * 0.7f, uiAccent.z * 0.5f, 0.9f));
                    ImGui::Spacing();
                    if (ImGui::Button("APPLY SCENARIO & REALLOCATE", { -1, 32 * scale })) {
                        current_scene = selected_scene;
                        ApplyScenario(current_scene, state);

                        edit_w = ctx.NX;
                        edit_h = ctx.NZ;

                        cudaGraphicsUnregisterResource(gl.cudaSeisRes);
                        glBindTexture(GL_TEXTURE_2D, gl.seisTex);
                        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, ctx.NX, ctx.NZ, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
                        cudaGraphicsGLRegisterImage(&gl.cudaSeisRes, gl.seisTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

                        first_align_needed = true;
                        g_resetSimRequested = true;
                    }
                    ImGui::Spacing();

                    // =========================================================================
                    // 1. 【新增】：鼠标左键画笔与激发工具下拉菜单 (12345 对齐)
                    // =========================================================================
                    ImGui::SeparatorText("Mouse Interaction Tool");

                    // 下拉菜单显示文本，与 ToolMode (0 ~ 4) 的 1、2、3、4、5 键完美对齐
                    const char* brush_names[] = {
                        "1. Default (Trigger Source） ",//默认激发震源
                        "2. High Velocity Brush ",//(高速硬岩画笔)
                        "3. Low Velocity Brush ",//(低速泥层画笔)
                        "4. Custom Material Brush ",//(自定义物性画笔)
                        "5. Eraser "//(橡皮擦 / 恢复背景砂岩)
                    };

                    int current_tool = state.brushType; // 读取当前状态
                    ImGui::PushItemWidth(-1); // 宽度撑满，视觉上更规整

                    if (ImGui::Combo("##LeftClickToolCombo", &current_tool, brush_names, IM_ARRAYSIZE(brush_names))) {
                        state.brushType = current_tool; // 鼠标选择时同步更新状态
                    }
                    ImGui::PopItemWidth();

                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Choose what happens when you left-click on the wavefield.\nOr use hotkeys [1] to [5] on your keyboard.");
                    }

                    // =========================================================================
                    // 2. 【新增】：画笔大小调节滑块 (仅在选择画笔模式 2, 3, 4, 5 时自动展示)
                    // =========================================================================
                    if (state.brushType != TOOL_NONE) {
                        ImGui::Spacing();
                        // 调节 brushRadius 半径大小 (1 ~ 50 像素单元)
                        ImGui::SliderFloat("Brush Radius (px)", &state.brushRadius, 1.0f, 50.0f, "Radius: %.0f px");
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Adjust the physical size of your paintbrush circle.");
                        }
                    }
                    ImGui::Spacing();
                    ImGui::PopStyleColor(2);
                    ImGui::EndTabItem();
                }
                // ==========================================
                // Tab 4: Model_EXPORT 空间裁剪与 SEGY 导出
                // ==========================================
                if (ImGui::BeginTabItem("Model I/O")) {
                    // -------------------------------------------------------------
                    // 外部自定义文本 (TXT) 模型高速加载器
                    // -------------------------------------------------------------
                    ImGui::Separator();
                    ImGui::TextColored(uiAccent, "EXTERNAL MODEL IMPORT (TXT)");

                    static char txt_file_path[512] = "";

                    ImGui::InputText("TXT File", txt_file_path, IM_ARRAYSIZE(txt_file_path));
                    ImGui::SameLine();
                    if (ImGui::Button("Browse##Txt", ImVec2(65 * scale, 0))) {
                        char filePath[512] = { 0 };
                        if (OpenSystemFileDialog(filePath, sizeof(filePath))) {
                            strcpy_s(txt_file_path, filePath);
                        }
                    }

                    static bool flipImportY = false;
                    ImGui::Checkbox("Flip Vertically on Import", &flipImportY);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Check this if your external model appears upside down.\nCommon for Python/Matlab exported data.");
                    }

                    ImGui::Spacing();
                    if (ImGui::Button("LOAD CUSTOM MODEL FILE", { -1, 35 * scale })) {
                        if (strlen(txt_file_path) > 0) {
                            if (LoadModelFromTxt(txt_file_path, flipImportY, gl, state)) {
                                popup_message = "External model loaded successfully!";
                                show_success_popup = true;
                                g_resetSimRequested = true;
                            }
                            else {
                                popup_message = "Failed to load model file.\nPlease check file formatting or dimensions.";
                                show_error_popup = true;
                            }
                        }
                        else {
                            popup_message = "Error: Please select a TXT model file first!";
                            show_error_popup = true;
                        }
                    }
                    ImGui::Spacing();

                    // -------------------------------------------------------------
                    // 外部 SEG-Y 二维物性测线多组导入控制台
                    // -------------------------------------------------------------
                    ImGui::Separator();
                    ImGui::TextColored(uiAccent, "EXTERNAL MODEL IMPORT (SEG-Y GROUP)");

                    static char vp_file_path[512] = "";
                    static char vs_file_path[512] = "";
                    static char rho_file_path[512] = "";

                    ImGui::InputText("Vp (.sgy)", vp_file_path, IM_ARRAYSIZE(vp_file_path));
                    ImGui::SameLine();
                    if (ImGui::Button("Browse##Vp", ImVec2(65 * scale, 0))) {
                        char filePath[512] = { 0 };
                        if (OpenSystemFileDialog(filePath, sizeof(filePath))) {
                            strcpy_s(vp_file_path, filePath);
                        }
                    }

                    ImGui::InputText("Vs (.sgy)", vs_file_path, IM_ARRAYSIZE(vs_file_path));
                    ImGui::SameLine();
                    if (ImGui::Button("Browse##Vs", ImVec2(65 * scale, 0))) {
                        char filePath[512] = { 0 };
                        if (OpenSystemFileDialog(filePath, sizeof(filePath))) {
                            strcpy_s(vs_file_path, filePath);
                        }
                    }

                    ImGui::InputText("Rho(.sgy)", rho_file_path, IM_ARRAYSIZE(rho_file_path));
                    ImGui::SameLine();
                    if (ImGui::Button("Browse##Rho", ImVec2(65 * scale, 0))) {
                        char filePath[512] = { 0 };
                        if (OpenSystemFileDialog(filePath, sizeof(filePath))) {
                            strcpy_s(rho_file_path, filePath);
                        }
                    }

                    ImGui::SliderFloat("Original SEGY Spacing (dx)", &edit_segy_dx, 0.01f, 50.0f, "%.3f m");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("The original spatial spacing of your loaded SGY files.\nExample: 12.5m for standard Marmousi.");
                    }

                    static bool flipSegyY = true;
                    static bool flipSegyX = false;
                    ImGui::Checkbox("Flip Vertically on SEGY Import", &flipSegyY);
                    ImGui::Checkbox("Flip Horizontally on SEGY Import", &flipSegyX);
                    ImGui::Spacing();

                    if (ImGui::Button("LOAD SEGY MODEL GROUP", { -1, 35 * scale })) {
                        if (strlen(vp_file_path) > 0 && strlen(vs_file_path) > 0 && strlen(rho_file_path) > 0) {
                            if (LoadModelFromSegy(vp_file_path, vs_file_path, rho_file_path, flipSegyY, flipSegyX, gl, state)) {
                                popup_message = "SEGY Model Group loaded successfully!";
                                show_success_popup = true;
                                g_resetSimRequested = true;
                            }
                            else {
                                popup_message = "Failed to load SEGY files.\nPlease check file formatting or grid dimensions.";
                                show_error_popup = true;
                            }
                        }
                        else {
                            popup_message = "Error: Please select all three SGY files (Vp, Vs, Rho)!";
                            show_error_popup = true;
                        }
                    }
                   
                    ImGui::Separator();
                    ImGui::TextColored(uiAccent, "EXPORT CURRENT MODEL TO SEG-Y GROUP");

                    static bool export_density_gcm3 = true;
                    static bool export_flip_x = false;

                    static int crop_x_min = 0;
                    static int crop_z_min = 0;
                    static int crop_x_max = 0;
                    static int crop_z_max = 0;

                    if (crop_x_max <= 0 || crop_x_max >= ctx.NX || first_align_needed) {
                        crop_x_min = 0;
                        crop_z_min = 0;
                        crop_x_max = ctx.NX - 1;
                        crop_z_max = ctx.NZ - 1;
                    }

                    // 【自适应对齐】：每次切换模型时，自动将默认导出目标步长设为当前模拟的步长 ctx.h
                    if (first_align_needed) {
                        export_target_dx = ctx.h;
                    }

                    ImGui::Checkbox("Convert Density to g/cm3 on Export", &export_density_gcm3);
                    ImGui::Checkbox("Flip Horizontally on Export", &export_flip_x);

                    ImGui::TextDisabled("Crop Sub-Region Coordinates (Grid Units):");
                    ImGui::SliderInt("X Min (Left)", &crop_x_min, 0, ctx.NX - 1);
                    ImGui::SliderInt("X Max (Right)", &crop_x_max, 0, ctx.NX - 1);
                    ImGui::SliderInt("Z Min (Top)", &crop_z_min, 0, ctx.NZ - 1);
                    ImGui::SliderInt("Z Max (Bottom)", &crop_z_max, 0, ctx.NZ - 1);

                    // =========================================================
                    // 【核心新增】：重采样目标空间步长调节滑块
                    // =========================================================
                    ImGui::SliderFloat("Export Target dx (m)", &export_target_dx, 0.1f, 20.0f, "%.2f m");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Target spatial spacing for exported SGY grid.\nSetting this different from active dx will trigger 2D Resampling.\ne.g., downsample Marmousi from 1.5m to 2.5m.");
                    }

                    float sub_btn_w = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) / 2.0f;

                    if (ImGui::Button("SET TO FULL GRID", ImVec2(sub_btn_w, 0))) {
                        crop_x_min = 0;
                        crop_z_min = 0;
                        crop_x_max = ctx.NX - 1;
                        crop_z_max = ctx.NZ - 1;
                    }
                    ImGui::SameLine();

                    if (ImGui::Button("STRIP PML BOUNDARY", ImVec2(sub_btn_w, 0))) {
                        crop_x_min = ctx.npml;
                        crop_z_min = ctx.npml;
                        crop_x_max = ctx.NX - ctx.npml - 1;
                        crop_z_max = ctx.NZ - ctx.npml - 1;
                    }

                    ImGui::Spacing();
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(uiAccent.x * 0.15f, uiAccent.y * 0.45f, uiAccent.z * 0.55f, 0.75f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(uiAccent.x * 0.2f, uiAccent.y * 0.65f, uiAccent.z * 0.75f, 0.9f));
                    if (ImGui::Button("EXPORT SEG-Y GROUP", { -1, 35 * scale })) {
                        char savePath[512] = { 0 };
                        if (SaveSystemFileDialog(savePath, sizeof(savePath))) {
                            // 传入新参数 export_target_dx 进行高保真重采样导出
                            if (ExportModelToSegy(savePath, export_density_gcm3, export_flip_x, crop_x_min, crop_z_min, crop_x_max, crop_z_max, export_target_dx)) {
                                popup_message = "Cropped & Resampled SEGY Model Group exported successfully!\nCheck out your saved _vp, _vs, _rho files.";
                                show_success_popup = true;
                            }
                            else {
                                popup_message = "Failed to export model.\nPlease check target path permissions.";
                                show_error_popup = true;
                            }
                        }
                    }
                    ImGui::Spacing();
                    ImGui::PopStyleColor(2);
                    ImGui::EndTabItem();
                }

                // ==========================================
                // Tab 4: Data Export 数据采集与观测排列布设
                // ==========================================
                if (ImGui::BeginTabItem("Data Export")) {
                    ImGui::Spacing();

                    // ----------------------------------------------------------
                    // 1. 采集与导出参数
                    // ----------------------------------------------------------
                    ImGui::SeparatorText("Acquisition Settings");
                    ImGui::SliderFloat("Sampling Rate (Hz)", &rec.samplingRateHz, 10.0f, 10000.0f, "%.0f Hz");
                    ImGui::DragFloat("Record Duration (s)", &rec.totalDurationSec, 1.0f, 0.1f, 7200.0f, "%.2f s");
                    ImGui::DragFloat("Export Every (s)", &rec.exportIntervalSec, 1.0f, 0.1f, 3600.0f, "%.1f s");

                    // 计算点数提示
                    int expectedPoints = (int)(rec.samplingRateHz * rec.totalDurationSec);
                    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "Target Samples: %d points/channel", expectedPoints);
                    ImGui::Spacing();

                    // ----------------------------------------------------------
                    // 2. 检波器布局 (增加 X 范围设置)
                    // ----------------------------------------------------------
                    ImGui::SeparatorText("Receiver Geometry");
                    ImGui::SliderInt("Channels", &rec.numChannels, 1, 200, "%d");

                    // 自适应对齐当前物理网格的最大尺寸
                    ImGui::DragIntRange2("X Range", &rec.startX, &rec.endX, 1.0f, 0, ctx.NX - 1, "Start: %d", "End: %d");
                    ImGui::SliderInt("Depth (Y)", &rec.receiverDepth, 0, ctx.NZ - 1);

                    if (ImGui::Button("DEPLOY LINEAR ARRAY", ImVec2(-1, 0))) {
                        rec.receivers.clear();
                        rec.initBuffer();

                        // 根据用户设置的 startX 和 endX 计算道间距
                        float rangeX = (float)(rec.endX - rec.startX);
                        float spacing = (rec.numChannels > 1) ? (rangeX / (rec.numChannels - 1)) : 0;

                        for (int i = 0; i < rec.numChannels; ++i) {
                            rec.receivers.push_back({ (int)(rec.startX + i * spacing), rec.receiverDepth });
                        }
                        // 同步检波器空间拓扑到显卡端
                        SyncReceiversToGPU();
                    }
                    ImGui::Checkbox("Show/Hide LINEAR ARRAY", &rec.isShowarray);
                    ImGui::Spacing();

                    // ----------------------------------------------------------
                    // 3. 录制与一键激发控制
                    // ----------------------------------------------------------
                    ImGui::SeparatorText("Recording Control");

                    if (rec.isRecording) {
                        float progress = rec.totalTimer / rec.totalDurationSec;
                        char  progBuf[64];
                        sprintf(progBuf, "Progress: %.2f / %.2f s", rec.totalTimer, rec.totalDurationSec);
                        ImGui::ProgressBar(progress, ImVec2(-1, 30), progBuf);

                        if (ImGui::Button("STOP & SAVE NOW", ImVec2(-1, 40))) {
                            rec.isRecording = false;
                            state.running = false;

                            // 保存两份：一份标准 SGY (含道头物理坐标)，一份 CSV 文本
                            ExportToSegy(rec, "manual_export.sgy");
                            ExportToCSV(rec, "manual_export.csv");

                            popup_message = "Gather saved to 'manual_export.sgy' & 'manual_export.csv'";
                            show_success_popup = true;
                        }
                    }
                    else {
                        bool ready = !rec.receivers.empty() && (rec.receivers.size() == rec.numChannels);
                        if (!ready) ImGui::BeginDisabled();

                        float availW = ImGui::GetContentRegionAvail().x;
                        float spacingW = ImGui::GetStyle().ItemSpacing.x;

                        // 按钮 A: 仅开启采集 (静默录制当前时有时无的干扰)
                        if (ImGui::Button("START CAPTURE", ImVec2(availW * 0.4f, 45))) {
                            rec.totalTimer = 0.0f;
                            rec.exportTimer = 0.0f;
                            rec.timer = 0.0f;
                            rec.currentFileIndex = 1;
                            rec.initBuffer();

                            SyncReceiversToGPU();
                            rec.isRecording = true;
                            state.running = true;
                        }
                        ImGui::SameLine();

                        // 按钮 B: 一键自动激发并开始全自动采集 (高亮红色)
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.1f, 0.1f, 1.0f));
                        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.9f, 0.2f, 0.2f, 1.0f));
                        if (ImGui::Button("TRIGGER & ACQUIRE", ImVec2(availW * 0.6f - spacingW, 45))) {
                            rec.totalTimer = 0.0f;
                            rec.exportTimer = 0.0f;
                            rec.timer = 0.0f;
                            rec.currentFileIndex = 1;
                            rec.initBuffer();

                            // 清空波场重定位到指定位置放源，并一键完成检波器显存同步
                            TriggerSingleShot(gl, state);

                            rec.isRecording = true;
                            state.running = true;
                        }
                        ImGui::PopStyleColor(2);

                        if (!ready) {
                            ImGui::EndDisabled();
                            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Hint: Deploy receivers first.");
                        }
                    }
                    ImGui::EndTabItem();
                }

                // ==========================================
                // Tab 5: visual 色谱渲染与标尺网格调节
                // ==========================================
                if (ImGui::BeginTabItem("Visual")) {
                    ImGui::Spacing();
                    ImGui::SliderFloat("Color Gain", &color_scale, 0.01f, 100.0f, "%.3f", ImGuiSliderFlags_Logarithmic);

                    // 完全对齐图片中的专业物理场分量类别
                    const char* components[] = {
                        "Velocity Mag",
                        "Velocity X",
                        "Velocity Y",
                        "Stress Normal",
                        "Stress Shear",
                        "P-Wave (Div)",
                        "S-Wave (Curl)"
                    };
                    ImGui::Combo("Show Component", &show_component, components, IM_ARRAYSIZE(components));

                    ImGui::Spacing();

                    const char* style_types[] = {
                        "0_Seismic Bipolar",      // 标准地震红白蓝
                        "1_Coolwarm",             // Matplotlib 经典冷暖
                        "2_PuOr",                 // 紫橙双极振幅
                        "3_Grayscale",            // 灰度振幅图
                        "4_Inferno",              // 深邃红黄能量场
                        "5_Plasma",               // 感知均匀霓虹
                        "6_Viridis",              // Matplotlib 标准
                        "7_Turbo",                // Google Turbo 高动态
                        "8_Magma Glow",           // 熔岩发光风格
                        "9_Jet Rainbow",          // 经典彩虹
                        "10_Hot",                 // 热火色谱
                        "11_Cold",                // 冰冷色谱
                        "12_Terrain Map"          // 地貌地质图
                    };
                    ImGui::Combo("Visual Style", &waveStyle, style_types, IM_ARRAYSIZE(style_types));
                    ImGui::Spacing();

                    const char* model_styles[] = {
                        "Dark Titanium",
                        "Geological Map",
                        "Grayscale Velocity",
                        "WaveStyle Synced",
                        "Scientific White"
                    };
                    ImGui::Combo("Geological Background", &modelStyle, model_styles, IM_ARRAYSIZE(model_styles));
                    ImGui::Spacing();

                    ImGui::Checkbox("Show Background Grid & Dots", &showGrid);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Toggle the visibility of the grid lines and glowing intersection dots.");
                    }

                    ImGui::SliderFloat("Grid Spacing (m)", &ui_gridSpacing, 10.0f, 200.0f);
                    ImGui::SliderFloat("Grid Opacity", &ui_gridOpacity, 0.0f, 1.0f);

                    ImGui::EndTabItem();
                }

                ImGui::EndTabBar();
            }

            // 底部主题变色器色板
            ImGui::SetCursorPosY(ImGui::GetWindowHeight() - 55 * scale);
            float avail_w = ImGui::GetContentRegionAvail().x;
            float reset_btn_w = 60.0f * scale;
            float picker_w = avail_w - reset_btn_w - ImGui::GetStyle().ItemSpacing.x;

            ImGui::PushItemWidth(picker_w);
            ImGui::ColorEdit3("##AccentColorPicker", (float*)&uiAccent, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueBar);
            ImGui::PopItemWidth();
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Interface Hue Accent Color");
            }

            ImGui::SameLine();
            if (ImGui::Button("Reset##Theme", ImVec2(reset_btn_w, 0))) {
                uiAccent = ImVec4(0.5f, 1.0f, 0.5f, 1.0f); // 恢复荧光绿
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Reset UI Accent Color to Default Cyan");
            }
        }
        ImGui::End();

        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(8);
    }

    // -------------------------------------------------------------------------
    // F. 独立监测视窗 (Simulation & PML Monitor)
    // -------------------------------------------------------------------------
    if (show_Monitor_par) {
        ImGui::SetNextWindowPos(ImVec2(30 * scale, 80 * scale), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(420 * scale, 520 * scale), ImGuiCond_FirstUseEver);

        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.05f, 0.08f, 0.1f, 0.85f));   // 深色钛金背景
        ImGui::PushStyleColor(ImGuiCol_Border, uiAccent);                             // 荧光边框
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.5f);

        ImGui::PushStyleColor(ImGuiCol_HeaderActive, uiAccent);
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(uiAccent.x * 0.8f, uiAccent.y * 0.8f, uiAccent.z * 0.8f, 0.8f));

        if (ImGui::Begin("Simulation & PML Monitor", &show_Monitor_par)) {
            static float displayFPS = 60.0f;
            displayFPS = displayFPS * 0.95f + io.Framerate * 0.05f;
            ImGui::TextColored(uiAccent, "MONITOR STATE  |  FPS: %.1f", displayFPS);
            ImGui::Separator();

            ImGui::Spacing();
            if (ImGui::CollapsingHeader("ACTIVE PARAMETERS MONITOR", ImGuiTreeNodeFlags_DefaultOpen)) {
                // 呼吸警告色 (闪烁频率 6Hz，平滑正弦插值)
                float time = (float)ImGui::GetTime();
                float alpha = 0.35f + 0.65f * (sinf(time * 6.0f) * 0.5f + 0.5f);
                ImVec4 warningCol = ImVec4(1.0f, 0.82f, 0.0f, alpha); // 琥珀荧光黄

                ImGui::TextDisabled("Grid Dimensions:"); ImGui::SameLine();
                ImGui::Text("%d x %d (Total: %d)", ctx.NX, ctx.NZ, ctx.total_grid);

                ImGui::TextDisabled("Physical Scale:"); ImGui::SameLine();
                ImGui::Text("%.1f m x %.1f m", ctx.NX * ctx.h, ctx.NZ * ctx.h);

                // =========================================================================
                // 【核心修复】：基于 cudaMemGetInfo 的实时物理硬件 VRAM 占用诊断器
                // =========================================================================
                size_t free_mem = 0;
                size_t total_mem = 0;
                float used_vram_mb = 0.0f;
                float total_vram_mb = 0.0f;

                // 直接向 CUDA 运行时驱动发起快速硬件查询
                if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
                    used_vram_mb = static_cast<float>(total_mem - free_mem) / (1024.0f * 1024.0f);
                    total_vram_mb = static_cast<float>(total_mem) / (1024.0f * 1024.0f);
                }

                // 1. 展现系统底层真实的显存整体负载
                ImGui::TextDisabled("System GPU VRAM:"); ImGui::SameLine();
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.8f, 1.0f), "%.1f / %.1f MB", used_vram_mb, total_vram_mb);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("System-wide allocated VRAM on the active GPU (Current Used / Total Capacity).");
                }

                // 2. 展现当前数值网格（系数场+波动应力场）估算的占用量 (让学者清晰掌握算法开销)
                ImGui::TextDisabled("Grid VRAM (Est.):"); ImGui::SameLine();
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.8f, 1.0f), "%.2f MB", cached_vram_mb);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Estimated memory allocated strictly for this finite-difference simulation grid.");
                }
                // 1. 尝试惰性载入 NVML
                g_nvml.Load();

                unsigned int gpu_util = 0;
                unsigned int gpu_temp = 0;
                unsigned int gpu_power_mw = 0;
                bool         nvml_ok = false;

                // 2. 如果成功连接，直接向显卡驱动提取实时硬件状态
                if (g_nvml.loaded) {
                    nvmlUtilization_t util{};
                    if (g_nvml.getUtil(g_nvml.devHandle, &util) == NVML_SUCCESS) {
                        gpu_util = util.gpu;
                    }
                    if (g_nvml.getTemp(g_nvml.devHandle, NVML_TEMPERATURE_GPU, &gpu_temp) == NVML_SUCCESS) {
                        // 成功捕获核心温度
                    }
                    if (g_nvml.getPower(g_nvml.devHandle, &gpu_power_mw) == NVML_SUCCESS) {
                        // 成功捕获实时功耗 (毫瓦)
                    }
                    nvml_ok = true;
                }

                // -------------------------------------------------------------
                // 3. 展现 GPU Core 实时运行占用率
                // -------------------------------------------------------------
                ImGui::TextDisabled("GPU Core Usage:"); ImGui::SameLine();
                if (nvml_ok) {
                    ImGui::TextColored(uiAccent, "%u %%", gpu_util);
                }
                else {
                    ImGui::TextDisabled("N/A (NVML Offline)");
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Active SM core utilization rate (same as Task Manager's CUDA chart).");
                }

                // -------------------------------------------------------------
                // 4. 展现 GPU 核心实时工作温度
                // -------------------------------------------------------------
                ImGui::TextDisabled("GPU Temperature:"); ImGui::SameLine();
                if (nvml_ok) {
                    // 温度警告：当核心温度超过 82°C 时，自动变色警示
                    ImVec4 tempColor = (gpu_temp >= 82) ? ImVec4(1.0f, 0.2f, 0.2f, 1.0f) : uiAccent;
                    ImGui::TextColored(tempColor, "%u oC", gpu_temp);
                }
                else {
                    ImGui::TextDisabled("N/A (NVML Offline)");
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Real-time GPU core temperature.");
                }

                // -------------------------------------------------------------
                // 5. 展现 GPU 实时核级工作功耗
                // -------------------------------------------------------------
                ImGui::TextDisabled("GPU Power Draw:"); ImGui::SameLine();
                if (nvml_ok) {
                    ImGui::TextColored(uiAccent, "%.1f W", static_cast<float>(gpu_power_mw) / 1000.0f);
                }
                else {
                    ImGui::TextDisabled("N/A (NVML Offline)");
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Real-time GPU board power consumption (Watts).");
                }


                ImGui::TextDisabled("Active Spacing (dx):"); ImGui::SameLine();
                ImGui::Text("%.1f m", ctx.h);
                if (std::abs(temp_dx - ctx.h) > 1e-4f) {
                    ImGui::SameLine();
                    ImGui::TextColored(warningCol, "[Apply Pending]");
                }

                ImGui::TextDisabled("Active Time Step (dt):"); ImGui::SameLine();
                ImGui::Text("%.6f s", ctx.dt);
                if (std::abs(temp_dt - ctx.dt) > 1e-7f) {
                    ImGui::SameLine();
                    ImGui::TextColored(warningCol, "[Apply Pending]");
                }

                ImGui::TextDisabled("Active Max Steps (nt):"); ImGui::SameLine();
                ImGui::Text("%d steps (%.2f s simulated)", ctx.nt, ctx.nt * ctx.dt);
                if (temp_nt != ctx.nt) {
                    ImGui::SameLine();
                    ImGui::TextColored(warningCol, "[Apply Pending]");
                }

                ImGui::TextDisabled("Active PML Width:"); ImGui::SameLine();
                ImGui::Text("%d px", ctx.npml);
                if (temp_pml != ctx.npml) {
                    ImGui::SameLine();
                    ImGui::TextColored(warningCol, "[Apply Pending]");
                }

                int active_src_x = ctx.src_idx % ctx.NX;
                int active_src_z = ctx.src_idx / ctx.NX;
                ImGui::TextDisabled("Active Source Pos:"); ImGui::SameLine();
                ImGui::Text("(%d, %d)", active_src_x, active_src_z);
                if (edit_src_x != active_src_x || edit_src_z != active_src_z) {
                    ImGui::SameLine();
                    ImGui::TextColored(warningCol, "[Apply Pending]");
                }

                ImGui::Separator();
                ImGui::TextColored(uiAccent, "ACTIVE GEOPHYSICAL METRICS");

                ImGui::TextDisabled("P-Wave Velocity (Vp):"); ImGui::SameLine();
                ImGui::Text("%.1f m/s ~ %.1f m/s", cached_min_vp, cached_max_vp);

                ImGui::TextDisabled("S-Wave Velocity (Vs):"); ImGui::SameLine();
                ImGui::Text("%.1f m/s ~ %.1f m/s", cached_min_vs, cached_max_vs);

                ImGui::TextDisabled("Density Range (Rho):"); ImGui::SameLine();
                ImGui::Text("%.1f ~ %.1f kg/m^3", cached_min_rho, cached_max_rho);
            }

            ImGui::Spacing();
            ImGui::TextDisabled("PML Damping Profile (dx) Real-Time Monitor:");

            // 生成临时 X 轴网格坐标
            std::vector<float> pml_x_axis(ctx.NX);
            for (int j = 0; j < ctx.NX; ++j) {
                pml_x_axis[j] = (float)j;
            }

            ImPlot::PushStyleColor(ImPlotCol_FrameBg, IM_COL32(10, 12, 15, 255));
            ImPlot::PushStyleColor(ImPlotCol_PlotBg, IM_COL32(0, 0, 0, 255));
            ImPlot::PushStyleColor(ImPlotCol_Line, IM_COL32(0, 255, 120, 255));
            ImPlot::PushStyleColor(ImPlotCol_AxisGrid, IM_COL32(35, 35, 35, 120));

            if (ImPlot::BeginPlot("##PmlProfilePlot", ImVec2(-1, 140 * scale), ImPlotFlags_NoLegend)) {
                ImPlot::SetupAxes("Grid Node Index", "Damping (Hz)", ImPlotAxisFlags_NoTickLabels, 0);

                float max_d = 1.0f;
                for (float val : ctx.dx) {
                    if (val > max_d) max_d = val;
                }
                ImPlot::SetupAxesLimits(0.0f, ctx.NX, -max_d * 0.05f, max_d * 1.1f);

                ImPlot::PlotLine("Damping dx", pml_x_axis.data(), ctx.dx.data(), ctx.NX);

                double pml_left_boundary = ctx.npml;
                double pml_right_boundary = ctx.NX - ctx.npml;

                ImVec2 left_p1 = ImPlot::PlotToPixels(ImPlotPoint(pml_left_boundary, -max_d * 0.05f));
                ImVec2 left_p2 = ImPlot::PlotToPixels(ImPlotPoint(pml_left_boundary, max_d * 1.1f));
                ImVec2 right_p1 = ImPlot::PlotToPixels(ImPlotPoint(pml_right_boundary, -max_d * 0.05f));
                ImVec2 right_p2 = ImPlot::PlotToPixels(ImPlotPoint(pml_right_boundary, max_d * 1.1f));

                ImPlot::GetPlotDrawList()->AddLine(left_p1, left_p2, IM_COL32(255, 140, 0, 180), 1.5f);
                ImPlot::GetPlotDrawList()->AddLine(right_p1, right_p2, IM_COL32(255, 140, 0, 180), 1.5f);

                ImPlot::EndPlot();
            }
            ImPlot::PopStyleColor(4);
            ImGui::Spacing();
        }
        ImGui::End();

        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(4);
    }

    RenderAnalysisWindow(g_analyzerState);

    // -------------------------------------------------------------------------
    // G. 渲染底部专业状态监控栏 (BottomBar)
    // -------------------------------------------------------------------------
    {
        ImGui::SetNextWindowPos({ 0, (float)winH - barHeight });
        ImGui::SetNextWindowSize({ (float)winW, barHeight });

        ImVec4 botBarBg = ImVec4(0.95f, 1.0f, 0.98f, 0.2f);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, botBarBg);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10 * scale, 0));

        ImGui::Begin("BottomBar_Seis", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar);
        {
            float centerY = 14.0f * scale;

            ImGui::SetCursorPos({ 20 * scale, centerY });
            if (hover_valid) {
                ImGui::TextDisabled("GRID:"); ImGui::SameLine();
                ImGui::TextColored(uiAccent, "X:%d, Z:%d", hover_mx, hover_my);
                ImGui::SameLine();
                ImGui::TextDisabled("| COORD:"); ImGui::SameLine();
                ImGui::TextColored(uiAccent, "X:%.1f m, Z:%.1f m", hover_mx * ctx.h, hover_my * ctx.h);
            }
            else {
                ImGui::TextDisabled("GRID MONITOR: STANDBY (HOVER OVER WAVEFIELD)");
            }

            char paramStr[256];
            if (hover_valid) {
                sprintf(paramStr, "P-WAVE SPEED (Vp): %.1f m/s  |  S-WAVE SPEED (Vs): %.1f m/s  |  DENSITY (Rho): %.1f kg/m^3", hover_Vp, hover_Vs, hover_Rho);
            }
            else {
                sprintf(paramStr, "ROCK MEDIUM MONITOR: ACTIVE");
            }

            float textWidth = ImGui::CalcTextSize(paramStr).x;
            ImGui::SetCursorPos({ (winW - textWidth) * 0.5f, centerY });

            if (hover_valid) {
                ImGui::TextColored(uiAccent, "%s", paramStr);
            }
            else {
                ImGui::TextDisabled("%s", paramStr);
            }

            float rightSectionWidth = 220.0f * scale;
            ImGui::SetCursorPos({ (float)winW - rightSectionWidth, centerY });
            ImGui::TextDisabled("STATUS:"); ImGui::SameLine();
            if (hover_valid) {
                ImGui::TextColored({ 0.2f, 1.0f, 0.4f, 1.0f }, "INDEX LINKED");
            }
            else {
                ImGui::TextColored({ 1.0f, 0.5f, 0.0f, 1.0f }, "CURSOR OUT");
            }
        }
        ImGui::End();
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(1);

        ImDrawList* botDraw = ImGui::GetForegroundDrawList();
        botDraw->AddLine(
            { 0, (float)winH - barHeight },
            { (float)winW, (float)winH - barHeight },
            IM_COL32(uiAccent.x * 255, uiAccent.y * 255, uiAccent.z * 255, 60),
            1.0f
        );
    }
}

// =============================================================================
// 30. GPU 核心渲染入口主驱动 (弹性波数值模拟屏幕核心渲染流水线)
// =============================================================================
inline void RenderSeisSimScreen_GPU(SimState& state, int winW, int winH, GLHandles& gl, const GpuInfo& info) {
    // 1. 基于窗口物理宽度计算 OSD 比例因子与顶部/底部菜单栏高度
    const float scale = (float)winW / 1920.0f;
    const float barHeight = 48.0f * scale;

    // 2. 惰性初始化：仅在首次进入或重算时执行显存与系数分配
    InitializeSeismicSimulation(gl);

    // 3. 视口自适应对齐与安全吸附
    ApplyCameraAutoAlignment(winW, winH, barHeight);

    // 4. 用户输入捕获：处理平移缩放、一键重置信号、鼠标激发注入
    HandleSeismicInteractions(state, winW, winH, barHeight);

    // 5. 有限差分时演推进：在 GPU 上更新下一帧应力、速度波场
    UpdateWavefieldSimulation(state);

    // 6. 底图渲染：执行显存 2D 拷贝与 OpenGL 全屏着色渲染
    RenderFullBackbufferWavefield(winW, winH, barHeight, gl);

    // 7. GUI/HUD 覆盖：在最上层绘制刻度物理标尺、科技画笔和 ImGui 交互面板
    RenderSeisHUD(state, winW, winH, barHeight, scale, gl, info);
    // =========================================================================
    // 【核心新增】：挂载全局模态弹窗，使其完美浮在所有层级最顶端，恢复弹窗响应
    // =========================================================================
    RenderSeisPopups();
}