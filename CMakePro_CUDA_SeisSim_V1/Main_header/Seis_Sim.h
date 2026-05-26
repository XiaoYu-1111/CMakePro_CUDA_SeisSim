#pragma once
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

// =================================================================================
// 1. 平台相关定义与系统头文件 (Platform Specifics)
// =================================================================================
#ifdef _WIN32
#define NOMINMAX          // 必须放在 windows.h 之前，禁用 min/max 宏，避免与 std::min/max 冲突
#include <windows.h>      // Windows API
#include <direct.h>       // _mkdir
#include <io.h>           // _access
#else
#include <sys/stat.h>     // mkdir
#include <sys/types.h>
#include <unistd.h>
#endif

#include "Common.h"
#include "Cuda_Check.cuh" // CUDA计算头文件
#include "SeismicIO.h"


// 声明外部未绑定的 FDM 主机端物理配置函数
extern "C" {
    void setupTestContext(SimulationContext& ctx, const Parameters& par);
    void generateRickerWavelet(std::vector<float>& wavelet, int nt, float dt, float f0, float t0);
}

// =============================================================================
// 1. C++20 文件级全局持久化状态控制变量 (Header-Safe inline)
// =============================================================================
static SimulationContext ctx;
static GPUSimData gpu_data;
static bool is_seis_initialized = false;

// 全局控制信号
inline bool g_resetSimRequested = false;      // C 键重置模拟信号
inline bool g_resetViewportRequested = false;  // R 键重置视口信号

// --- 物理与网格大小控制 ---
inline int edit_w = 1000;
inline int edit_h = 500;
inline float edit_f0 = 100.0f;
inline float edit_t0 = 1/edit_f0;

inline float temp_dt = 0.0002f;
inline float temp_dx = 1.0f;
inline int temp_nt = 10000;
inline int temp_pml = 50;

// --- 时间演化与渲染属性 ---
inline int current_it = 0;
inline float accumulated_compute_time = 0.0f;
inline float color_scale = 10.0f;
inline int show_component = 0; // 0: Vz, 1: Vx
inline int steps_per_frame = 20;
inline bool showHUD = true;
inline bool show_Monitor_par = false;
inline int waveStyle = 0; // 默认采用 Style 0: Magma Glow

// --- 物理震源位置与受力角度 ---
inline int edit_src_x = 500;
inline int edit_src_z = 125;
inline float edit_angle = 0.0f;

// --- 视口平移与滚轮缩放 ---
inline float viewZoom = 1.0f;
inline ImVec2 viewOffset = { 0.0f, 0.0f };
inline bool first_align_needed = false;
inline Parameters par;

// --- 动态多震源与无限模式 ---
inline bool multi_source_mode = false;
inline std::vector<GPUSource> active_sources;
inline bool infinite_mode = false;

// --- 鼠标悬停实时监控数据 ---
inline int hover_mx = 0;
inline int hover_my = 0;
inline float hover_Vp = 0.0f;
inline float hover_Vs = 0.0f;
inline float hover_Rho = 0.0f;
inline bool hover_valid = false;

// --- 全局 UI 荧光主题色 (默认绿色荧光) ---
inline ImVec4 uiAccent = ImVec4(0.5f, 1.0f, 0.5f, 1.0f);

inline bool is_mouse_inside = false;

inline int modelStyle = 0; // 0: 钛金灰, 1: 科学地质图, 2: 灰度Vp, 3: Viridis, 4: Cyber
inline int current_scene = SCENE_UNIFORM; // 当前加载的物理场景

// =============================================================================
// 【核心新增】：将 Vp、Vs、Density 提升至全局 inline，彻底解决参数在重算时回滚的 Bug [3]
// =============================================================================
inline float edit_Vp = 2000.0f;      // 全局共享：纵波速度
inline float edit_Vs = 1400.0f;      // 全局共享：横波速度
inline float edit_Density = 2000.0f; // 全局共享：密度

inline GLuint g_modelTex = 0; // 离屏地质物性纹理句柄

inline bool showGrid = true; // 默认开启背景网格与矩阵点显示

inline static bool show_success_popup = false;
inline static bool show_error_popup = false;
inline static std::string popup_message="";

// =============================================================================
// 【高级缓存】：CPU 侧物理极值与显存估算缓存（避开每帧循环 400 万点导致的 FPS 暴降瓶颈） [3]
// =============================================================================
inline float cached_max_vp = 0.0f;
inline float cached_min_vp = 0.0f;
inline float cached_max_vs = 0.0f;
inline float cached_min_vs = 0.0f;
inline float cached_max_rho = 0.0f;
inline float cached_min_rho = 0.0f;
inline float cached_vram_mb = 0.0f; // 估算显存占用

// =============================================================================
// 【核心新增】：外部导入时原 SEG-Y 测线的实际道间距（物理间距，单位：米） [3]
//  用于自适应重采样（例如 301 道、0.1m 间距的 30m 模型会自动重构为 31 个 1m 的仿真网格点）
// =============================================================================
inline float edit_segy_dx = 1.0f;

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
    // 包含应力、速度、PML 分裂场、物性等共约 20 个双精度/单精度二维浮点显存块 [3]
    size_t estimated_bytes = ctx.total_grid * sizeof(float) * 20;
    cached_vram_mb = static_cast<float>(estimated_bytes) / (1024.0f * 1024.0f);
}

// =============================================================================
// 2. 标尺绘制模块 (Symmetric Titanium Background 风格自适应物理标尺)
// =============================================================================
inline void RenderGridRulerOnBackbuffer(
    const SimulationContext& ctx,
    const SimState& state,
    int winW, int winH,
    float barHeight,
    const ImVec2& viewOffset,
    float viewZoom
) {
    ImDrawList* draw_list = ImGui::GetBackgroundDrawList();
    ImU32 textCol = IM_COL32(220, 220, 220, 225); // 荧光白
    ImU32 tickCol = IM_COL32(0, 255, 200, 100);   // 青色刻度线

    float winAspect = (float)winW / (float)winH;
    float simAspect = (float)ctx.NX / (float)ctx.NZ;
    ImVec2 aspectCorr = { 1.0f, 1.0f };
    if (winAspect > simAspect) {
        aspectCorr.x = winAspect / simAspect;
    }
    else {
        aspectCorr.y = simAspect / winAspect;
    }

    // 根据当前的缩放比动态调节刻度的间距，防止文字重叠 [1.2.7]
    float spacing = 50.0f;
    if (viewZoom < 0.35f)  spacing = 100.0f;
    if (viewZoom < 0.15f)  spacing = 200.0f;
    if (viewZoom < 0.06f)  spacing = 500.0f;
    if (viewZoom > 2.2f)   spacing = 20.0f;
    if (viewZoom > 5.0f)   spacing = 10.0f;
    if (viewZoom > 12.0f)  spacing = 5.0f;

    // A. 绘制顶部 X 轴物理刻度 (100% 像素级对齐)
    for (float simX = 0.0f; simX <= ctx.NX; simX += spacing) {
        float aPos_x = (simX / (float)ctx.NX) * 2.0f - 1.0f;
        float Pndc_x = (aPos_x - viewOffset.x) * viewZoom / aspectCorr.x;
        float screenX = (Pndc_x * 0.5f + 0.5f) * winW;

        if (screenX >= 0.0f && screenX <= winW) {
            float screenY = barHeight + 2.0f;
            draw_list->AddLine(ImVec2(screenX, screenY), ImVec2(screenX, screenY + 6.0f), tickCol, 1.5f);

            float physVal = simX * ctx.h;
            char buf[32];
            snprintf(buf, 32, "%.0fm", physVal);

            ImVec2 txtSize = ImGui::CalcTextSize(buf);
            float offsetX = -txtSize.x * 0.5f;

            ImVec2 pos(screenX + offsetX, screenY + 8.0f);
            draw_list->AddText(ImVec2(pos.x + 1, pos.y + 1), IM_COL32(0, 0, 0, 200), buf);
            draw_list->AddText(pos, textCol, buf);
        }
    }

    // B. 绘制左侧 Y 轴物理深度刻度 (100% 像素级对齐)
    for (float simY = 0.0f; simY <= ctx.NZ; simY += spacing) {
        float aPos_y = 1.0f - (simY / (float)ctx.NZ) * 2.0f;
        float Pndc_y = (aPos_y - viewOffset.y) * viewZoom / aspectCorr.y;
        float screenY = (0.5f - Pndc_y * 0.5f) * winH;

        if (screenY >= barHeight && screenY <= winH) {
            float screenX = 4.0f;
            draw_list->AddLine(ImVec2(screenX, screenY), ImVec2(screenX + 6.0f, screenY), tickCol, 1.5f);

            float physVal = simY * ctx.h;
            char buf[32];
            snprintf(buf, 32, "%.0fm", physVal);

            ImVec2 txtSize = ImGui::CalcTextSize(buf);
            ImVec2 pos(screenX + 8.0f, screenY - txtSize.y * 0.5f);

            draw_list->AddText(ImVec2(pos.x + 1, pos.y + 1), IM_COL32(0, 0, 0, 200), buf);
            draw_list->AddText(pos, textCol, buf);
        }
    }
}

// =============================================================================
// 3. 移植版：星空/科技主题鼠标画笔光标渲染
// =============================================================================
inline void RenderBrushCursor(const SimState& state, const ViewportInfo& vp) {
    ImVec2 mousePos = ImGui::GetMousePos();
    bool isInside = (mousePos.x >= vp.x && mousePos.x < (vp.x + vp.w) &&
        mousePos.y >= vp.y && mousePos.y < (vp.y + vp.h));

    if (ImGui::GetIO().WantCaptureMouse) return;

    if (isInside) {
        ImDrawList* fg = ImGui::GetForegroundDrawList();

        if (state.brushType == TOOL_NONE) {
            float t = (float)ImGui::GetTime();
            const float crosshairSize = 18.0f;
            const float outerRingRadius = 35.0f;

            ImU32 col_main = IM_COL32(102, 191, 217, 255);      // 科技青
            ImU32 col_dynamic = IM_COL32(255, 230, 51, 255);    // 荧光黄
            ImU32 col_shadow = IM_COL32(13, 20, 56, 150);

            fg->AddLine(ImVec2(mousePos.x - crosshairSize, mousePos.y), ImVec2(mousePos.x + crosshairSize, mousePos.y), col_shadow, 4.0f);
            fg->AddLine(ImVec2(mousePos.x, mousePos.y - crosshairSize), ImVec2(mousePos.x, mousePos.y + crosshairSize), col_shadow, 4.0f);
            fg->AddLine(ImVec2(mousePos.x - crosshairSize, mousePos.y), ImVec2(mousePos.x + crosshairSize, mousePos.y), col_main, 2.0f);
            fg->AddLine(ImVec2(mousePos.x, mousePos.y - crosshairSize), ImVec2(mousePos.x, mousePos.y + crosshairSize), col_main, 2.0f);

            fg->AddCircle(mousePos, outerRingRadius, col_main, 32, 1.5f);

            float angle = t * 4.0f;
            ImVec2 scanEnd = ImVec2(mousePos.x + cos(angle) * outerRingRadius,
                mousePos.y + sin(angle) * outerRingRadius);
            fg->AddLine(mousePos, scanEnd, col_dynamic, 2.0f);
            fg->AddCircleFilled(scanEnd, 3.5f, col_dynamic);

            float pulse_t = fmodf(t, 1.5f) / 1.5f;
            float pulse_radius = pulse_t * outerRingRadius;
            int pulse_alpha = (int)(sin(pulse_t * 3.14159f) * 120);
            ImU32 col_pulse = IM_COL32(255, 230, 51, pulse_alpha);
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
// 4. 新增：高精度随背景平移缩放、无漂移的物理震源准星 (Orange Crosshair)
// =============================================================================
inline void RenderSourceMarker(
    const SimulationContext& ctx,
    int src_x, int src_z,
    int winW, int winH,
    float barHeight,
    const ImVec2& viewOffset,
    float viewZoom
) {
    float winAspect = (float)winW / (float)winH;
    float simAspect = (float)ctx.NX / (float)ctx.NZ;
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
        ImVec2 pos(screenX, screenY);

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
// 简单的雷克子波产生器
void generateRickerWavelet(std::vector<float>& wavelet, int nt, float dt, float f0, float t0) {
    wavelet.resize(nt);
    for (int i = 0; i < nt; ++i) {
        float t = i * dt - t0;
        float pi2_f0 = M_PI * M_PI * f0 * f0;
        wavelet[i] = (1.0f - 2.0f * pi2_f0 * t * t) * expf(-pi2_f0 * t * t);
    }
}

// =============================================================================
// 【物理核心】：根据当前网格物性最大波速 Vp_max、PML 宽度 npml 和网格间距 h，
//  重新计算并更新 CPU 端高阶 Staggered-grid 2.5 阶黄金 PML 阻尼衰减系数数组！
//  每次重置模拟、切换场景、改变网格大小或滑动条应用时，该函数都会自动触发，保证绝对对齐。
// =============================================================================
inline void UpdatePmlDampingArrays() {
    float max_vp = 0.0f;
    for (int k = 0; k < ctx.total_grid; ++k) {
        float vp = sqrtf(ctx.lambda2mu[k] / ctx.rho[k]);
        if (vp > max_vp) max_vp = vp;
    }

    ctx.dx.assign(ctx.NX, 0.0f);
    ctx.dx_half.assign(ctx.NX, 0.0f);
    ctx.dz.assign(ctx.NZ, 0.0f);
    ctx.dz_half.assign(ctx.NZ, 0.0f);

    if (ctx.npml > 0) {
        float L = ctx.npml * ctx.h; // PML 吸收层物理总厚度

        // 采用 3.5 倍的学术级阻尼安全余量，配合 2.5 阶黄金阻尼，彻底根除流固发散
        float d_max = (3.0f * max_vp * 16.12f * 3.5f) / (2.0f * L);
        float pml_power = 2.5f;

        // X 轴 PML 阻尼系数初始化 (交错网格 0.5 半步长偏移对齐) [6]
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

        // Z 轴 PML 阻尼系数初始化
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

// 模拟测试上下文初始化
void setupTestContext(SimulationContext& ctx, const Parameters& par) {
    ctx.NX = par.model.xnum;
    ctx.NZ = par.model.znum;
    ctx.total_grid = ctx.NX * ctx.NZ;
    ctx.dt = par.FDM.dt;
    ctx.nt = static_cast<int>(par.FDM.nt);
    ctx.npml = static_cast<int>(par.FDM.npml);

    // =============================================================================
    // 【核心修复 1】：取消硬编码 ctx.flag_type = 3
    // 仅在首次启动未初始化（值为 0）时设为默认值 3；其余情况继承并保留用户在 UI 上的选择
    // =============================================================================
    if (ctx.flag_type == 0) {
        ctx.flag_type = 3;
    }

    ctx.upFlag = (par.FDM.upFlag > 0.5f);

    // 这里以 par.model.dx 为准（或者 par.FDM.xPace，两者在物理上是一致的）
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

    // =============================================================================
    // 【核心修复 2】：取消本地 static 变量
    // 直接读取我们刚刚定义在头文件最顶部的、与 UI 强绑定的全局共享物性变量 [3]
    // =============================================================================
    float mu_val = edit_Density * edit_Vs * edit_Vs;
    float lambda2mu_val = edit_Density * edit_Vp * edit_Vp;
    float lambda_val = lambda2mu_val - 2.0f * mu_val;

    // 重新填充本地 Context (防止重算时参数被覆盖回滚)
    ctx.rho.assign(ctx.total_grid, edit_Density);
    ctx.mu.assign(ctx.total_grid, mu_val);
    ctx.lambda.assign(ctx.total_grid, lambda_val);
    ctx.lambda2mu.assign(ctx.total_grid, lambda2mu_val);

    // =============================================================================
    // 【核心精简】：直接调用重构后的 PML 阻尼曲线重算函数，实现单一逻辑出口！
    // =============================================================================
    UpdatePmlDampingArrays();
    // 自由表面 dp_flat 计算 (保持不变)
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
    //震源
    ctx.src_z_idx = ctx.NZ / 4;                                      // Z轴深度设定在 1/4 处
    ctx.src_idx = ctx.src_z_idx * ctx.NX + (ctx.NX / 2);             // X轴定位在中心 1/2 处
    ctx.src_angle = par.FDM.angle;
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
// 2. 核心物性注入器：将 Vp、Vs、密度转换为 CPU 端弹性拉梅常数
// =============================================================================
inline void SetMaterialAt(int x, int y, float vp, float vs, float rho) {
    /*int k = y * ctx.NX + x;
    if (k >= 0 && k < ctx.total_grid) {
        ctx.rho[k] = rho;
        ctx.mu[k] = rho * vs * vs;
        ctx.lambda2mu[k] = rho * vp * vp;
        ctx.lambda[k] = ctx.lambda2mu[k] - 2.0f * ctx.mu[k];
    }*/
    // 核心修复：将 y 进行垂直镜像，使 LoadScenario 的 y=0 对应 CUDA 网格的最底层 (NZ - 1 - y)
    int k = (ctx.NZ - 1 - y) * ctx.NX + x;
    if (k >= 0 && k < ctx.total_grid) {
        ctx.rho[k] = rho;
        ctx.mu[k] = rho * vs * vs;
        ctx.lambda2mu[k] = rho * vp * vp;
        ctx.lambda[k] = ctx.lambda2mu[k] - 2.0f * ctx.mu[k];
    }
}

// =============================================================================
// 3. 场景装载器：完美适配 11 套高维地球物理仿真场景
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
        int cx = W / 2;
        int cy = H / 2;
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
                    SetMaterialAt(x, y, 4000.0f, 0.0f, 4000.0f); // 液态地核：Vs=0
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
        edit_src_x = W / 2; edit_src_z = H / 2;
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
        edit_src_x = W / 4; edit_src_z = cy;
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
        edit_src_x = W / 2; edit_src_z = H / 2 + 50;
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
                    SetMaterialAt(x, y,100.0f, 60.0f,10000.0f); // 低包层
                }
            }
        }
        edit_src_x = 100; edit_src_z = cy;
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
        edit_src_x = 100; edit_src_z = H / 2;
    }
    else if (type == SCENE_CRYSTAL) {
        // 晶体交错六角声子晶格 (钢球格栅阻碍与干涉波前)
        float bgVp = 1500.0f, bgVs = 0.0f, bgRho = 1000.0f;       // 水背景
        float scVp = 5800.0f, scVs = 3200.0f, scRho = 7800.0f;    // 钢球
        int radius = 12, spacingX = 50, spacingY = 50;
        int startX = 250, endX = W - 100, startY = 100, endY = H - 100;

        for (int i = 0; i < W * H; ++i) SetMaterialAt(i % W, i / W, bgVp, bgVs, bgRho);

        int cols = (endX - startX) / spacingX;
        int rows = (endY - startY) / spacingY;
        float r2 = (float)(radius * radius);

        for (int j = 0; j < rows; ++j) {
            for (int i = 0; i < cols; ++i) {
                int cx = startX + i * spacingX;
                int cy = startY + j * spacingY;
                if (j % 2 == 1) cx += spacingX / 2; // 蜂窝交错

                for (int y = cy - radius; y <= cy + radius; ++y) {
                    for (int x = cx - radius; x <= cx + radius; ++x) {
                        if (x >= 0 && x < W && y >= 0 && y < H) {
                            float dx = (float)(x - cx); float dy = (float)(y - cy);
                            if (dx * dx + dy * dy <= r2) {
                                SetMaterialAt(x, y, scVp, scVs, scRho);
                            }
                        }
                    }
                }
            }
        }
        edit_src_x = 100; edit_src_z = H / 2;
    }
    else if (type == SCENE_RANDOM_SCATTER) {
        // 随机泡泡介质强散射场
        float bgVp = 2500.0f, bgVs = 1200.0f, bgRho = 2000.0f;
        float scVp = 4500.0f, scVs = 2500.0f, scRho = 2800.0f;

        for (int i = 0; i < W * H; ++i) SetMaterialAt(i % W, i / W, bgVp, bgVs, bgRho);

        int numScatterers = 300;
        int minR = 5, maxR = 15;
        for (int k = 0; k < numScatterers; ++k) {
            int cx = rand() % (W - 100) + 50;
            int cy = rand() % (H - 100) + 50;
            int r = rand() % (maxR - minR) + minR;
            float r2 = (float)(r * r);

            for (int y = cy - r; y <= cy + r; ++y) {
                for (int x = cx - r; x <= cx + r; ++x) {
                    if (x >= 0 && x < W && y >= 0 && y < H) {
                        float dx = x - cx; float dy = y - cy;
                        if (dx * dx + dy * dy <= r2) SetMaterialAt(x, y, scVp, scVs, scRho);
                    }
                }
            }
        }
        edit_src_x = W / 2; edit_src_z = H / 2;
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
        edit_src_x = W / 2; edit_src_z = H - 50;
    }
    else if (type == SCENE_REFRACTION) {
        // 速度连续线性增加的梯度介质 (折射和回转回弹波)
        float vTop = 2000.0f, vBot = 5500.0f, rhoTop = 1800.0f, rhoBot = 3000.0f;
        for (int y = 0; y < H; ++y) {
            float ratio = (float)y / (float)H;
            float vp = vBot + (vTop - vBot) * ratio;
            float vs = vp / 1.732f;
            if (vp < 1600.0f) vs = 0.0f;
            float rho = rhoBot + (rhoTop - rhoBot) * ratio;

            for (int x = 0; x < W; ++x) SetMaterialAt(x, y, vp, vs, rho);
        }
        edit_src_x = 100; edit_src_z = H - 50; // 浅层激发
    }
    else if (type == SCENE_PENROSE_ROOM) {
        // 彭罗斯椭圆房间 (展示波形双焦点汇聚透镜效应)
        float rockVp = 3000.0f, rockVs = 1732.0f, rockRho = 2500.0f;
        float airVp = 0.0f, airVs = 0.0f, airRho = 20.0f;
        float cx = W * 0.5f, cy = H * 0.5f;
        float s = std::min(W, H) * 0.38f;

        auto isInsidePenrose = [&](float px, float py) -> bool {
            float x = (px - cx) / s; float y = (py - cy) / s;
            float room_half_width = 1.5f;
            bool top = (y < -0.4f) && ((x * x) / (room_half_width * room_half_width) + pow((y + 0.4f) / 0.7f, 2) <= 1.0f);
            bool bottom = (y > 0.4f) && ((x * x) / (room_half_width * room_half_width) + pow((y - 0.4f) / 0.7f, 2) <= 1.0f);
            bool middle_rect = (abs(y) <= 0.4f) && (abs(x) <= room_half_width);
            bool room_base = top || bottom || middle_rect;

            float mush_pos_x = 1.3f;
            auto is_mushroom = [&](float mx, float my) -> bool {
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
        edit_src_x = static_cast<int>(cx); edit_src_z = static_cast<int>(cy);
    }
}
// =============================================================================
// 新增：高保真物理场纹理上传器 (将 CPU 端的非均匀物性打包高速送入 OpenGL)
// =============================================================================
inline void UpdateModelTexture() {
    if (g_modelTex == 0) {
        glGenTextures(1, &g_modelTex);
    }

    // 分配一个 RGBA32F 浮点型临时缓冲区 (对应 ctx.total_grid 个像素)
    std::vector<float> temp_data(ctx.total_grid * 4, 0.0f);

    for (int i = 0; i < ctx.total_grid; ++i) {
        float rho = ctx.rho[i];
        float vp = 0.0f;
        float vs = 0.0f;
        if (rho > 0.0f) {
            vp = sqrtf(ctx.lambda2mu[i] / rho);
            vs = sqrtf(ctx.mu[i] / rho);
        }

        // 归一化后存入通道
        temp_data[i * 4 + 0] = vp / 6000.0f;     // R 通道: 纵波速度 Vp (映射至 6000m/s)
        temp_data[i * 4 + 1] = vs / 3500.0f;     // G 通道: 横波速度 Vs
        temp_data[i * 4 + 2] = rho / 3000.0f;    // B 通道: 密度 Rho
        temp_data[i * 4 + 3] = 1.0f;             // A 通道
    }

    glBindTexture(GL_TEXTURE_2D, g_modelTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, ctx.NX, ctx.NZ, 0, GL_RGBA, GL_FLOAT, temp_data.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
}
// 一键应用场景 (完美同步修正版)
inline void ApplyScenario(int type, SimState& state) {
    state.running = false;
    current_it = 0;
    accumulated_compute_time = 0.0f;
    active_sources.clear(); // 清空多点激发队列

    // 1. 确保 CPU 物理场上下文 NX/NZ、步长、c1_h等完全建立
    setupTestContext(ctx, par);
    // 2. 根据场景类型，精确定制重写 CPU 数组（此时 rho, mu, lambda2mu 被写入地壳/水层等真实参数）
    LoadScenario(type);
    // =============================================================================
    // 【核心修复 1】：由于 LoadScenario 重置了物性参数，必须在这里重新根据新场景的
    //  实际最大纵波速度，重新计算并匹配最新、最完美的 PML 阻尼吸收线！
    // =============================================================================
    UpdatePmlDampingArrays();

    // =============================================================================
    // 【核心修复 2】：重新计算物理对齐的 dp_flat (保持不变)
    // =============================================================================
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
    
    // 3. 注销并重装 GPU 显存
    freeGPUSimulation(gpu_data);
    initGPUSimulation(gpu_data, ctx);

    // 4. 同步更新地质背景图纹理
    UpdateModelTexture();
    RecalculateCachedModelMetrics();
}

// --- Open File Dialog ---
inline bool OpenSystemFileDialog(char* buffer, int bufferSize) {
    // Clear the buffer
    memset(buffer, 0, bufferSize);

#ifdef _WIN32
    // Windows Implementation
    OPENFILENAMEA ofn;
    memset(&ofn, 0, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFile = buffer;
    ofn.nMaxFile = bufferSize;
    ofn.lpstrFilter = "Model Data (*.txt;*.sgy;*.csv)\0*.txt;*.sgy;*.segy;*.csv\0All Files (*.*)\0*.*\0";
    ofn.nFilterIndex = 1; // 默认选中第一个 (Model Data)
    ofn.nFilterIndex = 1;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
    return GetOpenFileNameA(&ofn);
#else
    // Linux Implementation (requires zenity)
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
            buffer[len - 1] = '\0'; // Remove trailing newline
        }
        success = (strlen(buffer) > 0);
    }
    pclose(pipe);
    return success;
#endif
}

#ifdef _WIN32
// =============================================================================
// 新增：Windows 原生保存文件对话框 (安全保护：OFN_NOCHANGEDIR 杜绝工作路径被篡改)
// =============================================================================
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
// 1. 【重构版】：列优先 (Column-Major) 2D 物理模型网格重采样算法 (最近邻插值)
// =============================================================================
inline std::vector<std::vector<float>> ResampleGrid2D(
    const std::vector<std::vector<float>>& input,
    float target_dx,
    float original_dx
) {
    if (input.empty() || input[0].empty()) return {};

    int old_nx = static_cast<int>(input.size());       // 原始 X 点数 (Traces)
    int old_nz = static_cast<int>(input[0].size());    // 原始 Z 点数 (Samples)

    // 计算实际地质体的物理宽高 [3]
    float physical_width = (old_nx - 1) * original_dx;
    float physical_depth = (old_nz - 1) * original_dx;

    // 计算对应 target_dx 步长下的新网格点数
    int new_nx = static_cast<int>(physical_width / target_dx) + 1;
    int new_nz = static_cast<int>(physical_depth / target_dx) + 1;

    std::cout << "[Resample] Resampling grid: " << old_nx << "x" << old_nz
        << " -> " << new_nx << "x" << new_nz << " (spacing: " << original_dx << "m -> " << target_dx << "m)" << std::endl;

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
// 【终极物理优化】：列优先 2D 物理地层边界外推扩边算法 (带流体安全固化机制)
//  1. 采用 Clamp 边缘外推，保证 Vp 和 密度在 PML 交界处 100% 物理连续，实现 P 波零反射。
//  2. 【核心稳定】：自动检测外推进入 PML 内部的网格。若该网格为流体 (Vs=0, 如水层)，
//     强制将其 Vs 提升为 Vp / 2，使 PML 运行在稳定的“固体”介质上，彻底消灭流固 PML 发散奇点！
// =============================================================================
inline std::vector<std::vector<float>> PadPmlToGrid2D(
    const std::vector<std::vector<float>>& grid_vp,
    const std::vector<std::vector<float>>& grid_vs,
    const std::vector<std::vector<float>>& grid_rho,
    int npml,
    int component_type // 0: Vp, 1: Vs, 2: Rho
) {
    if (grid_vp.empty() || grid_vp[0].empty()) return {};

    int old_nx = static_cast<int>(grid_vp.size());
    int old_nz = static_cast<int>(grid_vp[0].size());

    int new_nx = old_nx + 2 * npml;
    int new_nz = old_nz + 2 * npml;

    std::vector<std::vector<float>> output(new_nx, std::vector<float>(new_nz, 0.0f));

    for (int x = 0; x < new_nx; ++x) {
        int src_x = std::max(0, std::min(x - npml, old_nx - 1));
        bool in_x_pml = (x < npml) || (x >= new_nx - npml);

        for (int z = 0; z < new_nz; ++z) {
            int src_z = std::max(0, std::min(z - npml, old_nz - 1));
            bool in_z_pml = (z < npml) || (z >= new_nz - npml);

            // 获取外推对齐后的原始物理量
            float vp = grid_vp[src_x][src_z];
            float vs = grid_vs[src_x][src_z];
            float rho = grid_rho[src_x][src_z];

            // 智能量纲转换 (克/立方厘米 转 千克/立方米)
            if (rho > 0.0f && rho < 10.0f) rho *= 1000.0f;
            if (rho < 10.0f) rho = 1000.0f;

            // =================================================================
            // 【核心安全保护】：如果该点处于 PML 内部，且原先是水层（Vs == 0）
            //  我们在 PML 内部强行将其“固化”为剪切波速为 Vp/2 的稳定固体介质！
            // =================================================================
            if (in_x_pml || in_z_pml) {
                if (vs <= 0.0f && vp > 350.0f) {
                    vs = vp / 2.0f; // 稳定剪切速度
                }
            }

            // 根据当前请求的导出分量类型，返回对应的物理值
            if (component_type == 0)      output[x][z] = vp;
            else if (component_type == 1) output[x][z] = vs;
            else                          output[x][z] = rho;
        }
    }

    return output;
}

// =============================================================================
// 升级版 TXT 导入器：全面引入列优先重采样与 PML 扩边流体稳定化管线
//  1. 解析 '# Dimensions' 头部后，将数据统一读入临时 CPU 2D 数组。
//  2. 自动进行重采样 (对齐系统当前 dx) 与边缘物性平滑外推 (Pad & Fluid Stabilizer)。
//  3. 安全重设 OpenGL 纹理尺寸，一键重装 GPU 显存，彻底消灭由于边界漏波或剪切模量为 0 导致的发散！
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
    int file_w = 0, file_h = 0;
    bool dimensionsFound = false;
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

    // =============================================================================
    // 【核心新增 1】：动态重采样 (对齐系统当前的空间步长 par.model.dx)
    //  这里假定您的自定义 TXT 模型原始间距为 1.0m (您可以根据您的 Python 脚本输出调整)
    // =============================================================================
    float original_txt_dx = 1.0f;
    float target_system_dx = par.model.dx;

    auto grid_vp = ResampleGrid2D(raw_vp, target_system_dx, original_txt_dx);
    auto grid_vs = ResampleGrid2D(raw_vs, target_system_dx, original_txt_dx);
    auto grid_rho = ResampleGrid2D(raw_rho, target_system_dx, original_txt_dx);

    // =============================================================================
    // 【核心新增 2】：PML 扩边与流体安全固化 (一气呵成)
    //  外推边界的同时，自动将 PML 阻尼层内部的水层（Vs=0）固化为剪切波速非零的固体，根治发散！
    // =============================================================================
    int current_npml = static_cast<int>(par.FDM.npml);
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

            int targetX = x; // 如需要，在此处亦可加上水平翻转：flipHorizontally ? (ctx.NX - 1 - x) : x

            int targetZ = z;
            if (flipVertically) {
                targetZ = ctx.NZ - 1 - z; // 垂直地表对齐
            }

            SetMaterialAt(targetX, targetZ, vp, vs, rho);
        }
    }

    // 5. 重置震源位置与表面波 dp_flat
    ctx.src_z_idx = ctx.NZ / 4;
    ctx.src_idx = ctx.src_z_idx * ctx.NX + (ctx.NX / 2);
    edit_src_x = ctx.NX / 2;
    edit_src_z = ctx.NZ / 4;

    // 核心修复：重新计算并标定最高波速下的 PML 阻尼系数 [3]
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
// 【高级模型加载器】：从 3 个标准的二维 SEG-Y 文件中同时载入地层物理常数 (Vp, Vs, Rho)
//  1. 自动对齐并验证三个 SGY 文件的 NX、NZ 尺寸是否一致
//  2. 100% 物理对齐：利用 flipVertically 确保地表 Z=0 完美对齐到视口顶端
//  3. 安全注销并重整 OpenGL 纹理尺寸，重装 GPU 显存，防止 CUDA 运行期崩溃
// =============================================================================
// =============================================================================
// 升级版导入器：支持水平 (flipHorizontally) 和垂直 (flipVertically) 双向翻转
// =============================================================================
inline bool LoadModelFromSegy(
    const std::string& vp_path,
    const std::string& vs_path,
    const std::string& rho_path,
    bool flipVertically,
    bool flipHorizontally,
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

    // 2. 【一气呵成】：PML 扩边 + 100% P波阻抗匹配 + 水层安全固化
    int current_npml = static_cast<int>(par.FDM.npml);
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

            int targetX = x;
            if (flipHorizontally) targetX = ctx.NX - 1 - x;

            int targetZ = z;
            if (flipVertically)   targetZ = ctx.NZ - 1 - z;

            SetMaterialAt(targetX, targetZ, vp, vs, rho);
        }
    }
    // 4. 重置激发位置与 PML 表面波 dp_flat (保持不变)
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
// 终极版导出器：支持空间局部裁剪（Cropping）与水平翻转导出
//  1. 自动对裁剪坐标进行安全钳位 (Clamp)，防止越界崩溃。
//  2. 自动根据裁剪后的新尺寸 (new_NX, new_NZ) 重新计算并创建标准的二进制 SEG-Y 道头。
//  3. 完美结合水平翻转 (flipHorizontally)，只针对裁剪出的子区域执行翻转映射。
// =============================================================================
inline bool ExportModelToSegy(
    const std::string& base_filepath,
    bool density_gcm3,
    bool flipHorizontally,
    int x_min, int z_min, // 左上角格点坐标 [1.2.7]
    int x_max, int z_max  // 右下角格点坐标
) {
    if (ctx.total_grid <= 0) return false;

    // 1. 安全边界检查与自适应纠错 (防止用户输入颠倒或越界坐标) [4]
    x_min = std::clamp(x_min, 0, ctx.NX - 1);
    x_max = std::clamp(x_max, 0, ctx.NX - 1);
    z_min = std::clamp(z_min, 0, ctx.NZ - 1);
    z_max = std::clamp(z_max, 0, ctx.NZ - 1);

    if (x_min > x_max) std::swap(x_min, x_max);
    if (z_min > z_max) std::swap(z_min, z_max);

    // 计算裁剪后的新模型尺寸 (新宽度、新深度)
    int new_NX = x_max - x_min + 1;
    int new_NZ = z_max - z_min + 1;
    size_t new_total_grid = static_cast<size_t>(new_NX) * new_NZ;

    std::cout << "[IO] Preparing Cropped SEGY Export..." << std::endl;
    std::cout << "     - Crop Range: X[" << x_min << " ~ " << x_max << "], Z[" << z_min << " ~ " << z_max << "]" << std::endl;
    std::cout << "     - Output Size: " << new_NX << "x" << new_NZ << " (Total: " << new_total_grid << " cells)" << std::endl;

    // 自动剔除原有后缀并格式化
    std::string clean_path = base_filepath;
    size_t dot_pos = clean_path.find_last_of('.');
    if (dot_pos != std::string::npos) {
        clean_path = clean_path.substr(0, dot_pos);
    }
    std::string out_vp_path = clean_path + "_vp.sgy";
    std::string out_vs_path = clean_path + "_vs.sgy";
    std::string out_rho_path = clean_path + "_rho.sgy";

    // 创建裁剪后的 1D 导出缓冲区
    std::vector<float> flat_vp(new_total_grid, 0.0f);
    std::vector<float> flat_vs(new_total_grid, 0.0f);
    std::vector<float> flat_rho(new_total_grid, 0.0f);

    // 2. 核心算法：高并发/空间转置与裁剪重组
    for (int x = 0; x < new_NX; ++x) {

        // 如果开启了水平翻转，则对裁剪出的子区域进行左右镜像
        int targetX = flipHorizontally ? (x_max - x) : (x_min + x);

        for (int z = 0; z < new_NZ; ++z) {
            int targetZ = z_min + z; // 对齐深度

            int k_cuda = targetZ * ctx.NX + targetX; // 读取全网格中对应的位置
            int k_segy = x * new_NZ + z;             // 写入新模型的 SEGY 道内位置

            float rho_val = ctx.rho[k_cuda];
            float vp_val = 0.0f;
            float vs_val = 0.0f;

            if (rho_val > 0.0f) {
                vp_val = sqrtf(ctx.lambda2mu[k_cuda] / rho_val);
                vs_val = sqrtf(ctx.mu[k_cuda] / rho_val);
            }

            flat_vp[k_segy] = vp_val;
            flat_vs[k_segy] = vs_val;

            if (density_gcm3) {
                flat_rho[k_segy] = rho_val / 1000.0f;
            }
            else {
                flat_rho[k_segy] = rho_val;
            }
        }
    }

    float dummy_dt = 0.001f;
    SeismicIO::writeSegyFile2D(flat_vp, new_NX, new_NZ, out_vp_path, dummy_dt);
    SeismicIO::writeSegyFile2D(flat_vs, new_NX, new_NZ, out_vs_path, dummy_dt);
    SeismicIO::writeSegyFile2D(flat_rho, new_NX, new_NZ, out_rho_path, dummy_dt);

    std::cout << "[IO] Cropped Model Export Completed." << std::endl;
    return true;
}

// =============================================================================
// 新增：标准的 SEG-Y 地震记录数据装载器
//  1. 调用 SeismicIO 一键读入道数据
//  2. 自动从二进制卷头 3216 字节处抽取大端序采样率 dt 并自动换算为秒，无需人工输入！
//  3. 实时计算全局最大振幅用于增益自适应归一化 [3]
// =============================================================================
inline bool LoadSeismicSegy(const std::string& filepath, AnalysisState& state) {
    // 1. 调用您的标准 SEGY 2D 接口读取
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

    // 3. 计算全局最大振幅用于 Wiggle/Heatmap 动态边界自适应 [3]
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

// 惰性更新一维 Heatmap 矩阵的辅助函数
inline void UpdateHeatmapData(AnalysisState& state) {
    state.heatmapData.resize(state.numSamples * state.numChannels);
    for (int r = 0; r < state.numSamples; ++r) {
        for (int c = 0; c < state.numChannels; ++c) {
            // 将 [channel_idx][sample_idx] 映射为行优先矩阵，完美适配 ImPlot::PlotHeatmap
            state.heatmapData[r * state.numChannels + c] = state.traces[c][r];
        }
    }
}
// 视口 LOD 高效回调画笔
inline ImPlotPoint SeismicDataGetter(int idx, void* data) {
    PlotContext* ctx = reinterpret_cast<PlotContext*>(data);
    int sampleIdx = ctx->startSample + idx * ctx->step;
    float val = (*ctx->traceData)[sampleIdx];

    // 振幅横向展开偏移
    double x = ctx->offsetX + val * ctx->gain;
    double y = ctx->useTime ? (sampleIdx * ctx->dt) : (double)sampleIdx;

    return ImPlotPoint(x, y);
}
inline AnalysisState g_analyzerState; // 全局唯一的分析器状态实例
// =============================================================================
// 升级版：高保真 SEG-Y 探查数据分析器窗口 (Seismic Data Analyzer)
// =============================================================================
void RenderAnalysisWindow(AnalysisState& state) {
    if (!state.isOpen) return;

    ImGui::SetNextWindowSize(ImVec2(1000, 700), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Seismic Data Analyzer", &state.isOpen)) {

        static bool useTimeAxis = true;

        // ============================================================
        // 1. 顶部工具栏 (改用原生 SEG-Y 选择与自动解析加载)
        // ============================================================
        if (ImGui::Button("Load SEG-Y Data")) {
            char filePath[1024] = { 0 };
            if (OpenSystemFileDialog(filePath, sizeof(filePath))) {

                // 一键载入标准的二进制 SEG-Y 数据组
                if (LoadSeismicSegy(filePath, state)) {
                    // 自动计算 Wiggle 增益
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
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Reset axes to full data range.");

        ImGui::SameLine();
        ImGui::TextDisabled("| %d Traces x %d Samples | MaxAmp: %.2e | dt: %.5f s",
            state.numChannels, state.numSamples, state.globalMaxAmp, state.samplingInterval);

        // ============================================================
        // 2. 参数控制面板
        // ============================================================
        if (ImGui::CollapsingHeader("Display Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Columns(2, "viz_cols", false);

            // --- 左列：Wiggle 线条绘制设置 ---
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

            // --- 右列：Heatmap 密度着色设置 ---
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

            // --- 底部：物理采样参数（由于从 SEGY 中自动提取，这里设为只读以体现专业性） ---
            ImGui::Separator();
            ImGui::Text("Physical Parameters:");
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.8f, 1.0f), "Sampling Interval (dt): %.5f s  (Auto-Parsed from SEGY Header)", state.samplingInterval);
        }

        ImGui::Separator();

        // ============================================================
        // 3. 核心绘图区 (完全复用高性能 Plotting 逻辑)
        // ============================================================
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
                ImPlot::SetupAxis(ImAxis_Y1, NULL, ImPlotAxisFlags_Invert); // 标杆：0 深度在上

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
                    if (state.colormapIndex == 1) mapID = ImPlotColormap_PiYG;
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
                    ImVec2 plotSize = ImPlot::GetPlotSize();

                    // 空间水平视口剔除
                    int startCh = (int)floor(limits.X.Min) - 1;
                    int endCh = (int)ceil(limits.X.Max) + 1;
                    if (startCh < 0) startCh = 0;
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
                    if (startSample < 0) startSample = 0;
                    if (endSample > state.numSamples) endSample = state.numSamples;

                    if (startCh < endCh && startSample < endSample) {
                        // 动态 LOD 采样滤波，防止高采样率混叠
                        double visibleSamples = endSample - startSample;
                        int step = 1;
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
                            ImGui::PopID();
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
                IM_COL32(128, 128, 128, 255), "No SEG-Y Data Loaded");
        }
    }
    ImGui::End();
}
// =============================================================================
// 4. 模块化子函数一：初始化显存与网格资源
// =============================================================================

// =============================================================================
// 5. 模块化子函数一：初始化显存与网格资源
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
// 6. 模块化子函数二：物理与时间自适应对齐
// =============================================================================
inline void ApplyCameraAutoAlignment(int winW, int winH, float barHeight) {
    if (first_align_needed && winH > 0 && ctx.NX > 0) {
        float winAspect = (float)winW / (float)winH;
        float simAspect = (float)ctx.NX / (float)ctx.NZ;
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
// 7. 模块化子函数三：鼠标点击/按压及键盘信号交互处理
// =============================================================================
inline void HandleSeismicInteractions(SimState& state, int winW, int winH, float barHeight) {
    ImGuiIO& io = ImGui::GetIO();

    // A. 响应键盘 R 键或 C 键重置信号 (Command Pattern) [1.2.7]
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

    // B. 响应 R 键（或按钮）重置视角
    if (g_resetViewportRequested) {
        viewZoom = 1.0f;

        float winAspect = (float)winW / (float)winH;
        float simAspect = (float)ctx.NX / (float)ctx.NZ;
        ImVec2 aspectCorr = { 1.0f, 1.0f };
        if (winAspect > simAspect) {
            aspectCorr.x = winAspect / simAspect;
        }
        else {
            aspectCorr.y = simAspect / winAspect;
        }

        // 精确对齐全屏左上角
        viewOffset.x = aspectCorr.x - 1.0f;
        viewOffset.y = 1.0f - aspectCorr.y * (1.0f - 2.0f * barHeight / (float)winH);

        g_resetViewportRequested = false; // 消费信号，自动复位
        active_sources.clear();
    }

    // B. 计算屏幕比例裁剪
    float winAspect = (float)winW / (float)winH;
    float simAspect = (float)ctx.NX / (float)ctx.NZ;
    ImVec2 aspectCorr = { 1.0f, 1.0f };
    if (winAspect > simAspect) aspectCorr.x = winAspect / simAspect;
    else                         aspectCorr.y = simAspect / winAspect;

    // C. 鼠标左键激发 (支持单点点击和多点连续按压拖动) [1.2.7]
    static double last_inject_time = 0.0;
    double current_time = glfwGetTime();
    bool trigger_active = false;

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

    if (!io.WantCaptureMouse && trigger_active) {
        float normX = io.MousePos.x / (float)winW;
        float normY = io.MousePos.y / (float)winH;

        float worldX = (normX - 0.5f) * (aspectCorr.x / viewZoom) + 0.5f + (0.5f * viewOffset.x);
        float worldY = (normY - 0.5f) * (aspectCorr.y / viewZoom) + 0.5f - (0.5f * viewOffset.y);

        int mx = (int)(worldX * (float)ctx.NX);
        int my = (int)(worldY * (float)ctx.NZ);

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

    // D. 鼠标中键拖拽平移与滚轮缩放 [2]
    if (!io.WantCaptureMouse) {
        if (io.MouseWheel != 0) {
            float mouseSpeed = 0.1f * viewZoom;
            viewZoom += io.MouseWheel * mouseSpeed;
            if (viewZoom < 0.1f) viewZoom = 0.1f;
            if (viewZoom > 100.0f) viewZoom = 100.0f;
        }
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
            viewOffset.x -= 2 * io.MouseDelta.x / (winW * viewZoom);
            viewOffset.y += 2 * io.MouseDelta.y / (winH * viewZoom);
        }
    }
    ImVec2 mousePos = ImGui::GetMousePos();
    // E. 鼠标悬停位置逆向映射与物性提取 (用于 BottomBar 监控)
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

// =============================================================================
// 8. 模块化子函数四：执行 CUDA 有限差分时演更新
// =============================================================================
inline void UpdateWavefieldSimulation(SimState& state) {
    ImGuiIO& io = ImGui::GetIO();
    bool can_run = state.running && (infinite_mode || current_it < ctx.nt);

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
                current_it++;
            }
        }
    }
}

// =============================================================================
// 9. 模块化子函数五：OpenGL 渲染全屏背景波场纹理
// =============================================================================
inline void RenderFullBackbufferWavefield(int winW, int winH, float barHeight, GLHandles& gl) {
    // 1. 显存色彩映射与 2D copy 写入纹理
    cudaGraphicsMapResources(1, &gl.cudaSeisRes, 0);
    cudaGraphicsSubResourceGetMappedArray(&gl.cudaSeisArray, gl.cudaSeisRes, 0, 0);

    static uchar4* d_rgba_out = nullptr;
    static int last_size = 0;
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

    // =============================================================================
    // 【核心修复】：激活单元 1 并绑定地质物性纹理，告诉着色器开始读取真实的二维层位 [1.2.7]
    // =============================================================================
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, g_modelTex);
    glUniform1i(glGetUniformLocation(gl.seisProg, "modelTexture"), 1);

    // 传递地震及地质背景相关的 Uniform 变量 [1.2.7]
    glUniform1i(glGetUniformLocation(gl.seisProg, "waveStyle"), waveStyle);
    glUniform1i(glGetUniformLocation(gl.seisProg, "modelStyle"), modelStyle); // <-- 传递背景风格 [1.2.7]
    // 实时将 C++ 的开关状态传递给着色器 [1.2.7]
    glUniform1i(glGetUniformLocation(gl.seisProg, "showGrid"), showGrid ? 1 : 0);

    // 【关键】：无缝向下传递当前滑块设置的 Vp 和 密度 (Rho)，建立物理场关联 [3]
    // 假定使用均匀模型：从当前静态滑块中获取实时 Vp/Rho（支持后续加载 SEGY 的二维纹理接口）
    glUniform1f(glGetUniformLocation(gl.seisProg, "uniformVp"), edit_Vp);
    glUniform1f(glGetUniformLocation(gl.seisProg, "uniformRho"), edit_Density);
    glUniform1i(glGetUniformLocation(gl.seisProg, "useModelTexture"), 1); // 暂不开启非均匀纹理，使用滑块数据

    glUniform2f(glGetUniformLocation(gl.seisProg, "winSize"), (float)winW, (float)winH);
    glUniform2f(glGetUniformLocation(gl.seisProg, "simSize"), (float)ctx.NX, (float)ctx.NZ);
    glUniform2f(glGetUniformLocation(gl.seisProg, "viewOffset"), viewOffset.x, viewOffset.y);
    glUniform1f(glGetUniformLocation(gl.seisProg, "viewZoom"), viewZoom);
    glUniform1f(glGetUniformLocation(gl.seisProg, "totalTime"), (float)glfwGetTime());
    glUniform1f(glGetUniformLocation(gl.seisProg, "npml"), (float)ctx.npml);

    // 绑定并载入纹理
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, gl.seisTex);
    glUniform1i(glGetUniformLocation(gl.seisProg, "seisTexture"), 0);

    glBindVertexArray(gl.quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}

// =============================================================================
// 10. 模块化子函数六：绘制 UI 面板、标尺、BottomBar 监控条及星空准星
// =============================================================================
inline void RenderSeisHUD(SimState& state, int winW, int winH, float barHeight, float scale, GLHandles& gl, const GpuInfo& info) {
    ImGuiIO& io = ImGui::GetIO();

    // A. 绘制主缓冲区自适应标尺 (在波场之上，ImGui 控制面板之下)
    RenderGridRulerOnBackbuffer(ctx, state, winW, winH, barHeight, viewOffset, viewZoom);

    // B. 绘制高亮科技鼠标光标 (Fg 图层)
    ViewportInfo vp;
    vp.x = 0.0f; vp.y = 0.0f; vp.w = (float)winW; vp.h = (float)winH;
    vp.scaleX = vp.w / (float)ctx.NX;
    vp.scaleY = vp.h / (float)ctx.NZ;
    RenderBrushCursor(state, vp);

    // C. 绘制当前编辑/激发的物理震源橙色准星
    RenderSourceMarker(ctx, edit_src_x, edit_src_z, winW, winH, barHeight, viewOffset, viewZoom);

    // D. 渲染顶部 TopBar
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
                int minutes = (int)(accumulated_compute_time) / 60;
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

    // E. 渲染悬浮控制面板 (LAB_CONTROLS)
    if (showHUD) {
        ImGui::SetNextWindowPos({ 30 * scale, 80 * scale }, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize({ 420 * scale, 780 * scale }, ImGuiCond_FirstUseEver);

        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.95f, 1.0f, 0.98f, 0.3f)); // 半透明绿色
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
                    ImGui::SliderFloat("Grid Spacing (dx)", &temp_dx, 0.1f, 20.0f, "%.1f m");
                    // 【回归】：PML 吸收层宽度实时滑块控制 [1.2.7]
                    ImGui::SliderInt("PML Layer Width", &temp_pml, 10, 100, "%d px");
                    ImGui::SliderInt("Max Steps (nt)", &temp_nt, 500, 50000);

                    ImGui::SliderInt("Steps / Frame", &steps_per_frame, 1, 100);

                    float dt_limit = 0.5f * temp_dx / max_vp;
                    bool is_stable = (temp_dt <= dt_limit);
                    ImGui::Text("Max Allowed dt (CFL Limit): %.6f s", dt_limit);
                    if (is_stable) ImGui::TextColored({ 0.2f, 1.0f, 0.4f, 1.0f }, "[ STATUS: CFL STABLE ]");
                    else            ImGui::TextColored({ 1.0f, 0.2f, 0.2f, 1.0f }, "[ STATUS: DIVERGENCE RISK! ]");

                    ImGui::Spacing();
                    if (ImGui::Button("APPLY TIME & SPACING CONTROLS", { -1, 30 * scale })) {
                        ctx.dt = temp_dt;
                        ctx.nt = temp_nt;
                        ctx.h = temp_dx;
                        par.model.dx = temp_dx;
                        par.model.dz = temp_dx;

                        // 【物理重算核心】：将全新设置的 PML 层数写入 context 和 parameters
                        ctx.npml = temp_pml;
                        par.FDM.npml = (float)temp_pml;

                        ctx.c1_h = 1.125022f / temp_dx;
                        ctx.c2_h = -0.04687594f / temp_dx;
                        ctx.c3_h = 0.00416669f / temp_dx;
                        ctx.c4_h = -0.00019234f / temp_dx;
                        // =============================================================================
                        // 【核心修复】：由于 PML 宽度或网格步长变了，在此处强制重新计算并对齐 PML 阻尼曲线
                        // =============================================================================
                        UpdatePmlDampingArrays();
                        ctx.dp_flat.assign(ctx.NX, 0.0f);
                        int fs_idx = ctx.npml; // 自由表面自动对齐全新设置的 PML 深度 [3]
                        for (int j = 0; j < ctx.NX; ++j) {
                            int k = fs_idx * ctx.NX + j;
                            if (k < ctx.total_grid && ctx.lambda2mu[k] > 0.0f) {
                                float l2m = ctx.lambda2mu[k];
                                float lam = ctx.lambda[k];
                                ctx.dp_flat[j] = (l2m * l2m - lam * lam) / l2m;
                            }
                        }

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
                        ImGui::SetTooltip("Check to click and spawn multiple intersecting waves simultaneously\nUncheck for standard single-point reset mode.");
                    }
                    //ImGui::SetCursorPosY(ImGui::GetWindowHeight() - 240 * scale);
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

                    // =========================================================
                    // 【新增】：一键唤醒/打开 SEG-Y 地震数据分析仪按钮 [1.2.7]
                    // =========================================================
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(uiAccent.x * 0.12f, uiAccent.y * 0.32f, uiAccent.z * 0.42f, 0.8f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(uiAccent.x * 0.18f, uiAccent.y * 0.52f, uiAccent.z * 0.62f, 1.0f));
                    if (ImGui::Button("OPEN DATA ANALYZER", { -1, 28 * scale })) {
                        g_analyzerState.isOpen = true; // 发射打开分析窗口信号
                    }
                    ImGui::PopStyleColor(2);
                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem("Source")) {
                    ImGui::Spacing();
                    ImGui::TextColored(uiAccent, "Source Position & Angle");

                    int min_valid_grid = ctx.npml + 5;
                    int max_valid_x = ctx.NX - ctx.npml - 5;
                    int max_valid_z = ctx.NZ - ctx.npml - 5;

                    ImGui::SliderInt("Source X (Grid)", &edit_src_x, min_valid_grid, max_valid_x);
                    ImGui::SliderInt("Source Z (Grid)", &edit_src_z, min_valid_grid, max_valid_z);
                    ImGui::SliderFloat("Force Angle", &edit_angle, -180.0f, 180.0f, "%.1f deg");

                    ImGui::Separator();
                    ImGui::TextColored(uiAccent, "Ricker Wavelet Properties");
                    ImGui::SliderFloat("Peak Freq (f0)", &edit_f0, 5.0f, 300.0f, "%.1f Hz");
                    ImGui::SliderFloat("Time Delay (t0)", &edit_t0, 0.001f, 0.2f, "%.3f s");

                    ImGui::TextDisabled("Waveform Real-Time Preview:");
                    float preview_dt = 0.0005f;
                    std::vector<float> preview_x(200);
                    std::vector<float> preview_y(200);
                    for (int i = 0; i < 200; ++i) {
                        float t = i * preview_dt - edit_t0;
                        float pi2_f0 = M_PI * M_PI * edit_f0 * edit_f0;
                        preview_x[i] = i * preview_dt;
                        preview_y[i] = (1.0f - 2.0f * pi2_f0 * t * t) * expf(-pi2_f0 * t * t);
                    }

                    ImPlot::PushStyleColor(ImPlotCol_FrameBg, IM_COL32(125, 125, 125, 255));
                    ImPlot::PushStyleColor(ImPlotCol_PlotBg, IM_COL32(125, 125, 125, 255));
                    ImPlot::PushStyleColor(ImPlotCol_Line, IM_COL32(0, 255, 120, 255));
                    ImPlot::PushStyleColor(ImPlotCol_AxisGrid, IM_COL32(35, 35, 35, 120));

                    if (ImPlot::BeginPlot("##WaveletPreview", ImVec2(-1, 95 * scale), ImPlotFlags_NoLegend)) {
                        ImPlot::SetupAxes("Time", "Amp", ImPlotAxisFlags_NoTickLabels, ImPlotAxisFlags_NoTickLabels);
                        ImPlot::SetupAxesLimits(0.0f, 200 * preview_dt, -1.1f, 1.1f);
                        ImPlot::PlotLine("Ricker", preview_x.data(), preview_y.data(), 200);
                        ImPlot::EndPlot();
                    }
                    ImPlot::PopStyleColor(4);

                    ImGui::Spacing();
                    if (ImGui::Button("APPLY SOURCE CONFIG", { -1, 30 * scale })) {
                        ctx.src_z_idx = edit_src_z;
                        ctx.src_idx = edit_src_z * ctx.NX + edit_src_x;
                        ctx.src_angle = edit_angle;

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
                        g_resetSimRequested = true;
                    }
                    ImGui::Spacing();
                    ImGui::EndTabItem();
                }
                // =============================================================================
                // 3. 【新重构的 TabItem】：Model & Preset (模型底图、预设场景与外部导入) [1.2.7]
                // =============================================================================
                if (ImGui::BeginTabItem("Model_INPUT")) {
                    ImGui::Spacing();

                    // -------------------------------------------------------------
                    // A. 岩石物理属性基本耦合 (Density Slide)
                    // -------------------------------------------------------------
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

                        g_resetSimRequested = true; // 发射重置信号并重新上传
                    }
                    ImGui::Spacing();

                    // -------------------------------------------------------------
                    // B. 场景预设一键加载区
                    // -------------------------------------------------------------
                    ImGui::Separator();
                    ImGui::TextColored(uiAccent, "GEOPHYSICAL SCENARIO PRESETS");

                    const char* scene_names[] = {
                        "Uniform Medium",            // 默认均匀地层 (0)
                        "Earth Shell & Core",        // 地球核幔分层 (1)
                        "Double Slit Interference",  // 双缝挡板干涉 (2)
                        "2-Layered Medium",          // 双层高速低速分界面 (3)
                        "Straight Waveguide",        // 直条状低速波导通道 (4)
                        "Curved Waveguide",          // 正弦曲线弯曲波导 (5)
                        "Phononic Crystal Hex",      // 六角钢球声子晶体 (6)
                        "Random Scattering",         // 300个随机气泡强散射 (7)
                        "Sinusoidal Interface",      // 正弦起伏分层地质 (8)
                        "Linear Velocity Gradient",  // 连续线性速度梯度 (9)
                        "Penrose Room"               // 彭罗斯椭圆房间聚焦反射 (10)
                    };

                    static int selected_scene = current_scene;
                    ImGui::Combo("Select Preset", &selected_scene, scene_names, IM_ARRAYSIZE(scene_names));
                    // 亮荧光青色/橙色一键应用按钮 (自动继承 uiAccent 主题变色)
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(uiAccent.x * 0.15f, uiAccent.y * 0.5f, uiAccent.z * 0.4f, 0.75f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(uiAccent.x * 0.2f, uiAccent.y * 0.7f, uiAccent.z * 0.5f, 0.9f));
                    ImGui::Spacing();
                    if (ImGui::Button("APPLY SCENARIO & REALLOCATE", { -1, 32 * scale })) {
                        current_scene = selected_scene;

                        // 核心调用：重写 CPU 端弹性常数矩阵
                        ApplyScenario(current_scene, state);

                        // 尺寸滑块强制双向同步 [1.2.7]
                        edit_w = ctx.NX;
                        edit_h = ctx.NZ;

                        // 注销并重设 OpenGL 纹理尺寸，防止显存崩溃
                        cudaGraphicsUnregisterResource(gl.cudaSeisRes);
                        glBindTexture(GL_TEXTURE_2D, gl.seisTex);
                        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, ctx.NX, ctx.NZ, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
                        cudaGraphicsGLRegisterImage(&gl.cudaSeisRes, gl.seisTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

                        first_align_needed = true; // 视口重置吸附
                        g_resetSimRequested = true; // 发射重置信号并重新上传
                    }
                    ImGui::Spacing();
// -------------------------------------------------------------
                    // C. 外部文本模型导入 (双列 Browse 浏览与一键安全加载) [1.2.7]
                    // -------------------------------------------------------------
                    ImGui::Separator();
                    ImGui::TextColored(uiAccent, "EXTERNAL MODEL IMPORT (TXT)");

                    static char txt_file_path[512] = ""; // 静态 TXT 模型文件路径缓冲区 [3]

                    // 1. 绘制路径输入框与右侧浏览按钮 (Symmetrical Layout)
                    ImGui::InputText("TXT File", txt_file_path, IM_ARRAYSIZE(txt_file_path));
                    ImGui::SameLine();
                    if (ImGui::Button("Browse##Txt", ImVec2(65 * scale, 0))) {
                        char filePath[512] = { 0 };
                        if (OpenSystemFileDialog(filePath, sizeof(filePath))) {
                            strcpy_s(txt_file_path, filePath); // 复制路径到文本框缓冲区
                        }
                    }

                    // 用户选择的 Y 轴翻转静态状态变量
                    static bool flipImportY = false;
                    ImGui::Checkbox("Flip Vertically on Import", &flipImportY);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Check this if your external model appears upside down.\nCommon for Python/Matlab exported data.");
                    }

                    ImGui::Spacing();

                    // 2. 独立的加载按钮，点击后开始物理重整与显存拷贝
                    if (ImGui::Button("LOAD CUSTOM MODEL FILE", { -1, 35 * scale })) {
                        // 安全防空检查：确保用户已经通过 Browse 选择了有效路径
                        if (strlen(txt_file_path) > 0) {
                            // 调用我们重构后的 LoadModelFromTxt，传入用户勾选的 flipImportY 开关、gl 和 state！
                            if (LoadModelFromTxt(txt_file_path, flipImportY, gl, state)) {
                                popup_message = "External model loaded successfully!";
                                show_success_popup = true;
                                g_resetSimRequested = true; // 发射重置信号并重新上传
                            }
                            else {
                                popup_message = "Failed to load model file.\nPlease check file formatting or dimensions.";
                                show_error_popup = true;
                            }

                            // 自动切换为 1: 科学地质图背景，并隐藏网格，获得最完美的加载视觉
                            modelStyle = 1;
                            showGrid = false;
                        }
                        else {
                            popup_message = "Error: Please select a TXT model file first!";
                            show_error_popup = true;
                        }
                    }
                    ImGui::Spacing();
                    // =============================================================================
                    // C. 外部标准的 SEG-Y 弹性模型组合导入控制台 [1.2.7]
                    // =============================================================================
                    ImGui::Separator();
                    ImGui::TextColored(uiAccent, "EXTERNAL MODEL IMPORT (SEG-Y GROUP)");

                    // 静态路径缓冲区
                    static char vp_file_path[512] = "";
                    static char vs_file_path[512] = "";
                    static char rho_file_path[512] = "";

                    // Vp 路径选择
                    ImGui::InputText("Vp (.sgy)", vp_file_path, IM_ARRAYSIZE(vp_file_path));
                    ImGui::SameLine();
                    if (ImGui::Button("Browse##Vp", ImVec2(65 * scale, 0))) {
                        char filePath[512] = { 0 };
                        if (OpenSystemFileDialog(filePath, sizeof(filePath))) {
                            strcpy_s(vp_file_path, filePath);
                        }
                    }

                    // Vs 路径选择
                    ImGui::InputText("Vs (.sgy)", vs_file_path, IM_ARRAYSIZE(vs_file_path));
                    ImGui::SameLine();
                    if (ImGui::Button("Browse##Vs", ImVec2(65 * scale, 0))) {
                        char filePath[512] = { 0 };
                        if (OpenSystemFileDialog(filePath, sizeof(filePath))) {
                            strcpy_s(vs_file_path, filePath);
                        }
                    }

                    // Rho 密度路径选择
                    ImGui::InputText("Rho(.sgy)", rho_file_path, IM_ARRAYSIZE(rho_file_path));
                    ImGui::SameLine();
                    if (ImGui::Button("Browse##Rho", ImVec2(65 * scale, 0))) {
                        char filePath[512] = { 0 };
                        if (OpenSystemFileDialog(filePath, sizeof(filePath))) {
                            strcpy_s(rho_file_path, filePath);
                        }
                    }
                    // 【核心新增】：原 SEG-Y 文件道间距 (X 采样间隔) 调节滑块 [3]
                    ImGui::SliderFloat("Original SEGY Spacing (dx)", &edit_segy_dx, 0.01f, 50.0f, "%.3f m");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("The original spatial spacing (trace/sample interval) of your loaded SGY files.\nExample: 0.1m for 30m core samples, 12.5m for standard Marmousi.");
                    }
                    // 垂直翻转开关
                    static bool flipSegyY = true;
                    static bool flipSegyX = false; // <-- 新增：导入水平翻转复选框
                    ImGui::Checkbox("Flip Vertically on SEGY Import", &flipSegyY);
                    ImGui::Checkbox("Flip Horizontally on SEGY Import", &flipSegyX);
                    ImGui::Spacing();

                    // 按钮执行加载
                    if (ImGui::Button("LOAD SEGY MODEL GROUP", { -1, 35 * scale })) {
                        // 确保用户选择了全部三个物性文件
                        if (strlen(vp_file_path) > 0 && strlen(vs_file_path) > 0 && strlen(rho_file_path) > 0) {
                            // 传入新参数 flipSegyX！
                            if (LoadModelFromSegy(vp_file_path, vs_file_path, rho_file_path, flipSegyY, flipSegyX, gl, state)) {
                                popup_message = "SEGY Model Group loaded successfully!";
                                show_success_popup = true;
                                g_resetSimRequested = true; // 发射重置信号并重新上传
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
                        modelStyle = 1;
                        showGrid = false;
                    }
                    ImGui::Spacing();
                    ImGui::PopStyleColor(2);
                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem("Model_EXPORT")) {
                    
                    // =============================================================================
                    // D. 导出当前场景中的非均匀地层模型为标准的 SEG-Y 文件群 (支持空间高精度裁剪) [1.2.7]
                    // =============================================================================
                    ImGui::Separator();
                    ImGui::TextColored(uiAccent, "EXPORT CURRENT MODEL TO SEG-Y GROUP");

                    static bool export_density_gcm3 = true;
                    static bool export_flip_x = false;

                    // 裁剪区域控制的静态局部变量
                    static int crop_x_min = 0;
                    static int crop_z_min = 0;
                    static int crop_x_max = 0;
                    static int crop_z_max = 0;

                    // 【自适应对齐】：当网格尺寸发生变化或首次启动时，默认选择全网格范围导出
                    if (crop_x_max <= 0 || crop_x_max >= ctx.NX || first_align_needed) {
                        crop_x_min = 0;
                        crop_z_min = 0;
                        crop_x_max = ctx.NX - 1;
                        crop_z_max = ctx.NZ - 1;
                    }

                    ImGui::Checkbox("Convert Density to g/cm3 on Export", &export_density_gcm3);
                    ImGui::Checkbox("Flip Horizontally on Export", &export_flip_x);

                    // 空间局部裁剪输入滑条 [1.2.7]
                    ImGui::TextDisabled("Crop Sub-Region Coordinates (Grid Units):");
                    ImGui::SliderInt("X Min (Left)", &crop_x_min, 0, ctx.NX - 1);
                    ImGui::SliderInt("X Max (Right)", &crop_x_max, 0, ctx.NX - 1);
                    ImGui::SliderInt("Z Min (Top)", &crop_z_min, 0, ctx.NZ - 1);
                    ImGui::SliderInt("Z Max (Bottom)", &crop_z_max, 0, ctx.NZ - 1);

                    // 一键对齐便捷按钮设计 [1.2.7]
                    float sub_btn_w = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) / 2.0f;

                    if (ImGui::Button("SET TO FULL GRID", ImVec2(sub_btn_w, 0))) {
                        crop_x_min = 0;
                        crop_z_min = 0;
                        crop_x_max = ctx.NX - 1;
                        crop_z_max = ctx.NZ - 1;
                    }
                    ImGui::SameLine();

                    if (ImGui::Button("STRIP PML BOUNDARY", ImVec2(sub_btn_w, 0))) {
                        // 一键剥离外侧的 PML 阻尼带，仅将最核心、最纯净的中央地质体导出！ [1.2.7]
                        crop_x_min = ctx.npml;
                        crop_z_min = ctx.npml;
                        crop_x_max = ctx.NX - ctx.npml - 1;
                        crop_z_max = ctx.NZ - ctx.npml - 1;
                    }

                    ImGui::Spacing();

                    // 亮荧光色导出按钮
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(uiAccent.x * 0.15f, uiAccent.y * 0.45f, uiAccent.z * 0.55f, 0.75f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(uiAccent.x * 0.2f, uiAccent.y * 0.65f, uiAccent.z * 0.75f, 0.9f));
                    if (ImGui::Button("EXPORT SEG-Y GROUP", { -1, 35 * scale })) {
                        char savePath[512] = { 0 };
                        if (SaveSystemFileDialog(savePath, sizeof(savePath))) {
                            // 核心调用：执行裁剪导出
                            if (ExportModelToSegy(savePath, export_density_gcm3, export_flip_x, crop_x_min, crop_z_min, crop_x_max, crop_z_max)) {
                                popup_message = "Cropped SEGY Model Group exported successfully!\nCheck out your saved _vp, _vs, _rho files.";
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

                if (ImGui::BeginTabItem("visual")) {
                    ImGui::Spacing();
                    ImGui::SliderFloat("Color Gain", &color_scale, 1e0f, 1e3f, "%.1f");
                    const char* components[] = { "Vz (Vertical)", "Vx (Horizontal)" };
                    ImGui::Combo("Show Component", &show_component, components, IM_ARRAYSIZE(components));

                    ImGui::Spacing();
                    const char* style_types[] = { "Magma Glow ", "3D Specular Coolwarm", "Neon Bipolar" };
                    ImGui::Combo("Visual Style", &waveStyle, style_types, IM_ARRAYSIZE(style_types));
                    ImGui::Spacing();

                    // =================================================================
                    // 【新增】：地质模型背景切换控件 [1.2.7]
                    // =================================================================
                    const char* model_styles[] = { "Titanium Grey", "Geological Map", "Grayscale Vp", "Viridis Colormap", "Cyber Neon" };
                    ImGui::Combo("Geological Background", &modelStyle, model_styles, IM_ARRAYSIZE(model_styles));
                    ImGui::Spacing();
                    // =================================================================
                    // 【新增】：背景网格线与发光矩阵交点小圆点的实时控制开关 [1.2.7]
                    // =================================================================
                    ImGui::Checkbox("Show Background Grid & Dots", &showGrid);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Toggle the visibility of the Symmetric Titanium grid lines and glowing intersection dots.");
                    }

                    ImGui::EndTabItem();
                }

                ImGui::EndTabBar();
            }

            
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
                uiAccent = ImVec4(0.5f, 1.0f, 0.5f, 1.0f); // 默认绿色荧光;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Reset UI Accent Color to Default Cyan");
            }
        }
        ImGui::End();

        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(8);
    }
    // =============================================================================
    // 【核心新增】：在 HUD 渲染的最末端，挂载数据分析器主窗口
    //  使其完美享受 g_analyzerState 全局共享信号控制，无需修改 main.cpp！
    // =============================================================================

    // =============================================================================
    // 【 独立监测视窗 】：Simulation & PML Monitor (由主控制台 show_par 开关控制)
    //  功能：只读展示物理场活性参数与 1D 阻尼曲线对齐情况，防高度堆叠，支持独立拖拽
    // =============================================================================
    if (show_Monitor_par) {
        ImGui::SetNextWindowPos(ImVec2(30 * scale, 80 * scale), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(420 * scale, 520 * scale), ImGuiCond_FirstUseEver); // 优化高度为 520，更紧凑

        // 绑定主题色到窗口边框
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.05f, 0.08f, 0.1f, 0.85f));   // 深色钛金背景
        ImGui::PushStyleColor(ImGuiCol_Border, uiAccent);                             // 荧光主题色边框
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.5f);

        // 统一注入主题色到折叠栏标头
        ImGui::PushStyleColor(ImGuiCol_HeaderActive, uiAccent);
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(uiAccent.x * 0.8f, uiAccent.y * 0.8f, uiAccent.z * 0.8f, 0.8f));

        if (ImGui::Begin("Simulation & PML Monitor", &show_Monitor_par)) {
            // A. 顶部只读 FPS 帧率指示
            static float displayFPS = 60.0f;
            displayFPS = displayFPS * 0.95f + io.Framerate * 0.05f;
            ImGui::TextColored(uiAccent, "MONITOR STATE  |  FPS: %.1f", displayFPS);
            ImGui::Separator();

            // =============================================================================
            // 1. 【 实时监控 】：当前运行期参数只读监视器 (ACTIVE PARAMETERS MONITOR)
            //  功能：将用户前台“草稿参数”与显卡底层“真实计算参数”分流
            //  提示：如果不一致，智能触发 6Hz 偏色呼吸闪烁警告 [1.2.7]
            // =============================================================================
            ImGui::Spacing();
            if (ImGui::CollapsingHeader("ACTIVE PARAMETERS MONITOR", ImGuiTreeNodeFlags_DefaultOpen)) {

                // 1. 计算高对比度的呼吸警告色 (闪烁频率 6Hz，平滑正弦插值)
                float time = (float)ImGui::GetTime();
                float alpha = 0.35f + 0.65f * (sinf(time * 6.0f) * 0.5f + 0.5f);
                ImVec4 warningCol = ImVec4(1.0f, 0.82f, 0.0f, alpha); // 琥珀荧光黄 [Apply Pending] [1.2.7]

                // 2. 标定当前物理网格规模 (只读)
                ImGui::TextDisabled("Grid Dimensions:"); ImGui::SameLine();
                ImGui::Text("%d x %d (Total: %d)", ctx.NX, ctx.NZ, ctx.total_grid);

                // 标定物理几何尺度 (长 x 深)
                ImGui::TextDisabled("Physical Scale:"); ImGui::SameLine();
                ImGui::Text("%.1f m x %.1f m", ctx.NX * ctx.h, ctx.NZ * ctx.h);

                // 标定显卡显存分配估计值 (只读)
                ImGui::TextDisabled("Est. GPU Memory:"); ImGui::SameLine();
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.8f, 1.0f), "%.2f MB", cached_vram_mb);

                // 3. 监控空间网格间距 (dx) 状态与对齐检查 [6]
                ImGui::TextDisabled("Active Spacing (dx):"); ImGui::SameLine();
                ImGui::Text("%.1f m", ctx.h);
                if (std::abs(temp_dx - ctx.h) > 1e-4f) {
                    ImGui::SameLine();
                    ImGui::TextColored(warningCol, "[Apply Pending]");
                }

                // 4. 监控时间步长 (dt) 状态与对齐检查 [6]
                ImGui::TextDisabled("Active Time Step (dt):"); ImGui::SameLine();
                ImGui::Text("%.6f s", ctx.dt);
                if (std::abs(temp_dt - ctx.dt) > 1e-7f) {
                    ImGui::SameLine();
                    ImGui::TextColored(warningCol, "[Apply Pending]");
                }

                // 5. 监控最大仿真步数 (nt) 状态与对齐检查
                ImGui::TextDisabled("Active Max Steps (nt):"); ImGui::SameLine();
                ImGui::Text("%d steps (%.2f s simulated)", ctx.nt, ctx.nt * ctx.dt);
                if (temp_nt != ctx.nt) {
                    ImGui::SameLine();
                    ImGui::TextColored(warningCol, "[Apply Pending]");
                }

                // 6. 监控 PML 边界宽度状态与对齐检查
                ImGui::TextDisabled("Active PML Width:"); ImGui::SameLine();
                ImGui::Text("%d px", ctx.npml);
                if (temp_pml != ctx.npml) {
                    ImGui::SameLine();
                    ImGui::TextColored(warningCol, "[Apply Pending]");
                }

                // 7. 监控物理震源网格坐标状态与对齐检查 [1.1.3]
                int active_src_x = ctx.src_idx % ctx.NX;
                int active_src_z = ctx.src_idx / ctx.NX;
                ImGui::TextDisabled("Active Source Pos:"); ImGui::SameLine();
                ImGui::Text("(%d, %d)", active_src_x, active_src_z);
                if (edit_src_x != active_src_x || edit_src_z != active_src_z) {
                    ImGui::SameLine();
                    ImGui::TextColored(warningCol, "[Apply Pending]");
                }

                // =============================================================================
                // 【学术级新增】：动态物性极值指标监控面板 (ACTIVE GEOPHYSICAL METRICS) [3]
                // =============================================================================
                ImGui::Separator();
                ImGui::TextColored(uiAccent, "ACTIVE GEOPHYSICAL METRICS");

                // 纵波速度极值范围
                ImGui::TextDisabled("P-Wave Velocity (Vp):"); ImGui::SameLine();
                ImGui::Text("%.1f m/s ~ %.1f m/s", cached_min_vp, cached_max_vp);

                // 横波速度极值范围
                ImGui::TextDisabled("S-Wave Velocity (Vs):"); ImGui::SameLine();
                ImGui::Text("%.1f m/s ~ %.1f m/s", cached_min_vs, cached_max_vs);

                // 密度极值范围
                ImGui::TextDisabled("Density Range (Rho):"); ImGui::SameLine();
                ImGui::Text("%.1f ~ %.1f kg/m^3", cached_min_rho, cached_max_rho);
            }

            // =============================================================================
            // 2. 【 PML 阻尼曲线与层宽度匹配诊断器 (PML Damping Profile Monitor) 】
            //  利用 ImPlot 实时绘制 1D 阻尼剖面，并绘制垂直荧光边界虚线
            // =============================================================================
            ImGui::Spacing();
            ImGui::TextDisabled("PML Damping Profile (dx) Real-Time Monitor:");

            // 1. 在 CPU 侧生成临时 X 轴网格索引坐标 (0 到 NX)
            std::vector<float> pml_x_axis(ctx.NX);
            for (int j = 0; j < ctx.NX; ++j) {
                pml_x_axis[j] = (float)j;
            }

            // 2. 局部修改图表配色：纯黑高对比底色 + 荧光绿阻尼线 + 琥珀橘边界线 [1.2.7]
            ImPlot::PushStyleColor(ImPlotCol_FrameBg, IM_COL32(10, 12, 15, 255)); // 边界框 (深钛金灰)
            ImPlot::PushStyleColor(ImPlotCol_PlotBg, IM_COL32(0, 0, 0, 255));     // 绘图底色 (纯黑)
            ImPlot::PushStyleColor(ImPlotCol_Line, IM_COL32(0, 255, 120, 255));   // 曲线颜色 (荧光绿)
            ImPlot::PushStyleColor(ImPlotCol_AxisGrid, IM_COL32(35, 35, 35, 120)); // 暗灰网格线

            // 传入 ImPlotFlags_NoLegend 隐藏多余标签，使诊断图更加高雅
            if (ImPlot::BeginPlot("##PmlProfilePlot", ImVec2(-1, 140 * scale), ImPlotFlags_NoLegend)) {
                ImPlot::SetupAxes("Grid Node Index", "Damping (Hz)", ImPlotAxisFlags_NoTickLabels, 0);

                // 动态获取当前最大阻尼作为 Y 轴上限 (预留 10% 顶空，防止曲线贴顶)
                float max_d = 1.0f;
                for (float val : ctx.dx) {
                    if (val > max_d) max_d = val;
                }
                ImPlot::SetupAxesLimits(0.0f, ctx.NX, -max_d * 0.05f, max_d * 1.1f);

                // 绘制阻尼线
                ImPlot::PlotLine("Damping dx", pml_x_axis.data(), ctx.dx.data(), ctx.NX);

                // 核心安全优化：使用底层像素换算，直接绘制两条垂直荧光橘色边界虚线，规避任何 ImPlot 版本冲突 [1.2.7]
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
            ImPlot::PopStyleColor(4); // 弹出 4 个局部颜色
            ImGui::Spacing();
        }
        ImGui::End();

        // 统一弹出样式，避免状态污染
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(4); // 弹出 WindowBg, Border, HeaderActive, HeaderHovered
    }

    RenderAnalysisWindow(g_analyzerState);


    // =============================================================================
    // 【 9. 新增：底部专业状态监控栏 (BottomBar) 】
    // =============================================================================
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
// 9. 主接口：重构后的超精简总调度器 (RenderSeisSimScreen_GPU)
// =============================================================================
inline void RenderSeisSimScreen_GPU(SimState& state, int winW, int winH, GLHandles& gl, const GpuInfo& info) {
    const float scale = (float)winW / 1920.0f;
    const float barHeight = 48.0f * scale;

    InitializeSeismicSimulation(gl);

    ApplyCameraAutoAlignment(winW, winH, barHeight);

    HandleSeismicInteractions(state, winW, winH, barHeight);

    UpdateWavefieldSimulation(state);

    RenderFullBackbufferWavefield(winW, winH, barHeight, gl);

    RenderSeisHUD(state, winW, winH, barHeight, scale, gl, info);
}
