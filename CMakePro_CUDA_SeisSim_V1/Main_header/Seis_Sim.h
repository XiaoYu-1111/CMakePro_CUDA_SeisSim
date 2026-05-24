#pragma once

#include "Common.h"
#include "Cuda_Check.cuh"//cuda计算头文件


/// 声明外部已经配置好的全局或静态计算上下文（用于连接我们之前的 FDM CUDA 逻辑）
static SimulationContext ctx;
static GPUSimData gpu_data;
static bool is_seis_initialized = false;

// =============================================================================
// C++20 共享全局信号（用于连接键盘快捷键与屏幕渲染器的物理重置行为）
// =============================================================================
inline bool g_resetSimRequested = false;      // C 键重置模拟信号
inline bool g_resetViewportRequested = false;  // R 键重置视口信号

void setupTestContext(SimulationContext& ctx, const Parameters& par);

void generateRickerWavelet(std::vector<float>& wavelet, int nt, float dt, float f0, float t0);

// =============================================================================
// 精英版：随 Backbuffer 物理画面【同步移动、缩放、自适应间距】的动态标尺
// =============================================================================
// =============================================================================
// 精英修正版：解决 2 倍尺度和速度漂移 Bug，实现标尺与波场 100% 绝对物理咬合
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

    // 1. 计算宽高比校正系数 (与着色器完全一致)
    float winAspect = (float)winW / (float)winH;
    float simAspect = (float)ctx.NX / (float)ctx.NZ;
    ImVec2 aspectCorr = { 1.0f, 1.0f };
    if (winAspect > simAspect) {
        aspectCorr.x = winAspect / simAspect;
    }
    else {
        aspectCorr.y = simAspect / winAspect;
    }

    // 2. 自适应间距自调节 [1.2.7]
    float spacing = 50.0f;
    if (viewZoom < 0.35f)  spacing = 100.0f;
    if (viewZoom < 0.15f)  spacing = 200.0f;
    if (viewZoom < 0.06f)  spacing = 500.0f;
    if (viewZoom > 2.2f)   spacing = 20.0f;
    if (viewZoom > 5.0f)   spacing = 10.0f;
    if (viewZoom > 12.0f)  spacing = 5.0f;

    // ========================================================
    // A. 绘制顶部 X 轴物理刻度 (100% 像素级对齐)
    // ========================================================
    for (float simX = 0.0f; simX <= ctx.NX; simX += spacing) {

        // 1. 【数学修复】：将格数 simX 先映射到顶点空间的 [-1.0, 1.0] (完美对应 aPos.x)
        float aPos_x = (simX / (float)ctx.NX) * 2.0f - 1.0f;

        // 2. 完美的顶点着色器前向矩阵投影公式
        float Pndc_x = (aPos_x - viewOffset.x) * viewZoom / aspectCorr.x;

        // 3. 将 NDC 坐标线性转换回屏幕实际像素坐标 [1.2.7]
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

    // ========================================================
    // B. 绘制左侧 Y 轴物理深度刻度 (100% 像素级对齐)
    // ========================================================
    for (float simY = 0.0f; simY <= ctx.NZ; simY += spacing) {

        // 1. 【数学修复】：将物理深度 simY 映射到顶点空间（地表 Y 轴翻转，0 刻度对应 1.0）
        float aPos_y = 1.0f - (simY / (float)ctx.NZ) * 2.0f;

        // 2. 矩阵投影公式
        float Pndc_y = (aPos_y - viewOffset.y) * viewZoom / aspectCorr.y;

        // 3. 转回屏幕像素坐标 (ImGui 中 0 在最顶端，1.0 在最底端) [1.2.7]
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
// 2. 移植版：星空/科技主题鼠标画笔光标渲染
// =============================================================================
void RenderBrushCursor(const SimState& state, const ViewportInfo& vp) {
    ImVec2 mousePos = ImGui::GetMousePos();
    bool isInside = (mousePos.x >= vp.x && mousePos.x < (vp.x + vp.w) &&
        mousePos.y >= vp.y && mousePos.y < (vp.y + vp.h));

    // 如果鼠标正在操作 UI（比如点悬浮面板），则不渲染自定义光标
    if (ImGui::GetIO().WantCaptureMouse) return;

    if (isInside) {
        ImDrawList* fg = ImGui::GetForegroundDrawList();

        // 如果没有选择任何画笔（TOOL_NONE），绘制科技感十足的准星
        // 注：若您的 SimState 中没有 brushType，可以直接保留 TOOL_NONE 分支作为炫酷的默认准星
        if (state.brushType == TOOL_NONE) {
            float t = (float)ImGui::GetTime();

            const float crosshairSize = 18.0f;
            const float outerRingRadius = 35.0f;

            ImU32 col_main = IM_COL32(102, 191, 217, 255);      // 科技青
            ImU32 col_dynamic = IM_COL32(255, 230, 51, 255);    // 荧光黄
            ImU32 col_shadow = IM_COL32(13, 20, 56, 150);

            // A. 绘制十字线阴影与亮线
            fg->AddLine(ImVec2(mousePos.x - crosshairSize, mousePos.y), ImVec2(mousePos.x + crosshairSize, mousePos.y), col_shadow, 4.0f);
            fg->AddLine(ImVec2(mousePos.x, mousePos.y - crosshairSize), ImVec2(mousePos.x, mousePos.y + crosshairSize), col_shadow, 4.0f);
            fg->AddLine(ImVec2(mousePos.x - crosshairSize, mousePos.y), ImVec2(mousePos.x + crosshairSize, mousePos.y), col_main, 2.0f);
            fg->AddLine(ImVec2(mousePos.x, mousePos.y - crosshairSize), ImVec2(mousePos.x, mousePos.y + crosshairSize), col_main, 2.0f);

            // B. 静态外环
            fg->AddCircle(mousePos, outerRingRadius, col_main, 32, 1.5f);

            // C. 动态雷达扫描线
            float angle = t * 4.0f;
            ImVec2 scanEnd = ImVec2(mousePos.x + cos(angle) * outerRingRadius,
                mousePos.y + sin(angle) * outerRingRadius);
            fg->AddLine(mousePos, scanEnd, col_dynamic, 2.0f);
            fg->AddCircleFilled(scanEnd, 3.5f, col_dynamic);

            // D. 脉冲扩散光环
            float pulse_t = fmodf(t, 1.5f) / 1.5f;
            float pulse_radius = pulse_t * outerRingRadius;
            int pulse_alpha = (int)(sin(pulse_t * 3.14159f) * 120);
            ImU32 col_pulse = IM_COL32(255, 230, 51, pulse_alpha);
            fg->AddCircle(mousePos, pulse_radius, col_pulse, 32, 3.0f);
        }
        else {
            // 画笔激活状态 (TOOL_HIGH, TOOL_LOW, TOOL_WALL, TOOL_ERASER)
            ImU32 brushColor;
            switch (state.brushType) {
            case TOOL_HIGH:   brushColor = IM_COL32(255, 100, 100, 200); break; // 红色代表高波速
            case TOOL_LOW:    brushColor = IM_COL32(100, 150, 255, 200); break; // 蓝色代表低波速
            case TOOL_WALL:   brushColor = IM_COL32(255, 255, 0, 200);   break; // 黄色代表反射障壁
            case TOOL_ERASER: brushColor = IM_COL32(200, 200, 200, 200); break; // 灰色代表橡皮擦
            default:          brushColor = IM_COL32(255, 255, 255, 200); break;
            }

            // 根据当前缩放比例动态缩放画笔圆圈的实际显示尺寸
            float screenRadius = state.brushRadius * vp.scaleX;

            fg->AddCircle(mousePos, screenRadius, brushColor, 0, 2.0f);
            fg->AddCircle(mousePos, screenRadius - 1.0f, IM_COL32(0, 0, 0, 150), 0, 1.0f);
            fg->AddCircleFilled(mousePos, 3.0f, IM_COL32(255, 255, 255, 255));
        }
    }
}

// =============================================================================
// 新增：高精度随背景平移缩放、无漂移的物理震源准星 (Orange Crosshair)
// =============================================================================
inline void RenderSourceMarker(
    const SimulationContext& ctx,
    int src_x, int src_z,
    int winW, int winH,
    float barHeight,
    const ImVec2& viewOffset,
    float viewZoom
) {
    // 1. 计算宽高比校正系数 (与着色器完全一致)
    float winAspect = (float)winW / (float)winH;
    float simAspect = (float)ctx.NX / (float)ctx.NZ;
    ImVec2 aspectCorr = { 1.0f, 1.0f };
    if (winAspect > simAspect) {
        aspectCorr.x = winAspect / simAspect;
    }
    else {
        aspectCorr.y = simAspect / winAspect;
    }

    // 2. 将格数 (src_x, src_z) 转换到顶点坐标空间的 [-1.0, 1.0] [1.1.3]
    float aPos_x = ((float)src_x / (float)ctx.NX) * 2.0f - 1.0f;
    float aPos_y = 1.0f - ((float)src_z / (float)ctx.NZ) * 2.0f; // 垂直 Y 轴翻转

    // 3. 计算 NDC 与屏幕像素坐标
    float Pndc_x = (aPos_x - viewOffset.x) * viewZoom / aspectCorr.x;
    float screenX = (Pndc_x * 0.5f + 0.5f) * winW;

    float Pndc_y = (aPos_y - viewOffset.y) * viewZoom / aspectCorr.y;
    float screenY = (0.5f - Pndc_y * 0.5f) * winH;

    // 4. 安全范围裁剪 (只在可见视口内且在 TopBar 下方时绘制) [1.2.7]
    if (screenX >= 0.0f && screenX <= winW && screenY >= barHeight && screenY <= winH) {
        ImDrawList* fg = ImGui::GetForegroundDrawList();
        ImVec2 pos(screenX, screenY);

        // 准星样式配置
        ImU32 colMain = IM_COL32(125, 255,125, 255); // 亮橙
        ImU32 colBlack = IM_COL32(0, 0, 0, 180);    // 黑色阴影
        float lineLen = 20.0f;                      // 十字臂长
        float gap = 3.0f;                           // 中心留空（保护视线）

        // A. 绘制十字准星阴影
        fg->AddLine(ImVec2(pos.x - lineLen, pos.y), ImVec2(pos.x + lineLen, pos.y), colBlack, 3.5f);
        fg->AddLine(ImVec2(pos.x, pos.y - lineLen), ImVec2(pos.x, pos.y + lineLen), colBlack, 3.5f);

        // B. 绘制亮橙色十字线
        fg->AddLine(ImVec2(pos.x - lineLen, pos.y), ImVec2(pos.x - gap, pos.y), colMain, 1.5f);
        fg->AddLine(ImVec2(pos.x + gap, pos.y), ImVec2(pos.x + lineLen, pos.y), colMain, 1.5f);
        fg->AddLine(ImVec2(pos.x, pos.y - lineLen), ImVec2(pos.x, pos.y - gap), colMain, 1.5f);
        fg->AddLine(ImVec2(pos.x, pos.y + gap), ImVec2(pos.x, pos.y + lineLen), colMain, 1.5f);

        // C. 绘制网格及物理距离文字 (例: 500, 250 (500m, 250m))
        char buf[64];
        snprintf(buf, 64, "%d, %d (%.0fm, %.0fm)", src_x, src_z, src_x * ctx.h, src_z * ctx.h);

        ImVec2 textPos = ImVec2(pos.x + 8, pos.y + 8);
        fg->AddText(ImVec2(textPos.x + 1, textPos.y + 1), colBlack, buf); // 阴影
        fg->AddText(textPos, colMain, buf);                              // 橙字

        // D. 呼吸圆环
        float t = (float)ImGui::GetTime();
        float pulse = (sin(t * 3.0f) * 0.5f + 0.5f) * 5.0f;
        fg->AddCircle(pos, 5.0f + pulse, IM_COL32(125, 255, 125, 155), 16, 1.0f);
    }
}

void RenderSeisSimScreen_GPU(SimState& state, int winW, int winH, GLHandles& gl, const GpuInfo& info) {
    ImGuiIO& io = ImGui::GetIO();
    const float scale = (float)winW / 1920.0f;
    const float barHeight = 48.0f * scale; // 状态栏高度

    // --- [1. 初始化地震模拟上下文与 GPU 显存] ---
    // --- [ 1. 初始化地震数据与纹理 ] ---
    static bool first_align_needed = false; // 用于标记是否需要执行首次对齐
    static Parameters par; // 使用您默认的参数
    if (!is_seis_initialized) {

        // 默认将模拟分辨率设置为 500x500 
        par.model.xnum = 1000;
        par.model.znum = 500;
        par.model.dx = 1.0f; // 空间步长 1.0m
        par.model.dz = 1.0f;
        par.FDM.f0 = 100.0f;  // 30Hz 配合 1.0m 网格，无频散 [8]
        
        setupTestContext(ctx, par);
        initGPUSimulation(gpu_data, ctx);

        // 初始化对应的 OpenGL 纹理与 CUDA 注册
        glGenTextures(1, &gl.seisTex);
        glBindTexture(GL_TEXTURE_2D, gl.seisTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, ctx.NX, ctx.NZ, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // 注册到 CUDA（WriteOnly，用于零拷贝直接写入纹理）
        cudaGraphicsGLRegisterImage(&gl.cudaSeisRes, gl.seisTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

        is_seis_initialized = true;
        first_align_needed = true; // 数据就绪，标记在下面进行首次精确对齐
    }

    // --- [2. 持久化模拟控制变量] ---
    static int current_it = 0;
    static int accumulated_compute_time = 0;
    static float color_scale = 1e1f;
    static int show_component = 0; // 0: Vz, 1: Vx
    static int steps_per_frame = 10; // 每帧迭代物理步数
    static bool showHUD = true;

    static int waveStyle = 0; // 默认采用极其绚丽的 Style 1

    static int edit_src_x = ctx.NX / 2;
    static int edit_src_z = ctx.NZ / 4;
    static float edit_angle = par.FDM.angle;



    // 【新增】：子波主频 (f0) 与时间延迟 (t0) 的持久化变量，默认对齐启动参数
    static float edit_f0 = 100.0f;
    static float edit_t0 = 1/edit_f0;


    // --- [3. 鼠标缩放与平移逻辑] ---
    static float viewZoom = 1.0f;
    static ImVec2 viewOffset = { 0.0f, 0.0f };
    // -- - [3. 【核心修复】：安全地进行首次左上角自适应吸附] -- -
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

        // 精确计算左上角吸附平移量
        viewOffset.x = aspectCorr.x - 1.0f;
        viewOffset.y = 1.0f - aspectCorr.y * (1.0f - 2.0f * barHeight / (float)winH);
        first_align_needed = false; // 首次对齐完成，关闭标记
    }


    if (!io.WantCaptureMouse) {
        if (io.MouseWheel != 0) {
            float mouseSpeed = 0.1f * viewZoom;
            viewZoom += io.MouseWheel * mouseSpeed;
            if (viewZoom < 0.1f) viewZoom = 0.1f;
            if (viewZoom > 100.0f) viewZoom = 100.0f;
        }
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
            viewOffset.x -= io.MouseDelta.x / (winW * viewZoom);
            viewOffset.y += io.MouseDelta.y / (winH * viewZoom);
        }
    }
    // ============================================================
    // 【完美修正版：鼠标左键点击注入震源】
    // ============================================================
    if (!io.WantCaptureMouse && ImGui::IsMouseClicked(0)) {
        float normX = io.MousePos.x / (float)winW;
        float normY = io.MousePos.y / (float)winH; // 0 at top, 1 at bottom

        // 计算屏幕比例裁剪
        float winAspect = (float)winW / (float)winH;
        float simAspect = (float)ctx.NX / (float)ctx.NZ;
        ImVec2 aspectCorr = { 1.0f, 1.0f };
        if (winAspect > simAspect) {
            aspectCorr.x = winAspect / simAspect;
        }
        else {
            aspectCorr.y = simAspect / winAspect;
        }

        // =========================================================
        // 【核心数学修正公式】
        // 1. viewOffset 必须乘以 0.5f，因为 NDC 空间[-1,1]到纹理空间[0,1]有2倍尺寸差
        // 2. Y 轴的平移项前必须使用减号（-），以抵消 Fragment Shader 中的 1.0-Y 反转
        // =========================================================
        float worldX = (normX - 0.5f) * (aspectCorr.x / viewZoom) + 0.5f + (0.5f * viewOffset.x);
        float worldY = (normY - 0.5f) * (aspectCorr.y / viewZoom) + 0.5f - (0.5f * viewOffset.y);

        // 转换为网格离散索引
        int mx = (int)(worldX * (float)ctx.NX);
        int my = (int)(worldY * (float)ctx.NZ);

        // PML 边界安全判定 [4]
        int min_valid = ctx.npml + 5;
        int max_valid_x = ctx.NX - ctx.npml - 5;
        int max_valid_z = ctx.NZ - ctx.npml - 5;

        if (mx >= min_valid && mx <= max_valid_x && my >= min_valid && my <= max_valid_z) {
            ctx.src_z_idx = my;
            ctx.src_idx = my * ctx.NX + mx;

            // =========================================================
            // 【新增】：鼠标点击时，直接反向同步刷新 UI 滑动条的值！
            // =========================================================
            edit_src_x = mx;
            edit_src_z = my;

            // 重置并立即重新激发波动
            current_it = 0;
            accumulated_compute_time = 0.0f; // <-- 新增归零
            state.running = true;

            freeGPUSimulation(gpu_data);
            initGPUSimulation(gpu_data, ctx);

            std::fill(ctx.record_vx.begin(), ctx.record_vx.end(), 0.0f);
            std::fill(ctx.record_vz.begin(), ctx.record_vz.end(), 0.0f);
        }
    }

    // =============================================================================
    // 【 新增：高精度鼠标悬停位置逆向映射与物性提取算法 】
    // =============================================================================
    ImVec2 mousePos = ImGui::GetMousePos();
    // 判定鼠标是否在有效的主缓冲区视口内（避开顶部和底部状态栏）
    bool is_mouse_inside = (mousePos.x >= 0.0f && mousePos.x < (float)winW &&
        mousePos.y >= barHeight && mousePos.y < (float)(winH - barHeight));

    // 如果鼠标正在操作 ImGui 面板，不捕获物理场位置
    if (ImGui::GetIO().WantCaptureMouse) {
        is_mouse_inside = false;
    }

    int mx = 0, my = 0;
    float h_Vp = 0.0f, h_Vs = 0.0f, h_Rho = 0.0f;
    bool is_valid_grid = false;

    if (is_mouse_inside) {
        // 计算屏幕比例裁剪
        float winAspect = (float)winW / (float)winH;
        float simAspect = (float)ctx.NX / (float)ctx.NZ;
        ImVec2 aspectCorr = { 1.0f, 1.0f };
        if (winAspect > simAspect) {
            aspectCorr.x = winAspect / simAspect;
        }
        else {
            aspectCorr.y = simAspect / winAspect;
        }

        // 归一化鼠标
        float normX = mousePos.x / (float)winW;
        float normY = mousePos.y / (float)winH; // 0 at top, 1 at bottom

        // 高精度逆向矩阵投影
        float worldX = (normX - 0.5f) * (aspectCorr.x / viewZoom) + 0.5f + (0.5f * viewOffset.x);
        float worldY = (normY - 0.5f) * (aspectCorr.y / viewZoom) + 0.5f - (0.5f * viewOffset.y);

        mx = (int)(worldX * (float)ctx.NX);
        my = (int)(worldY * (float)ctx.NZ);

        // 判定计算出的网格坐标是否越界 [4]
        if (mx >= 0 && mx < ctx.NX && my >= 0 && my < ctx.NZ) {
            int k = my * ctx.NX + mx;
            h_Rho = ctx.rho[k];
            // 从 CPU 弹性矩阵中提取当前网格物性，实时反推 Vp 和 Vs [3]
            if (h_Rho > 0.0f) {
                h_Vp = sqrtf(ctx.lambda2mu[k] / h_Rho);
                h_Vs = sqrtf(ctx.mu[k] / h_Rho);
                is_valid_grid = true;
            }
        }
    }

    // =============================================================================
    // 【 键盘与按钮信号集中处理执行区 】
    // =============================================================================

    // A. 响应 C 键（或按钮）重置模拟
    if (g_resetSimRequested) {
        current_it = 0;
        state.running = false;
        accumulated_compute_time = 0.0f; // 计时器归零

        freeGPUSimulation(gpu_data);
        initGPUSimulation(gpu_data, ctx);

        std::fill(ctx.record_vx.begin(), ctx.record_vx.end(), 0.0f);
        std::fill(ctx.record_vz.begin(), ctx.record_vz.end(), 0.0f);

        g_resetSimRequested = false; // 消费信号，自动复位
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
    }

    // --- [4. 执行 CUDA 有限差分时间演化] ---
    if (state.running && current_it < ctx.nt) {

        // 累加本次运行的真实计算耗时 [1.2.7]
        accumulated_compute_time += io.DeltaTime;
        for (int s = 0; s < steps_per_frame; ++s) {
            if (current_it < ctx.nt) {
                runGPUStep(gpu_data, current_it, ctx);
                current_it++;
            }
        }
    }

    // --- [5. 零拷贝互操作：CUDA 写入纹理] ---
    cudaGraphicsMapResources(1, &gl.cudaSeisRes, 0);
    cudaGraphicsSubResourceGetMappedArray(&gl.cudaSeisArray, gl.cudaSeisRes, 0, 0);

    // 在显存中分配一个临时RGBA缓冲区（仅在大小变化时重分配）
    static uchar4* d_rgba_out = nullptr;
    static int last_size = 0;
    if (last_size != ctx.total_grid) {
        if (d_rgba_out) cudaFree(d_rgba_out);
        cudaMalloc(&d_rgba_out, ctx.total_grid * sizeof(uchar4));
        last_size = ctx.total_grid;
    }
    // 调用我们在 Cuda_Check.cu 中编写的高并发色彩映射核函数
    generateWavefieldTextureCUDA(gpu_data, d_rgba_out, color_scale, show_component);

    // 直接将显存中的 RGBA 结果拷贝到 OpenGL 纹理中（完全不经过 CPU 主机）
    cudaMemcpy2DToArray(
        gl.cudaSeisArray, 0, 0,
        d_rgba_out,
        ctx.NX * sizeof(uchar4),
        ctx.NX * sizeof(uchar4),
        ctx.NZ,
        cudaMemcpyDeviceToDevice
    );
    cudaGraphicsUnmapResources(1, &gl.cudaSeisRes, 0);

    // --- [6. 绘制全屏波场纹理背景] ---
    // --- [ 渲染背景与全屏网格 ] ---
    glViewport(0, 0, winW, winH);
    glUseProgram(gl.seisProg);

    // 传递地震专用着色器所需的各种 Uniform 参数
    // 传入波场可视化风格代码
    glUniform1i(glGetUniformLocation(gl.seisProg, "waveStyle"), waveStyle); // <-- 传递新增参数 [1.2.7]
    glUniform2f(glGetUniformLocation(gl.seisProg, "winSize"), (float)winW, (float)winH);
    glUniform2f(glGetUniformLocation(gl.seisProg, "simSize"), (float)ctx.NX, (float)ctx.NZ);
    glUniform2f(glGetUniformLocation(gl.seisProg, "viewOffset"), viewOffset.x, viewOffset.y);
    glUniform1f(glGetUniformLocation(gl.seisProg, "viewZoom"), viewZoom);

    // 【传递新增参数】
    glUniform1f(glGetUniformLocation(gl.seisProg, "totalTime"), (float)glfwGetTime()); // 用于扫描线滚动
    glUniform1f(glGetUniformLocation(gl.seisProg, "npml"), (float)ctx.npml);           // 用于动态绘制 PML 边界

    // 绑定纹理
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, gl.seisTex);
    glUniform1i(glGetUniformLocation(gl.seisProg, "seisTexture"), 0);

    glBindVertexArray(gl.quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    // =============================================================================
     // 【 2. 标尺与光标渲染：直接覆盖在主缓冲区之上 】
     // =============================================================================
     // 构建全屏视口对应的 ViewportInfo 
    ViewportInfo vp;
    vp.x = 0.0f;
    vp.y = 0.0f;
    vp.w = (float)winW;
    vp.h = (float)winH;
    vp.scaleX = vp.w / (float)ctx.NX;
    vp.scaleY = vp.h / (float)ctx.NZ;

    // A. 绘制主缓冲区标尺 (自动在 FDM 背景之上、ImGui 浮动窗口之下) [1.2.7]
    RenderGridRulerOnBackbuffer(ctx, state, winW, winH, barHeight, viewOffset, viewZoom);

    // B. 绘制高亮科技鼠标光标 (使用 ForegroundDrawList 保证永远在最上层)
    RenderBrushCursor(state, vp);

    // 【新增】：绘制当前编辑位置的物理震源准星，传入刚刚定义在顶部的滑动条变量
    RenderSourceMarker(ctx, edit_src_x, edit_src_z, winW, winH, barHeight, viewOffset, viewZoom);

    // 全局 UI 荧光主题色定义 (设置为 static，支持实时修改)
    static ImVec4 uiAccent = ImVec4(0.5f, 1.0f, 0.5f, 1.0f); // 默认青色荧光
    // --- [7. 顶部专业状态栏 (TopBar)] ---
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

            // A. 左侧：GPU 信息
            ImGui::SetCursorPos({ 20 * scale, centerY });
            ImGui::TextDisabled("ACTIVE GPU:"); ImGui::SameLine();
            ImGui::TextColored(uiAccent, "%s", info.name);

            // B. 【升级】：中间模拟参数指示 (新增：物理波场时间 + 实际计算耗时)
            float physicalTime = current_it * ctx.dt; // 计算当前的物理传播时间 (秒) [6]

            // 对实际计算耗时进行精美格式化
            char computeTimeStr[64];
            if (accumulated_compute_time < 60.0f) {
                sprintf(computeTimeStr, "%.2f s", accumulated_compute_time);
            }
            else {
                int minutes = (int)(accumulated_compute_time) / 60;
                float seconds = fmodf(accumulated_compute_time, 60.0f);
                sprintf(computeTimeStr, "%02dm %.2fs", minutes, seconds);
            }

            char statusStr[512];
            sprintf(statusStr, "FORMULATION: TYPE %d  |  STEP: %d / %d  |  PHYS TIME: %.4f s  |  COMPUTE TIME: %s",
                ctx.flag_type, current_it, ctx.nt, physicalTime, computeTimeStr);

            float textWidth = ImGui::CalcTextSize(statusStr).x;
            ImGui::SetCursorPos({ (winW - textWidth) * 0.5f, centerY });
            ImGui::TextColored(uiAccent, "%s", statusStr);

            // C. 右侧：帧率与 HUD 触发
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

        // 状态栏底部荧光线
        ImDrawList* topDraw = ImGui::GetForegroundDrawList();
        topDraw->AddLine(
            { 0, barHeight },
            { (float)winW, barHeight },
            IM_COL32(uiAccent.x * 255, uiAccent.y * 255, uiAccent.z * 255, 60),
            1.0f
        );
    };

    // --- [8. 悬浮玻璃控制面板] ---
    if (showHUD) {

        ImGui::SetNextWindowPos({ 30 * scale, 80 * scale }, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize({ 420 * scale, 780 * scale }, ImGuiCond_FirstUseEver);

        // 动态绑定主题色到窗口边框
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.95f, 1.0f, 0.98f, 0.3f)); // 半透明绿色
        ImGui::PushStyleColor(ImGuiCol_Border, uiAccent);                             // 荧光主题色边框
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.5f);

        // 统一注入主题色到滑块和选项卡
        ImGui::PushStyleColor(ImGuiCol_SliderGrab, uiAccent);
        ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, ImVec4(uiAccent.x * 1.2f, uiAccent.y * 1.2f, uiAccent.z * 1.2f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_TabActive, uiAccent);
        ImGui::PushStyleColor(ImGuiCol_TabHovered, ImVec4(uiAccent.x * 0.8f, uiAccent.y * 0.8f, uiAccent.z * 0.8f, 0.8f));
        ImGui::PushStyleColor(ImGuiCol_HeaderActive, uiAccent);
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(uiAccent.x * 0.8f, uiAccent.y * 0.8f, uiAccent.z * 0.8f, 0.8f));

        if (ImGui::Begin("Control Panel", &showHUD)) {

            // A. 显示平滑滤波后的 FPS
            static float displayFPS = 60.0f;
            displayFPS = displayFPS * 0.95f + io.Framerate * 0.05f;
            ImGui::TextColored(uiAccent, "FPS: %.1f", displayFPS);
            ImGui::Separator();

            // =============================================================================
            // 【核心重构：双 Tab 选项卡布局】
            // =============================================================================
            if (ImGui::BeginTabBar("ControlTabs")) {

                // -------------------------------------------------------------------------
                // 选项卡一：Simulation (模拟算法与计算参数)
                // -------------------------------------------------------------------------
                if (ImGui::BeginTabItem("Simulation")) {
                    ImGui::Spacing();

                    // 1. 模拟公式与算法类型选择 (Formulation) [8]
                    ImGui::TextColored(uiAccent, "Simulation Protocol");
                    const char* sim_types[] = { "Type 1: Whole-domain PML", "Type 2: Interior FDM + Boundary PML", "Type 3: Free Surface + PML" };
                    int temp_type = ctx.flag_type - 1;
                    if (ImGui::Combo("Formulation", &temp_type, sim_types, IM_ARRAYSIZE(sim_types))) {
                        ctx.flag_type = temp_type + 1;
                        // 切换公式时，自动重置模拟并重装 GPU 显存
                        current_it = 0;
                        accumulated_compute_time = 0.0f; // <-- 新增归零
                        state.running = false;
                        freeGPUSimulation(gpu_data);
                        initGPUSimulation(gpu_data, ctx);
                    }
                    ImGui::Spacing();

                    // 2. 图像渲染与分量显示控制
                    ImGui::TextColored(uiAccent, "Visualization Settings");
                    ImGui::SliderInt("Steps / Frame", &steps_per_frame, 1, 10);

                    // 3. 时间步长 (dt) 与最大迭代次数 (nt) 动态控制 [6]
                    ImGui::Separator();
                    ImGui::TextColored(uiAccent, "Time Step & Iteration Control");

                    // 动态计算当前模型的最大 Vp 用于 CFL 稳定性检测
                    float max_vp = 0.0f;
                    for (int k = 0; k < ctx.total_grid; ++k) {
                        float vp = sqrtf(ctx.lambda2mu[k] / ctx.rho[k]);
                        if (vp > max_vp) max_vp = vp;
                    }
                    float dx_val = par.model.dx;
                    float dt_limit = 0.5f * dx_val / max_vp;

                    static float temp_dt = ctx.dt;
                    static float temp_dx = ctx.h; // 对应物理网格步长 dx
                    static int temp_nt = ctx.nt;

                    ImGui::SliderFloat("Time Step (dt)", &temp_dt, 0.00001f, 0.003f, "%.6f s");

                    // 【回归】：网格物理步长 dx 调节滑块，滑块拖动时将直接联动屏幕上标尺的刻度米数！
                    ImGui::SliderFloat("Grid Spacing (dx)", &temp_dx, 0.1f, 20.0f, "%.1f m");

                    ImGui::SliderInt("Max Steps (nt)", &temp_nt, 500, 20000);

                    // 稳定性 OSD 显示
                    bool is_stable = (temp_dt <= dt_limit);
                    ImGui::Text("Max Allowed dt (CFL Limit): %.6f s", dt_limit);
                    if (is_stable) {
                        ImGui::TextColored({ 0.2f, 1.0f, 0.4f, 1.0f }, "[ STATUS: CFL STABLE ]");
                    }
                    else {
                        ImGui::TextColored({ 1.0f, 0.2f, 0.2f, 1.0f }, "[ STATUS: DIVERGENCE RISK! ]");
                    }

                    ImGui::Spacing();
                    // 3. 应用按钮：点击后将最新的 dx 和 dt 写入物理计算核心
                    if (ImGui::Button("APPLY TIME & SPACING CONTROLS", { -1, 30 * scale })) {
                        ctx.dt = temp_dt;
                        ctx.nt = temp_nt;

                        // -------------------------------------------------------------
                        // 【物理重算核心】：更新空间步长 h，并重新计算高阶有限差分系数
                        // 确保空间步长改变后，弹性波动方程的数值求导结果完全缩放匹配
                        // -------------------------------------------------------------
                        ctx.h = temp_dx;
                        par.model.dx = temp_dx;
                        par.model.dz = temp_dx; // 保持二维网格均匀

                        ctx.c1_h = 1.125022f / temp_dx;
                        ctx.c2_h = -0.04687594f / temp_dx;
                        ctx.c3_h = 0.00416669f / temp_dx;
                        ctx.c4_h = -0.00019234f / temp_dx;

                        // 重新预计算自由表面 dp_flat 参数 (防止切换到 Type 3 时因空间步长改变导致溢出)
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

                        // 重新分配相关的接收器历史数据大小 (因为总步数可能变了)
                        ctx.record_vx.assign(ctx.num_rcv * ctx.nt, 0.0f);
                        ctx.record_vz.assign(ctx.num_rcv * ctx.nt, 0.0f);

                        // 重新生成雷克子波
                        generateRickerWavelet(ctx.wavelet, ctx.nt, ctx.dt, par.FDM.f0, par.FDM.t0);

                        // 重置当前步，重装 GPU 显存
                        current_it = 0;
                        state.running = false;
                        freeGPUSimulation(gpu_data);
                        initGPUSimulation(gpu_data, ctx);
                    }

                    ImGui::Spacing();
                    ImGui::Separator();
                    ImGui::Checkbox("V-Sync Control", &state.vsyncEnabled);
                    if (ImGui::IsItemDeactivatedAfterEdit()) {
                        glfwSwapInterval(state.vsyncEnabled ? 1 : 0);
                    }

                    ImGui::EndTabItem();
                }

                // -------------------------------------------------------------------------
                // 选项卡二：Physics & Source (物性与震源参数控制)
                // -------------------------------------------------------------------------
                if (ImGui::BeginTabItem("Physics & Source")) {
                    ImGui::Spacing();

                    // 1. 震源激发参数设置
                    ImGui::TextColored(uiAccent, "Source Position & Angle");
                    /*static int edit_src_x = ctx.NX / 2;
                    static int edit_src_z = ctx.NZ / 2;
                    static float edit_angle = par.FDM.angle;*/

                    int min_valid_grid = ctx.npml + 5;
                    int max_valid_x = ctx.NX - ctx.npml - 5;
                    int max_valid_z = ctx.NZ - ctx.npml - 5;

                    ImGui::SliderInt("Source X (Grid)", &edit_src_x, min_valid_grid, max_valid_x);
                    ImGui::SliderInt("Source Z (Grid)", &edit_src_z, min_valid_grid, max_valid_z);


                    ImGui::SliderFloat("Force Angle", &edit_angle, -180.0f, 180.0f, "%.1f deg");

                    // 2. 【核心新增】：雷克子波主频与延迟控制 [6]
                    ImGui::Separator();
                    ImGui::TextColored(uiAccent, "Ricker Wavelet Properties");

                    ImGui::SliderFloat("Peak Freq (f0)", &edit_f0, 5.0f, 300.0f, "%.1f Hz");
                    ImGui::SliderFloat("Time Delay (t0)", &edit_t0, 0.001f, 0.2f, "%.3f s");

                    // =================================================================
                    // 【高级黑科技】：实时动态雷克子波波形预览 (Wavelet Preview) [1.2.7]
                    // 随着滑动条拖动，波形在控制面板内实时进行拉伸/挤压变化
                    // =================================================================
                    ImGui::TextDisabled("Waveform Real-Time Preview:");
                    float preview_dt = 0.0005f; // 固定高采样率保证预览曲线光滑
                    std::vector<float> preview_x(200);
                    std::vector<float> preview_y(200);
                    for (int i = 0; i < 200; ++i) {
                        float t = i * preview_dt - edit_t0;
                        float pi2_f0 = M_PI * M_PI * edit_f0 * edit_f0;
                        preview_x[i] = i * preview_dt;
                        preview_y[i] = (1.0f - 2.0f * pi2_f0 * t * t) * expf(-pi2_f0 * t * t);
                    }

                    // 1. 【局部修改配色】：将本图表强制设为黑底、荧光绿线 [1.2.7]
                    ImPlot::PushStyleColor(ImPlotCol_FrameBg, IM_COL32(125, 125, 125, 255)); // 边框背景 (深钛金灰)
                    ImPlot::PushStyleColor(ImPlotCol_PlotBg, IM_COL32(125, 125, 125, 255));     // 绘图背景 (纯黑)
                    ImPlot::PushStyleColor(ImPlotCol_Line, IM_COL32(0, 255, 120, 255));   // 曲线颜色 (荧光绿)
                    ImPlot::PushStyleColor(ImPlotCol_AxisGrid, IM_COL32(35, 35, 35, 120)); // 暗灰色微弱网格线

                    // 2. 【核心修改】：在第三参数传入 ImPlotFlags_NoLegend 选项，直接优雅地隐藏 "Ricker" 标签 [1.2.7]
                    if (ImPlot::BeginPlot("##WaveletPreview", ImVec2(-1, 95 * scale), ImPlotFlags_NoLegend)) {
                        ImPlot::SetupAxes("Time", "Amp", ImPlotAxisFlags_NoTickLabels, ImPlotAxisFlags_NoTickLabels);
                        ImPlot::SetupAxesLimits(0.0f, 200 * preview_dt, -1.1f, 1.1f);

                        // 【备用方案】：如果您不想取消，而是想把图例放在右侧 (East)，
                        // 那么请删掉上面 BeginPlot 里的 ImPlotFlags_NoLegend，并取消下面这一行的注释：
                        // ImPlot::SetupLegend(ImPlotLocation_East, ImPlotLegendFlags_None);

                        ImPlot::PlotLine("Ricker", preview_x.data(), preview_y.data(), 200);
                        ImPlot::EndPlot();
                    }

                    // 3. 必须弹出 4 个颜色样式，防止污染底部的地震道 (Seismogram) 大图表 [1.2.7]
                    ImPlot::PopStyleColor(4);

                    // 提交应用子波与震源属性
                    ImGui::Spacing();
                    if (ImGui::Button("APPLY SOURCE CONFIG", { -1, 30 * scale })) {
                        ctx.src_z_idx = edit_src_z;
                        ctx.src_idx = edit_src_z * ctx.NX + edit_src_x;
                        ctx.src_angle = edit_angle;

                        par.FDM.f0 = edit_f0; // 写入静态持久化 par 中
                        par.FDM.t0 = edit_t0;
                        par.FDM.angle = edit_angle;

                        // 重新计算子波并重构模拟
                        generateRickerWavelet(ctx.wavelet, ctx.nt, ctx.dt, edit_f0, edit_t0);
                        current_it = 0;
                        state.running = false;
                        accumulated_compute_time = 0.0f; // 计时器重置
                        freeGPUSimulation(gpu_data);
                        initGPUSimulation(gpu_data, ctx);
                    }
                    ImGui::Spacing();
                    ImGui::Spacing();

                    // 2. 介质物性设置 (岩石物理耦合) [3]
                    ImGui::Separator();
                    ImGui::TextColored(uiAccent, "Rock Physics Coupling");

                    static float edit_Vp = 2000.0f;
                    static float edit_Vs = 1400.0f;
                    static float edit_Density = 2000.0f;

                    ImGui::SliderFloat("P-Wave Velocity (Vp)", &edit_Vp, 500.0f, 6000.0f, "%.1f m/s");
                    ImGui::SliderFloat("S-Wave Velocity (Vs)", &edit_Vs, 200.0f, 3500.0f, "%.1f m/s");
                    ImGui::SliderFloat("Density (Rho)", &edit_Density, 1000.0f, 3000.0f, "%.1f kg/m^3");

                    // 弹性介质物理防呆约束
                    if (edit_Vp < edit_Vs * 1.5f) {
                        edit_Vp = edit_Vs * 1.5f;
                    }

                    ImGui::Spacing();
                    if (ImGui::Button("APPLY PHYSICAL MEDIUM", { -1, 30 * scale })) {
                        // 自动转换为拉梅常数 [3]
                        float mu_val = edit_Density * edit_Vs * edit_Vs;
                        float lambda2mu_val = edit_Density * edit_Vp * edit_Vp;
                        float lambda_val = lambda2mu_val - 2.0f * mu_val;

                        ctx.rho.assign(ctx.total_grid, edit_Density);
                        ctx.mu.assign(ctx.total_grid, mu_val);
                        ctx.lambda.assign(ctx.total_grid, lambda_val);
                        ctx.lambda2mu.assign(ctx.total_grid, lambda2mu_val);

                        // 重新初始化并刷新 GPU 显存
                        current_it = 0;
                        accumulated_compute_time = 0.0f; // <-- 新增归零
                        state.running = false;
                        freeGPUSimulation(gpu_data);
                        initGPUSimulation(gpu_data, ctx);
                    }

                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem("vis")) {
                    ImGui::Spacing();
                    ImGui::SliderFloat("Color Gain", &color_scale, 1e0f, 1e3f, "%.1f");
                    const char* components[] = { "Vz (Vertical)", "Vx (Horizontal)" };
                    ImGui::Combo("Show Component", &show_component, components, IM_ARRAYSIZE(components));
                   
                    ImGui::Spacing();
                    const char* style_types[] = { "Magma Glow ", "3D Specular Coolwarm", "Neon Bipolar" };
                    ImGui::Combo("Visual Style", &waveStyle, style_types, IM_ARRAYSIZE(style_types));
                    ImGui::Spacing();

                    ImGui::EndTabItem();
                }

                ImGui::EndTabBar();
            }

            // =============================================================================
            // 【 7. 运行控制（底部固定区） 】
            // =============================================================================
            ImGui::SetCursorPosY(ImGui::GetWindowHeight() - 210 * scale);
            ImGui::Separator();
            ImGui::TextColored(uiAccent, "EXECUTION CONTROLS");

            // A. 运行暂停键 (带绿/黄动态变色)
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

            // B. 重置物理场模拟
            if (ImGui::Button("RESET SIMULATION", { -1, 28 * scale })) {
                g_resetSimRequested = true; // 发射重置模拟信号
            }

            // C. 【视图重置 (RESET VIEWPORT) 按钮回归】：一键吸附并校准回左上角
            if (ImGui::Button("RESET VIEWPORT", { -1, 28 * scale })) {
                g_resetViewportRequested = true; // 发射重置视口信号
            }

            // D. 全局 UI 荧光主题色选择器
            ImGui::SetCursorPosY(ImGui::GetWindowHeight() - 55 * scale);

            float avail_w = ImGui::GetContentRegionAvail().x;   // 获取当前窗口的总可用宽度
            float btn_w = 60.0f * scale;                        // 设定 Reset 按钮宽度
            float picker_w = avail_w - btn_w - ImGui::GetStyle().ItemSpacing.x; // 计算选择器宽度

            // A. 左侧：颜色选择器 (限制宽度为 picker_w)
            ImGui::PushItemWidth(picker_w);
            ImGui::ColorEdit3("##AccentColorPicker", (float*)&uiAccent, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueBar);
            ImGui::PopItemWidth();
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Interface Hue Accent Color");
            }

            // 同行并排放置
            ImGui::SameLine();

            // B. 右侧：重置按钮 (自动继承当前的 uiAccent 荧光变色，保持视觉高度统一)
            if (ImGui::Button("Reset##Theme", ImVec2(btn_w, 0))) {
                uiAccent = ImVec4(0.5f, 1.0f, 0.5f, 1.0f); // 默认绿色荧光;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Reset UI Accent Color to Default Cyan");
            }

            

        }
        ImGui::End();

        // 统一弹出样式，避免状态污染
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(8);
    }

    // =============================================================================
    // 【 9. 新增：底部专业状态监控栏 (BottomBar) 】
    // =============================================================================
    {
        ImGui::SetNextWindowPos({ 0, (float)winH - barHeight });
        ImGui::SetNextWindowSize({ (float)winW, barHeight });

        // 计算动态微暗底色 (背景有淡淡的主题色微光) [1.1.3]
        ImVec4 botBarBg = ImVec4(0.95f, 1.0f, 0.98f, 0.2f);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, botBarBg);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10 * scale, 0));

        ImGui::Begin("BottomBar_Seis", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar);
        {
            float centerY = 14.0f * scale;

            // A. 左侧：当前鼠标下的网格与物理位置
            ImGui::SetCursorPos({ 20 * scale, centerY });
            if (is_valid_grid) {
                ImGui::TextDisabled("GRID:"); ImGui::SameLine();
                ImGui::TextColored(uiAccent, "X:%d, Z:%d", mx, my);
                ImGui::SameLine();
                ImGui::TextDisabled("| COORD:"); ImGui::SameLine();
                ImGui::TextColored(uiAccent, "X:%.1f m, Z:%.1f m", mx * ctx.h, my * ctx.h);
            }
            else {
                ImGui::TextDisabled("GRID MONITOR: STANDBY (HOVER OVER WAVEFIELD)");
            }

            // B. 中间：当前网格点的物性数值实时监控 (Vp, Vs, Density) [3]
            char paramStr[256];
            if (is_valid_grid) {
                sprintf(paramStr, "P-WAVE SPEED (Vp): %.1f m/s  |  S-WAVE SPEED (Vs): %.1f m/s  |  DENSITY (Rho): %.1f kg/m^3", h_Vp, h_Vs, h_Rho);
            }
            else {
                sprintf(paramStr, "ROCK MEDIUM MONITOR: ACTIVE");
            }
            float textWidth = ImGui::CalcTextSize(paramStr).x;
            ImGui::SetCursorPos({ (winW - textWidth) * 0.5f, centerY });

            if (is_valid_grid) {
                ImGui::TextColored(uiAccent, "%s", paramStr);
            }
            else {
                ImGui::TextDisabled("%s", paramStr);
            }

            // C. 右侧：系统状态指示
            float rightSectionWidth = 220.0f * scale;
            ImGui::SetCursorPos({ (float)winW - rightSectionWidth, centerY });
            ImGui::TextDisabled("STATUS:"); ImGui::SameLine();
            if (is_valid_grid) {
                ImGui::TextColored({ 0.2f, 1.0f, 0.4f, 1.0f }, "INDEX LINKED");
            }
            else {
                ImGui::TextColored({ 1.0f, 0.5f, 0.0f, 1.0f }, "CURSOR OUT");
            }
        }
        ImGui::End();
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(1);

        // --- 底部栏顶部的荧光细线 ---
        ImDrawList* botDraw = ImGui::GetForegroundDrawList();
        botDraw->AddLine(
            { 0, (float)winH - barHeight },
            { (float)winW, (float)winH - barHeight },
            IM_COL32(uiAccent.x * 255, uiAccent.y * 255, uiAccent.z * 255, 60),
            1.0f
        );
    }
}