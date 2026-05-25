/**
 * @file    main.cpp
 * @brief   程序主入口：采用“Intro Screen + Simulation”双模式架构
 */

 // =================================================================================
 // 1. C/C++ 标准库 (Standard Libraries)
 // =================================================================================
#include <iostream>
#include <vector>
#include <deque>
#include <list>
#include <cmath>
#include <algorithm>
#include <complex>
#include <random>
#include <ctime>
#include <filesystem>     // C++17 跨平台路径处理

// =================================================================================
// 2. 图形基础库 (Graphics API & Window Management)
// =================================================================================
// 注意：GLEW 必须在 GLFW 之前包含，以确保正确的 OpenGL 函数指针加载
#include <GL/glew.h>
#include <GLFW/glfw3.h>

// =================================================================================
// 3. 用户界面库 (User Interface Libraries)
// =================================================================================
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"       // ImGui 绘图扩展

// =================================================================================
// 4. 图像处理工具 (Image Utilities - stb)
// =================================================================================
// 警告：IMPLEMENTATION 宏只能在一个 .cpp 文件中定义，否则会导致符号重定义错误
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// =================================================================================
// 5. 本项目自定义头文件 (Project Specific Headers)
// =================================================================================
#include "CMakePro_cuda.h" // 包含启动界面逻辑、系统结构体及核心计算单元
 // =================================================================================
 // 主函数
 // =================================================================================

static void glfw_error_callback(int error, const char* description) {
    std::cerr << "Glfw Error " << error << ": " << description << std::endl;
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
    // 【PML 终极超强吸收优化】：
    //  1. 将阻尼常数 d_max 额外乘以 8.0 倍！强制将有效波场消融范围压缩到 PML 宽度的前 1/4 区域。
    //  2. 将衰减剖面指数从 3.0 (立方) 改回 2.0 (平方)。
    //     因为立方曲线在前 1/4 区域的增长极为缓慢 (0.25^3 = 0.015, 几乎为0)，而平方
    //     能让阻尼系数在刚进入 PML 时就快速攀升，实现“刚进入 1/4 区域就吸收干净”的物理效果。
    // =============================================================================
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
        float L = ctx.npml * h; // PML 吸收层物理总厚度

        // =============================================================================
        // 【PML 黄金平衡优化】：
        //  1. 将阻尼暴兵系数从过饱和的 8.0f 下调至黄金平衡点 3.5f，消除 1/4 处的“数值反射墙”
        //  2. 将衰减剖面指数微调为 2.5f 阶 (介于平方与立方之间)。
        //     这能保证横波（短波长）在入口处极度平滑滑入（零反射），并在 PML 中期被快速消融，
        //     彻底根治由于阻尼过陡导致的流固边界/横波长时间累计发散问题！
        // =============================================================================
        float d_max = (3.0f * max_vp * 16.12f * 3.5f) / (2.0f * L);
        float pml_power = 2.5f; // 采用 2.5 阶黄金曲线，兼顾“入口极平滑”与“中期强消融”

        // 1. X 轴 PML 阻尼系数初始化 (采用 2.5f 阶黄金衰减曲线)
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

        // 2. Z 轴 PML 阻尼系数初始化 (采用 2.5f 阶黄金衰减曲线)
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

    // =============================================================================
    // 【核心修复 3】：纠正初始震源位置计算严重错位
    //  确保 1D 展平索引 src_idx 与 Z轴深度索引 src_z_idx 完美指向地层 1/4 的同一深度 [1.1.3]
    // =============================================================================
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
}

void InitQuad(GLHandles& gl) {
    // 定义 4 个顶点：位置(x,y,z) + 纹理坐标(u,v)
    float vertices[] = {
        // 位置              // 纹理坐标
        -1.0f,  1.0f, 0.0f,  0.0f, 1.0f,  // 左上
        -1.0f, -1.0f, 0.0f,  0.0f, 0.0f,  // 左下
         1.0f, -1.0f, 0.0f,  1.0f, 0.0f,  // 右下

        -1.0f,  1.0f, 0.0f,  0.0f, 1.0f,  // 左上
         1.0f, -1.0f, 0.0f,  1.0f, 0.0f,  // 右下
         1.0f,  1.0f, 0.0f,  1.0f, 1.0f   // 右上
    };

    glGenVertexArrays(1, &gl.quadVAO);
    glGenBuffers(1, &gl.quadVBO);

    glBindVertexArray(gl.quadVAO);

    glBindBuffer(GL_ARRAY_BUFFER, gl.quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), &vertices, GL_STATIC_DRAW);

    // 位置属性 (Location 0)
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);

    // 纹理坐标属性 (Location 1)
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));

    glBindVertexArray(0);
}

int main() {

    std::cout << "========= CUDA 硬件环境测试 =========" << std::endl;
    // 1. 测试获取 GPU 信息
    GpuInfo info = GetCudaDeviceInfo();
    if (info.success) {
        std::cout << "[成功] 找到 GPU 设备:" << std::endl;
        std::cout << "   - 设备名称: " << info.name << std::endl;
        std::cout << "   - 计算能力: " << info.computeCapabilityMajor << "." << info.computeCapabilityMinor << std::endl;
        std::cout << "   - 显存总量: " << (info.totalMem / 1024 / 1024) << " MB" << std::endl;
    }
    else {
        std::cerr << "[失败] 未找到 CUDA 设备或驱动程序异常！" << std::endl;
        return -1;
    }

    // 1. 系统与配置对象初始化
    SimState state;         // 状态管理
    SimConfig config;       // 程序配置
    GLHandles gl;           // OpenGL 资源句柄
    AsyncLoader g_loader;   // 后台加载器

    // ============================================================
    // 1.1 GLFW 与 OpenGL 窗口环境搭建
    // ============================================================
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // 设置 OpenGL 4.3 Core Profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(config.winWidth, config.winHeight, "CMakePro_CUDA_SeisSim_V1", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    glfwMaximizeWindow(window);
    glfwGetWindowSize(window, &config.winWidth, &config.winHeight);
    glfwMakeContextCurrent(window);
    glfwSetWindowUserPointer(window, &gl); // 将 GL 句柄绑定到窗口以便后续回调

    // 加载窗口图标
    GLFWimage images[1];
    std::string image_icon = "../resource_CUDA_V1/images/icon_Seis.png";
    images[0].pixels = stbi_load(image_icon.c_str(), &images[0].width, &images[0].height, 0, 4);
    if (images[0].pixels) {
        glfwSetWindowIcon(window, 1, images);
        stbi_image_free(images[0].pixels);
    }

    // 初始化 GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    InitQuad(gl);//
    // 加载渲染着色器 (可视化)
    gl.renderProg = glCreateProgram();
    std::string vsSrc = loadShaderFromFile("../resource_CUDA_V1/shader/vs_code.vert");
    std::string fsSrc = loadShaderFromFile("../resource_CUDA_V1/shader/fs_code.frag");
    GLuint vs = createShader(vsSrc, GL_VERTEX_SHADER);
    GLuint fs = createShader(fsSrc, GL_FRAGMENT_SHADER);
    glAttachShader(gl.renderProg, vs);
    glAttachShader(gl.renderProg, fs);
    glLinkProgram(gl.renderProg);

    // 在 gl.renderProg 链接之后
    glGenTextures(1, &gl.lifeTex);
    glBindTexture(GL_TEXTURE_2D, gl.lifeTex);
    // 我们使用 GL_R32F 存储热力值（0.0 ~ 1.0）
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, 1920, 1080, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    // 注册纹理到 CUDA
    cudaGraphicsGLRegisterImage(&gl.cudaRes, gl.lifeTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

    // 初始化 CUDA 内部缓冲区
    InitCudaLife(1920, 1080); // 初始化随机数生成器
    cudaMalloc(&gl.d_current, 1920 * 1080);
    cudaMalloc(&gl.d_next, 1920 * 1080);
    cudaMalloc(&gl.d_heatData, 1920 * 1080 * sizeof(float));
    // 必须：给初始数据！
    SeedCudaLife(gl.d_current, 1920, 1080, 0.3f);

    // 1. 编译并加载地震专用着色器 
    gl.seisProg = glCreateProgram();
    std::string seisVsSrc = loadShaderFromFile("../resource_CUDA_V1/shader/seis_code.vert");
    std::string seisFsSrc = loadShaderFromFile("../resource_CUDA_V1/shader/seis_code.frag");
    GLuint seisVs = createShader(seisVsSrc, GL_VERTEX_SHADER);
    GLuint seisFs = createShader(seisFsSrc, GL_FRAGMENT_SHADER);
    glAttachShader(gl.seisProg, seisVs);
    glAttachShader(gl.seisProg, seisFs);
    glLinkProgram(gl.seisProg);
    glDeleteShader(seisFs); // 链接后可安全删除临时 shader 句柄


    // 初始化 ImGui 与 ImPlot
    Init_Imgui(window);
    ImPlot::CreateContext();

    // ============================================================
    // 1.2 初始状态设定
    // ============================================================
    state.isIntroMode = true; // 强制从启动页开始
    // 在循环开始前，先根据初始状态设置一次
    glfwSwapInterval(state.vsyncEnabled ? 1 : 0);
    // ============================================================
    // 2. 主循环 (Main Loop)
    // ============================================================
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        // 1. 首先清除整个屏幕（必须在所有渲染之前）
        glClearColor(0.1f, 0.12f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        // VSync 动态切换逻辑 ---
        static bool lastVsync = state.vsyncEnabled;
        if (state.vsyncEnabled != lastVsync) {
            glfwSwapInterval(state.vsyncEnabled ? 1 : 0);
            lastVsync = state.vsyncEnabled;
        }

        // 帧信息获取
        int winW, winH;
        glfwGetWindowSize(window, &winW, &winH);
        CheckIdleStatus(window, state); // 检测用户是否空闲

        // 2. ImGui 准备新帧
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // --- 键盘切换逻辑 ---
        ProcessInput(window, config, state, gl);//处理输入 (Input Processing)
        // 使用新的分发器渲染

        RenderApp(state.currentScreen, state, info, winW, winH, gl, g_loader);
        state.isIntroMode = IsIntroScreen(state.currentScreen); // 同步状态

        // --- 渲染提交 ---
        ImGui::Render();

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // 处理多视口 (如果 ImGui 开启了多窗口支持)
        if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            GLFWwindow* backup_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_context);
        }
        glfwSwapBuffers(window);
    }

    // ============================================================
    // 3. 资源清理 (Cleanup)
    // ============================================================
    cleanup(gl); // 确保实现此函数以释放纹理/Shader资源
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();

    return 0;
}