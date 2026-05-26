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
// GLFW 错误捕获回调
// =================================================================================
static void glfw_error_callback(int error, const char* description) {
    std::cerr << "Glfw Error [" << error << "]: " << description << std::endl;
}

// =================================================================================
// 初始化离屏渲染四边形几何体 VAO/VBO
// =================================================================================
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

// =================================================================================
// 主函数入口
// =================================================================================
int main() {
    // -----------------------------------------------------------------------------
    // Stage 1: CUDA 硬件环境诊断
    // -----------------------------------------------------------------------------
    std::cout << "========= CUDA 硬件环境测试 =========" << std::endl;
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

    // 系统与配置对象初始化
    SimState state;         // 状态管理
    SimConfig config;       // 程序配置
    GLHandles gl;           // OpenGL 资源句柄
    AsyncLoader g_loader;   // 后台加载器

    // -----------------------------------------------------------------------------
    // Stage 2: GLFW 与 OpenGL 窗口环境搭建
    // -----------------------------------------------------------------------------
    glfwSetErrorCallback(glfw_error_callback); // 注册错误回调

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

    // -----------------------------------------------------------------------------
    // Stage 3: 离屏基础几何构建
    // -----------------------------------------------------------------------------
    InitQuad(gl);

    // -----------------------------------------------------------------------------
    // Stage 4: 着色器与纹理管线编译
    // -----------------------------------------------------------------------------
    // 1. 加载渲染着色器 (可视化热力图等)
    gl.renderProg = glCreateProgram();
    std::string vsSrc = loadShaderFromFile("../resource_CUDA_V1/shader/vs_code.vert");
    std::string fsSrc = loadShaderFromFile("../resource_CUDA_V1/shader/fs_code.frag");
    GLuint vs = createShader(vsSrc, GL_VERTEX_SHADER);
    GLuint fs = createShader(fsSrc, GL_FRAGMENT_SHADER);
    glAttachShader(gl.renderProg, vs);
    glAttachShader(gl.renderProg, fs);
    glLinkProgram(gl.renderProg);

    // 链接成功后及时删除临时着色器对象，防止资源泄露
    glDeleteShader(vs);
    glDeleteShader(fs);

    // 生成与注册 OpenGL 纹理（用于存储 0.0 ~ 1.0 的单通道热力值）
    glGenTextures(1, &gl.lifeTex);
    glBindTexture(GL_TEXTURE_2D, gl.lifeTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, 1920, 1080, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // 2. 编译并加载地震专用渲染着色器 
    gl.seisProg = glCreateProgram();
    std::string seisVsSrc = loadShaderFromFile("../resource_CUDA_V1/shader/seis_code.vert");
    std::string seisFsSrc = loadShaderFromFile("../resource_CUDA_V1/shader/seis_code.frag");
    GLuint seisVs = createShader(seisVsSrc, GL_VERTEX_SHADER);
    GLuint seisFs = createShader(seisFsSrc, GL_FRAGMENT_SHADER);
    glAttachShader(gl.seisProg, seisVs);
    glAttachShader(gl.seisProg, seisFs);
    glLinkProgram(gl.seisProg);

    // 及时释放临时着色器资源
    glDeleteShader(seisVs);
    glDeleteShader(seisFs);

    // -----------------------------------------------------------------------------
    // Stage 5: CUDA 互操作与内部缓冲区分配
    // -----------------------------------------------------------------------------
    // 将 OpenGL 纹理注册到 CUDA 互操作资源中
    cudaGraphicsGLRegisterImage(&gl.cudaRes, gl.lifeTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

    // 初始化 CUDA 内部生命模拟算法相关的状态及缓冲区
    InitCudaLife(1920, 1080); // 初始化随机数生成器
    cudaMalloc(&gl.d_current, 1920 * 1080);
    cudaMalloc(&gl.d_next, 1920 * 1080);
    cudaMalloc(&gl.d_heatData, 1920 * 1080 * sizeof(float));

    // 植入初始种子数据
    SeedCudaLife(gl.d_current, 1920, 1080, 0.3f);

    // -----------------------------------------------------------------------------
    // Stage 6: UI 框架（ImGui / ImPlot）环境构建
    // -----------------------------------------------------------------------------
    Init_Imgui(window);
    ImPlot::CreateContext();

    // 初始运行状态设定 (强制开启启动屏逻辑)
    state.isIntroMode = true;
    glfwSwapInterval(state.vsyncEnabled ? 1 : 0);

    // =============================================================================
    // 主渲染循环
    // =============================================================================
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // 1. 清空主后备缓冲区底色
        glClearColor(0.1f, 0.12f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // 2. 动态垂直同步检测
        static bool lastVsync = state.vsyncEnabled;
        if (state.vsyncEnabled != lastVsync) {
            glfwSwapInterval(state.vsyncEnabled ? 1 : 0);
            lastVsync = state.vsyncEnabled;
        }

        // 3. 窗口大小采集与用户闲置状态追踪
        int winW, winH;
        glfwGetWindowSize(window, &winW, &winH);
        CheckIdleStatus(window, state);

        // 4. GUI 准备新帧数据
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // 5. 事件响应与输入控制处理
        ProcessInput(window, config, state, gl);

        // 6. 场景渲染分发 (包含启动页与核心模拟页面)
        RenderApp(state.currentScreen, state, info, winW, winH, gl, g_loader);
        state.isIntroMode = IsIntroScreen(state.currentScreen); // 同步页面标识

        // 7. GUI 指令集提交
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // 8. 多视口支持 (针对分离窗口渲染状态)
        if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            GLFWwindow* backup_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_context);
        }

        glfwSwapBuffers(window);
    }

    // =============================================================================
    // 资源销毁与环境释放
    // =============================================================================
    cleanup(gl); // 调用原有资源释放函数释放句柄与 CUDA 缓冲区

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}