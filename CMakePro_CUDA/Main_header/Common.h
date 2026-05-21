#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <atomic>
#include <mutex>
#include "imgui.h"

// ===================================================================
// 1. 导航与界面枚举
// ===================================================================
enum class AppScreen {
    Intro0 = 0,     // 启动欢迎页 (Logo/版权)
    Intro1,         // 引导页1
    Intro2,         // 引导页2
    LifeGame,     // 核心分型渲染/模拟界面
    HamStation,
    CudaDiagno,
    Count
};

// ===================================================================
// 2. 模拟与应用程序状态
// ===================================================================
struct SimState {
    AppScreen currentScreen = AppScreen::Intro0; // 当前激活屏幕
    bool isIntroMode = true;        // 是否处于引导模式
    bool running = false;           // 模拟运行状态
    bool vsyncEnabled = true;       // 垂直同步开关

    // --- UI 显示控制 ---
    bool showControls = true;       // 控制面板显示
    bool showTelemetry = true;      // 遥测面板显示
    bool showBgGrid = true;         // 背景装饰网格显示

    bool autoZoom = true;

    // --- 分型渲染逻辑 ---
    double fractalZoom = 2.0;       // 缩放深度 (使用double保证深入精度的稳定)
    int styleMode = 0; // 0: 有机组织, 1: 绿叶脉络

    // --- 用户空闲检测 ---
    float lastInputTime = 0.0;      // 上次输入时间点
    float idleTimeout = 60.0;       // 空闲超时阈值 (秒)
    double lastMouseX = 0.0;        // 鼠标X轴缓存
    double lastMouseY = 0.0;        // 鼠标Y轴缓存
};

// ===================================================================
// 3. 全局配置参数
// ===================================================================
struct SimConfig {
    int simWidth = 1024;           // 模拟计算宽度
    int simHeight = 512;            // 模拟计算高度
    int winWidth = 1000;           // 初始窗口宽度
    int winHeight = 1000;           // 初始窗口高度
    ImVec4 back_color = ImVec4(0.12f, 0.13f, 0.18f, 1.0f); // 界面清除色
};

// ===================================================================
// 4. OpenGL/GPU 资源句柄与 Uniform 缓存
// ===================================================================
struct GLHandles {
    // --- 模拟计算资源 (FDTD/Compute) ---
    GLuint texVel[2] = { 0, 0 }; // 速度场纹理
    GLuint texStress[2] = { 0, 0 }; // 应力场纹理
    GLuint texMedium = 0;        // 介质属性纹理
    GLuint computeProg = 0;        // 计算着色器程序
    GLuint sourceSSBO = 0;        // 波源数据缓存
    GLuint statusSSBO = 0;        // 状态读取缓存

    // --- 渲染资源 (Fractal/Rendering) ---
    GLuint renderProg = 0;        // 核心渲染程序 (分型)
    GLuint fractalProg = 0;        // 备用分型计算程序
    GLuint fractalTex = 0;        // 分型结果缓存纹理
    GLuint quadVAO = 0, quadVBO = 0; // 全屏渲染矩形顶点

    // --- Uniform 位置缓存 (优化性能，避免每帧查找字符串) ---
    GLint loc_time = -1;           // u_time
    GLint loc_res = -1;           // u_resolution
    GLint loc_center = -1;           // u_center
    GLint loc_zoom = -1;           // u_zoom
    GLint loc_iters = -1;           // u_iters
    GLint loc_smooth = -1;           // u_enableSmooth
    GLint loc_scan = -1;           // u_enableScanlines
    GLint loc_style = 0;
};

// ===================================================================
// 5. 异步加载与模型数据结构
// ===================================================================
struct ModelLoadResult {
    bool success = false;
    int width = 0, height = 0;
    std::vector<float> data;
};

struct AsyncLoader {
    std::thread worker;                   // 后台加载线程
    std::atomic<bool> isLoading{ false }; // 是否加载中
    std::atomic<bool> isDataReady{ false };// 数据是否准备就绪
    std::atomic<float> progress{ 0.0f };  // 加载进度 (0.0-1.0)
    std::string statusMsg = "IDLE";       // 状态描述信息
    ModelLoadResult tempResult;           // 临时结果缓存
    std::mutex dataMutex;                 // 线程保护锁
};

// ===================================================================
// 6. 实体与基础结构
// ===================================================================
struct Particle {
    ImVec2 pos;  // 位置
    ImVec2 vel;  // 速度
    float type;  // 粒子类型 (0=纵波, 1=横波/渲染着色)
};

// ===================================================================
// 7. Shader 实用工具函数 (实现)
// ===================================================================

/**
 * @brief 从文件读取着色器源码
 */
inline std::string loadShaderFromFile(const std::string& filePath) {
    std::ifstream shaderFile;
    shaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try {
        shaderFile.open(filePath);
        std::stringstream shaderStream;
        shaderStream << shaderFile.rdbuf();
        return shaderStream.str();
    }
    catch (const std::exception& e) {
        std::cerr << "CORE::SHADER::FILE_NOT_FOUND: " << filePath << " | " << e.what() << std::endl;
        return "";
    }
}

/**
 * @brief 编译并创建单个着色器对象
 */
inline GLuint createShader(const std::string& source, GLenum type) {
    if (source.empty()) return 0;
    const char* src = source.c_str();
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "CORE::SHADER::COMPILATION_ERROR ("
            << (type == GL_VERTEX_SHADER ? "VERT" : (type == GL_FRAGMENT_SHADER ? "FRAG" : "COMP"))
            << ")\n" << infoLog << std::endl;
        return 0;
    }
    return shader;
}