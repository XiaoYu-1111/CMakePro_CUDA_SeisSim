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
    LifeGame,       // 生命游戏cpu
    LifeGame2,       // 生命游戏gpu
    HamStation,
    CudaDiagno,
    SeisSim_GPU,  // <-- 新增：CUDA 地震波场模拟屏
    Count
};
// 定义画笔类型枚举
enum ToolMode {
    TOOL_NONE = 0,   // 放置单点波源 (Source)
    TOOL_HIGH = 1,  // 画笔：高速体
    TOOL_LOW = 2,  // 画笔：低速体
    TOOL_WALL = 3,  // 画笔：自定义
    TOOL_ERASER = 4,// 画笔：橡皮擦
    TOOL_ARRAY = 5,   // 工具：检波器阵列 (拖拽放置)
    TOOL_RECT = 6,//填充材质
    TOOL_RECT_SEL = 7//只绘制框
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


    // === 新增：画笔相关 ===
    int brushType = TOOL_NONE; // 当前画笔模式
    float brushRadius = 30.0f;  // 画笔半径 (像素)
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
#include <cuda_runtime.h>
#include <cuda_gl_interop.h> // 必须包含这个头文件以支持互操作

struct GLHandles {
    // --- 1. 渲染程序与基础几何 ---
    GLuint renderProg = 0;           // 片段着色器程序 (渲染全屏纹理)
    GLuint quadVAO = 0, quadVBO = 0; // 全屏矩形的顶点数据

    // --- 2. OpenGL 纹理 (生命游戏数据中心) ---
    GLuint lifeTex = 0;              // 映射到显存的纹理句柄 (存储热力值)

    // --- 3. CUDA-OpenGL 互操作句柄 ---
    // 注意：cudaRes 必须是指针类型 (cudaGraphicsResource_t)
    struct cudaGraphicsResource* cudaRes = nullptr;

    // 映射后的 CUDA 数组，用于将 CUDA 计算结果拷贝到 OpenGL 纹理
    cudaArray_t cudaArray = nullptr;

    // --- 4. CUDA 专用显存缓冲区 (Device Pointers) ---
    // 为了效率，逻辑计算使用原始显存指针，计算后再拷贝到上面的纹理
    uint8_t* d_current = nullptr;    // 当前代细胞状态 (0 或 1)
    uint8_t* d_next = nullptr;       // 下一代细胞状态
    float* d_heatData = nullptr;   // 显存中的热力值数组 (0.0f - 1.0f)

    int simW = 1920;
    int simH = 1080;

    // --- 新增：地震模拟资源 ---
    GLuint seisProg;       // <-- 新增：地震模拟专用着色器
    GLuint seisTex;                 // 地震波场纹理
    cudaGraphicsResource_t cudaSeisRes; // 用于地震的互操作资源
    cudaArray_t cudaSeisArray;

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


struct SciParticle {
    ImVec2 pos;
    ImVec2 vel;
    ImU32  color;
    float  phase;
    float  orbitSize;

    static const int ABS_MAX_TRAIL = 2400;
    ImVec2 trail[ABS_MAX_TRAIL];
    int trail_ptr = 0;
    int trail_count = 0;

    void AddTrail(ImVec2 p) {
        trail[trail_ptr] = p;
        trail_ptr = (trail_ptr + 1) % ABS_MAX_TRAIL;
        if (trail_count < ABS_MAX_TRAIL) trail_count++;
    }
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

void InitQuad(GLHandles& gl);

#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

struct Model {
    int xnum = 500;
    int znum = 500;
    float dx = 1.0f;
    float dz = 1.0f;
};

struct Geometry {
    float srcPace = 10.0f;
    float rcvArray_inc = 10.0f;
    float srcBegin = 2500.0f;
    float srcZ = 20.0f;
    float srcNumBegin = 1.0f;
    float srcNumEnd = 1.0f;
    float rcvPace = 10.0f;
    float rcvBegin = 0.0f;
    float rcvEnd = 5000.0f;
    float rcvZ = 0.0f;
};

struct FDM {
    float npml = 50.0f;
    float xPace = 1.0f;
    float zPace = 1.0f;
    float f0 = 100.0f;
    float t0 = 1 / f0;
    float angle = 0.0f;
    float srcFlag = 0.0f;
    float dt = 0.000020f;
    float nt = 10000.0f;
    float snapPaceNum = 10.0f;
    float decflag = 0.0f;
    float correctionFlag = 0.0f;
    float upFlag = 1.0f;
    float dwnFlag = 1.0f;
    float lfFlag = 1.0f;
    float rtFlag = 1.0f;
};

struct Parameters {
    Model model;
    Geometry geom;
    FDM FDM;
};

struct SimulationContext {
    int NX, NZ;
    int total_grid;
    float h, dt;
    int nt, npml;
    int snap_spacenum;
    int flag_type; // 1, 2, 3

    std::vector<float> rho, mu, lambda, lambda2mu;
    std::vector<float> dp_flat;
    std::vector<float> dx, dx_half, dz, dz_half;

    float c1_h, c2_h, c3_h, c4_h;

    int src_idx;
    int src_z_idx;
    std::vector<float> wavelet;
    float src_angle;

    int num_rcv;
    std::vector<int> rcv_grid_idx;

    std::vector<float> record_vx;
    std::vector<float> record_vz;

    bool upFlag;
};

// 视口基本信息结构体（用于适配 ImGui 绘图区域坐标）
struct ViewportInfo {
    float x = 0.0f; // 视口在屏幕上的绝对左边界
    float y = 0.0f; // 视口在屏幕上的绝对上边界
    float w = 1.0f; // 视口的宽度 (像素)
    float h = 1.0f; // 视口的高度 (像素)
    float scaleX = 1.0f; // 屏幕 X 像素 / 模拟 X 格数
    float scaleY = 1.0f; // 屏幕 Y 像素 / 模拟 Z 格数
};

// CUDA 显存端多震源参数结构体
struct GPUSource {
    int idx;       // 震源在 2D 数组中的 1D 展平索引
    float t;       // 当前震源已传播的物理时间 (秒)
    float f_peak;  // 震源的主频 f0
    float amp;     // 震源的振幅
};

// =============================================================================
// 1. 场景预设枚举与全局 inline 持久化变量 (C++20 标准)
// =============================================================================
enum SceneType {
    SCENE_UNIFORM = 0,         // 均匀介质 (Default)
    SCENE_EARTH,               // 地球分层模型 (含液态外核横波消失现象)
    SCENE_DOUBLE_SLIT,         // 双缝干涉
    SCENE_TWO_LAYER,           // 双层介质分界面
    SCENE_WAVEGUIDE,           // 直波导通道
    SCENE_WAVEGUIDE_CURVED,    // 弯曲波导通道
    SCENE_CRYSTAL,             // 六角格声子晶体
    SCENE_RANDOM_SCATTER,      // 随机散射介质 (300个气泡)
    SCENE_CURVED,              // 起伏分界面 (正弦层位)
    SCENE_REFRACTION,          // 线性连续速度梯度 (折射回转波)
    SCENE_PENROSE_ROOM         // 彭罗斯椭圆房间 (波场双焦收敛)
};