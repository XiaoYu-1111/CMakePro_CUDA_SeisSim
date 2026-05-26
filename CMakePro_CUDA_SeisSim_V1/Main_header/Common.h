#pragma once

// =============================================================================
// 1. 系统与第三方依赖库 (External Headers)
// =============================================================================
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h> // CUDA 与 OpenGL 互操作支持
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <atomic>
#include <mutex>
#include <cmath>
#include "imgui.h"

// 防御性定义 M_PI 宏
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// =============================================================================
// 2. 全局枚举定义 (Global Enumerations)
// =============================================================================

// 应用程序激活页面/屏幕索引
enum class AppScreen {
    Intro0 = 0,     // 启动欢迎页 (Logo/版权)
    Intro1,         // 引导页1
    Intro2,         // 引导页2
    LifeGame,       // 生命游戏cpu
    LifeGame2,      // 生命游戏gpu
    HamStation,
    CudaDiagno,
    SeisSim_GPU,    // CUDA 地震波场模拟屏
    Count
};

// 交互鼠标画笔工具模式
enum ToolMode {
    TOOL_NONE = 0,  // 放置单点波源 (Source)
    TOOL_HIGH = 1,  // 画笔：高速体
    TOOL_LOW = 2,  // 画笔：低速体
    TOOL_WALL = 3,  // 画笔：自定义
    TOOL_ERASER = 4,  // 画笔：橡皮擦
    TOOL_ARRAY = 5,  // 工具：检波器阵列 (拖拽放置)
    TOOL_RECT = 6,  // 填充材质
    TOOL_RECT_SEL = 7   // 只绘制框
};

// 地震数值模拟预设场景类别
enum SceneType {
    SCENE_UNIFORM = 0,      // 均匀介质 (Default)
    SCENE_EARTH,            // 地球分层模型 (含液态外核横波消失现象)
    SCENE_DOUBLE_SLIT,      // 双缝干涉
    SCENE_TWO_LAYER,        // 双层介质分界面
    SCENE_WAVEGUIDE,        // 直波导通道
    SCENE_WAVEGUIDE_CURVED, // 弯曲波导通道
    SCENE_CRYSTAL,          // 六角格声子晶体
    SCENE_RANDOM_SCATTER,   // 随机散射介质 (300个气泡)
    SCENE_CURVED,           // 起伏分界面 (正弦层位)
    SCENE_REFRACTION,       // 线性连续速度梯度 (折射回转波)
    SCENE_PENROSE_ROOM      // 彭罗斯椭圆房间 (波场双焦收敛)
};

// =============================================================================
// 3. 基础仿真与应用状态管理 (Core State Structs)
// =============================================================================

// 应用全局交互状态追踪
struct SimState {
    AppScreen currentScreen = AppScreen::Intro0; // 当前激活屏幕
    bool isIntroMode = true;              // 是否处于引导模式
    bool running = false;             // 模拟运行状态
    bool vsyncEnabled = true;              // 垂直同步开关

    // --- UI 显示控制 ---
    bool showControls = true;              // 控制面板显示
    bool showTelemetry = true;              // 遥测面板显示
    bool showBgGrid = true;              // 背景装饰网格显示
    bool autoZoom = true;

    // --- 分形渲染逻辑 ---
    double fractalZoom = 2.0;               // 缩放深度 (保证深入精度的稳定)
    int    styleMode = 0;                 // 0: 有机组织, 1: 绿叶脉络

    // --- 用户空闲检测 ---
    float  lastInputTime = 0.0f;              // 上次输入时间点
    float  idleTimeout = 60.0f;             // 空闲超时阈值 (秒)
    double lastMouseX = 0.0;               // 鼠标X轴缓存
    double lastMouseY = 0.0;               // 鼠标Y轴缓存

    // --- 画笔相关 ---
    int    brushType = TOOL_NONE;         // 当前画笔模式
    float  brushRadius = 30.0f;             // 画笔半径 (像素)
};

// 静态应用环境配置
struct SimConfig {
    int   simWidth = 1024;                     // 模拟计算宽度
    int   simHeight = 512;                      // 模拟计算高度
    int   winWidth = 1000;                     // 初始窗口宽度
    int   winHeight = 1000;                     // 初始窗口高度
    ImVec4 back_color = ImVec4(0.12f, 0.13f, 0.18f, 1.0f); // 界面清除色
};

// OpenGL 及 CUDA 资源绑定句柄
struct GLHandles {
    // --- 1. 渲染程序与基础几何 ---
    GLuint renderProg = 0;           // 片段着色器程序 (渲染全屏纹理)
    GLuint quadVAO = 0;           // 全屏矩形的顶点数组对象
    GLuint quadVBO = 0;           // 全屏矩形的顶点缓冲对象

    // --- 2. OpenGL 纹理 ---
    GLuint lifeTex = 0;           // 存储热力值的 OpenGL 纹理

    // --- 3. CUDA-OpenGL 互操作句柄 ---
    struct cudaGraphicsResource* cudaRes = nullptr;
    cudaArray_t cudaArray = nullptr; // 映射后的 CUDA 数组指针

    // --- 4. CUDA 专用显存缓冲区 (Device Pointers) ---
    uint8_t* d_current = nullptr;   // 当前细胞状态
    uint8_t* d_next = nullptr;   // 下一代细胞状态
    float* d_heatData = nullptr;   // 显存热力值数组 (0.0f - 1.0f)

    int simW = 1920;
    int simH = 1080;

    // --- 5. 地震模拟专用资源 ---
    GLuint                 seisProg; // 地震模拟专用着色器程序
    GLuint                 seisTex;  // 地震波场纹理
    cudaGraphicsResource_t cudaSeisRes;
    cudaArray_t            cudaSeisArray;
};

// =============================================================================
// 4. 数据加载与辅助粒子实体 (Helper & Loader Structs)
// =============================================================================

// 后台异步加载结果
struct ModelLoadResult {
    bool               success = false;
    int                width = 0;
    int                height = 0;
    std::vector<float> data;
};

// 异步加载管理机制
struct AsyncLoader {
    std::thread       worker;                    // 后台加载线程
    std::atomic<bool> isLoading{ false };        // 是否加载中
    std::atomic<bool> isDataReady{ false };      // 数据是否准备就绪
    std::atomic<float> progress{ 0.0f };         // 加载进度 (0.0-1.0)
    std::string       statusMsg = "IDLE";        // 状态描述信息
    ModelLoadResult   tempResult;                // 临时结果缓存
    std::mutex        dataMutex;                 // 线程保护锁
};

// 科学粒子特效追踪
struct SciParticle {
    ImVec2 pos;
    ImVec2 vel;
    ImU32  color;
    float  phase;
    float  orbitSize;

    static const int ABS_MAX_TRAIL = 2400;
    ImVec2 trail[ABS_MAX_TRAIL];
    int    trail_ptr = 0;
    int    trail_count = 0;

    void AddTrail(ImVec2 p) {
        trail[trail_ptr] = p;
        trail_ptr = (trail_ptr + 1) % ABS_MAX_TRAIL;
        if (trail_count < ABS_MAX_TRAIL) trail_count++;
    }
};

// 基础物理质点
struct Particle {
    ImVec2 pos;  // 位置
    ImVec2 vel;  // 速度
    float  type; // 粒子类型 (0=纵波, 1=横波/渲染着色)
};

// =============================================================================
// 5. 地震波场模拟物理建模参数 (Physics Parameters)
// =============================================================================

// 基础网格物理几何
struct Model {
    int   xnum = 500;
    int   znum = 500;
    float dx = 1.0f;
    float dz = 1.0f;
};

// 观测系统几何配置
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

// 有限差分核心控制参数 (Finite Difference Method)
struct FDM {
    float npml = 50.0f;
    float xPace = 1.0f;
    float zPace = 1.0f;
    float f0 = 100.0f;
    float t0 = 1.0f / 100.0f;
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

// 汇总组合参数
struct Parameters {
    Model    model;
    Geometry geom;
    FDM      FDM;
};

// 活跃震源的单体参数
struct GPUSource {
    int   idx;    // 在2D网格平面中展平的 1D 偏移索引
    float t;      // 震源已累积传播的时间
    float f_peak; // 震源子波峰值频率
    float amp;    // 震源振幅系数
};

// GPU 端弹性波有限差分运行上下文
struct SimulationContext {
    int   NX, NZ;
    int   total_grid;
    float h, dt;
    int   nt, npml;
    int   snap_spacenum;
    int   flag_type; // 模拟方案选项: 1(标准场分裂), 2(非分裂PML), 3(含自由表面)

    std::vector<float> rho, mu, lambda, lambda2mu;
    std::vector<float> dp_flat;
    std::vector<float> dx, dx_half, dz, dz_half;

    float c1_h, c2_h, c3_h, c4_h; // 差分权重因子

    int                src_idx;
    int                src_z_idx;
    std::vector<float> wavelet;
    float              src_angle;

    int              num_rcv;
    std::vector<int> rcv_grid_idx;

    std::vector<float> record_vx;
    std::vector<float> record_vz;

    bool upFlag;
};

// =============================================================================
// 6. UI 可视化与地震剖面分析 (UI, Viewport & Analysis Structs)
// =============================================================================

// 屏幕视口映射工具结构
struct ViewportInfo {
    float x = 0.0f; // 视口在屏幕绝对坐标下的左侧坐标
    float y = 0.0f; // 视口在屏幕绝对坐标下的顶部坐标
    float w = 1.0f; // 视口绝对像素宽度
    float h = 1.0f; // 视口绝对像素高度
    float scaleX = 1.0f; // 屏幕 X 像素 与 模拟格点 转换比
    float scaleY = 1.0f; // 屏幕 Y 像素 与 模拟格点 转换比
};

// 实时地震记录波道图分析控制参数
struct AnalysisState {
    bool  isOpen = false;
    int   numChannels = 0;                         // 检波器总道数 (nTraces)
    int   numSamples = 0;                         // 每道的采样点数 (nSamples)
    float globalMaxAmp = 0.0f;                      // 全局最大绝对振幅
    float samplingInterval = 0.001f;                    // 采样时间增量 dt (秒)

    std::vector<std::vector<float>> traces;             // 剖面道数据 [channel][sample]
    std::vector<float>              heatmapData;        // 用于一维展平的背景矩阵数据

    bool  showWiggle = true;                         // 是否绘制变振幅 Wiggle 细线
    bool  showHeatmap = true;                         // 是否绘制 Heatmap 背景
    int   colormapIndex = 0;                            // 图谱索引 (0: RdBu, 1: PiYG, 2: Spectral)
    ImVec4 colorLine = ImVec4(0.0f, 0.0f, 0.0f, 1.0f); // Wiggle 波形线颜色
    float displayGain = 1.0f;                         // 道图显示增益系数
    float heatmapMin = -1.0f;
    float heatmapMax = 1.0f;
    bool  fitRequest = false;                        // 请求自动缩放贴合显示区域
};

// 道图绘图上下文，用于通过 ImGui/ImPlot 渲染 Wiggle 细波形线
struct PlotContext {
    const std::vector<float>* traceData;
    float offsetX;
    float gain;
    float dt;
    bool  useTime;
    int   startSample;
    int   step;
};

// =============================================================================
// 7. OpenGL 着色器全局加载编译工具函数 (Shader Utilities)
// =============================================================================

/**
 * @brief 从物理文件加载完整的着色器源码文本
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
 * @brief 创建、填充、编译单个着色器并输出编译日志
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

// 离屏渲染基础几何体的声明
void InitQuad(GLHandles& gl);