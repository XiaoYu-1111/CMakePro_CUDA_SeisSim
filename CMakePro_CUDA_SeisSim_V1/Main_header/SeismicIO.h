#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cstdint> // for uint32_t, int16_t, int32_t
#include <cmath>
#include <cstring>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace SeismicIO {

    // =============================================================================
    // 1. 【新增】：专门针对数据采集的单道元数据结构体 (符合标准 SEG-Y 道头物理位置)
    // =============================================================================
    struct TraceMetadata {
        int32_t fieldRecordNum = 1; // 原始野外记录生产号 (第 9-12 字节, Field Record Number)
        int32_t traceInRecord = 1; // 本道在当前记录/炮集中的序号 (第 13-16 字节, Trace Number)
        int32_t sourceX = 0; // 震源/炮点物理 X 坐标 (第 73-76 字节, Source Coordinate X, 乘系数后以米或分米为单位)
        int32_t sourceY = 0; // 震源/炮点物理 Y 坐标 (第 77-80 字节, Source Coordinate Y)
        int32_t groupX = 0; // 接收器/检波点物理 X 坐标 (第 81-84 字节, Group Coordinate X)
        int32_t groupY = 0; // 接收器/检波点物理 Y 坐标 (第 85-88 字节, Group Coordinate Y)
    };

    // =============================================================================
    // 2. SEG-Y 格式多维数据读取接口
    // =============================================================================
    // 读取 2D 二进制 SEG-Y 物理模型或波场记录数据
    std::vector<std::vector<float>> readSegyFile2D(std::string inputfile);

    // =============================================================================
    // 3. SEG-Y 格式数据写入接口 (多重载支持)
    // =============================================================================

    // 重载 1: 适用于导出标准的 2D 连续物性网格 (如 Vp, Vs, Rho 物理参数剖面)
    void writeSegyFile2D(const std::vector<std::vector<float>>& dataArray, const std::string outputfile, float dt);

    // 重载 2: 适用于 1D 展平物理网格数据的扁平化输出
    void writeSegyFile2D(const std::vector<float>& flatData, int nTraces, int nSamples, const std::string& outputfile, float dt);

    // 【新增】重载 3: 适用于采集到的高保真多道地震剖面 (Seismic Shot Gather) 导出
    // 该函数在写入时会根据 metadata 数组中的参数填充每道的 240 字节 Trace Header (炮检距、物理坐标等)，以便专业处理软件识别
    void writeSegyFile2D(const std::vector<std::vector<float>>& dataArray,
        const std::vector<TraceMetadata>& metadata,
        const std::string& outputfile,
        float                                  dt);

    // =============================================================================
    // 4. 辅助文本导出接口
    // =============================================================================
    // 简单文本写入 (CSV style)
    void writeTextFile2D(const std::vector<std::vector<float>>& dataArray, const std::string& outputfile);

    // =============================================================================
    // 5. 字节序转换及数据解析底层辅助函数
    // =============================================================================
    uint32_t swap4byte(uint32_t value);
    int16_t  swap2byte(int16_t value);
    float    ibm2float(uint32_t x);

} // namespace SeismicIO