#include "SeismicIO.h"

namespace SeismicIO {

    // =============================================================================
    // 1. 底层大/小端序转换与浮点数格式转换算法 (IBM/IEEE)
    // =============================================================================

    // 交换 4 字节大端转小端 (int/float)
    uint32_t swap4byte(uint32_t value) {
        return ((value & 0xFF000000) >> 24) |
            ((value & 0x00FF0000) >> 8) |
            ((value & 0x0000FF00) << 8) |
            ((value & 0x000000FF) << 24);
    }

    // 交换 2 字节大端转小端 (short)
    int16_t swap2byte(int16_t value) {
        // 强制转换为无符号进行移位，防止符号位扩展问题
        uint16_t uval = static_cast<uint16_t>(value);
        return static_cast<int16_t>((uval >> 8) | (uval << 8));
    }

    // IBM Float 转 IEEE Float (学术级高速转换实现)
    float ibm2float(uint32_t x) {
        if (x == 0) return 0.0f;

        // IBM 格式: 
        // Bit 0: Sign
        // Bit 1-7: Exponent (Base 16, Bias 64)
        // Bit 8-31: Mantissa (24 bits)
        int sign = (x >> 31) & 0x01;
        int exponent = (x >> 24) & 0x7F;
        int mantissa = x & 0x00FFFFFF;

        // IBM 指数是 16 进制的，Bias 为 64
        // 值 = (-1)^sign * 0.mantissa * 16^(exponent - 64)
        float f_mant = static_cast<float>(mantissa) * 5.96046448e-8f; // * 1.0 / pow(2, 24)
        float result = (1.0f - 2.0f * sign) * f_mant * std::pow(16.0f, exponent - 64);

        return result;
    }

    // =============================================================================
    // 2. 二维 SEG-Y 数据高保真读取驱动
    // =============================================================================
    std::vector<std::vector<float>> readSegyFile2D(std::string inputfile) {
        std::ifstream filein(inputfile, std::ios::binary);

        // 1. 错误检查：打开失败立即返回
        if (!filein.is_open()) {
            std::cerr << "Error: Cannot open file " << inputfile << std::endl;
            return {};
        }

        // 2. 读取卷头
        // 3200 字节 文本卷头 (EBCDIC or ASCII) - 暂时跳过不处理
        filein.seekg(3200, std::ios::beg);

        // 400 字节 二进制卷头
        std::vector<char> binHeader(400);
        filein.read(binHeader.data(), 400);

        // 读取采样点数 (Bytes 20-21 in binary header, i.e., index 20)
        // 注意：SEGY 标准是大端序 (Big-Endian)
        int16_t* pSampleCount = reinterpret_cast<int16_t*>(&binHeader[20]);
        int      trace_length = swap2byte(*pSampleCount);

        // 读取数据格式代码 (Bytes 24-25 in binary header, i.e., index 24)
        // 1 = IBM Float, 5 = IEEE Float
        int16_t* pFormatCode = reinterpret_cast<int16_t*>(&binHeader[24]);
        int      format_code = swap2byte(*pFormatCode);

        // 3. 计算道数
        filein.seekg(0, std::ios::end);
        long long fileSize = filein.tellg();
        long long trace_total_bytes = 240 + 4 * static_cast<long long>(trace_length); // 道头 240 字节 + 数据
        long long traces = (fileSize - 3600) / trace_total_bytes;

        std::cout << "File: " << inputfile << "\n"
            << "Trace Length: " << trace_length << "\n"
            << "Total Traces: " << traces << "\n"
            << "Format Code: " << format_code << (format_code == 1 ? " (IBM)" : (format_code == 5 ? " (IEEE)" : " (Unknown)"))
            << std::endl;

        if (traces <= 0 || trace_length <= 0) {
            std::cerr << "Error: Invalid trace count or length." << std::endl;
            return {};
        }

        // 4. 读取数据
        filein.seekg(3600, std::ios::beg); // 跳回第一道开始处

        // 预分配内存，避免 vector 动态缩容
        std::vector<std::vector<float>> dataArray(traces, std::vector<float>(trace_length));

        // 缓冲区：每次读取一道的数据部分 (4 bytes * length)
        std::vector<char>     traceHeader(240);
        std::vector<uint32_t> rawData(trace_length);

        for (int i = 0; i < traces; i++) {
            if (i % 10000 == 0 && i > 0) std::cout << "Reading trace: " << i << std::endl;

            // 读取道头和道数据
            filein.read(traceHeader.data(), 240);
            filein.read(reinterpret_cast<char*>(rawData.data()), trace_length * 4);

            for (int j = 0; j < trace_length; j++) {
                // SEGY 是大端序，PC 是小端序，必须交换字节
                uint32_t val = swap4byte(rawData[j]);

                if (format_code == 1) {
                    // IBM Floating Point
                    dataArray[i][j] = ibm2float(val);
                }
                else if (format_code == 5) {
                    // IEEE Floating Point
                    float f_val;
                    std::memcpy(&f_val, &val, sizeof(float));
                    dataArray[i][j] = f_val;
                }
                else {
                    // 默认降级为 IEEE
                    float f_val;
                    std::memcpy(&f_val, &val, sizeof(float));
                    dataArray[i][j] = f_val;
                }
            }
        }

        filein.close();
        return dataArray;
    }

    // =============================================================================
    // 3. 保存二维剖面物性模型为标准 SEGY 格式 (重载 1)
    // =============================================================================
    void writeSegyFile2D(const std::vector<std::vector<float>>& dataArray, const std::string outputfile, float dt) {
        std::ofstream fileout(outputfile, std::ios::binary);
        if (!fileout.is_open()) {
            std::cerr << "Error: Cannot create file " << outputfile << std::endl;
            return;
        }

        int num_traces = dataArray.size();
        if (num_traces == 0) return;
        int num_samples = dataArray[0].size();

        // ---------------------------------------------------------
        // 1. 写入文本卷头 (3200 bytes)
        // ---------------------------------------------------------
        std::vector<char> textHeader(3200, ' ');
        std::string       desc = "C01 SEGY FILE GENERATED BY SEIS-FDM-TOOL";
        std::memcpy(textHeader.data(), desc.c_str(), desc.size());
        fileout.write(textHeader.data(), 3200);

        // ---------------------------------------------------------
        // 2. 写入二进制卷头 (400 bytes)
        // ---------------------------------------------------------
        std::vector<char> binHeader(400, 0);

        // Byte 17-18: 采样间隔 (微秒)
        int16_t interval_us = static_cast<int16_t>(dt * 1000000);
        int16_t be_interval = swap2byte(interval_us);
        std::memcpy(&binHeader[16], &be_interval, 2);

        // Byte 21-22: 每道采样点数
        int16_t samples = static_cast<int16_t>(num_samples);
        int16_t be_samples = swap2byte(samples);
        std::memcpy(&binHeader[20], &be_samples, 2);

        // Byte 25-26: 数据格式代码 (5 = IEEE Float)
        int16_t format = swap2byte(5);
        std::memcpy(&binHeader[24], &format, 2);

        fileout.write(binHeader.data(), 400);

        // ---------------------------------------------------------
        // 3. 写入道数据 (Trace Header + Data)
        // ---------------------------------------------------------
        std::vector<char> traceHeader(240, 0);

        for (int i = 0; i < num_traces; ++i) {
            std::fill(traceHeader.begin(), traceHeader.end(), 0);

            // Byte 1-4: 线号/道顺序号 (Trace sequence number within line)
            int32_t trace_seq = swap4byte(i + 1);
            std::memcpy(&traceHeader[0], &trace_seq, 4);

            // Byte 115-116: 采样点数
            std::memcpy(&traceHeader[114], &be_samples, 2);

            // Byte 117-118: 采样间隔 (微秒)
            std::memcpy(&traceHeader[116], &be_interval, 2);

            // 写入道头
            fileout.write(traceHeader.data(), 240);

            // --- 写入数据 (IEEE Float Big-Endian) ---
            for (int j = 0; j < num_samples; ++j) {
                float val = dataArray[i][j];

                // 强转为 uint32 进行位操作
                uint32_t val_int;
                std::memcpy(&val_int, &val, 4);

                uint32_t be_val = swap4byte(val_int);
                fileout.write(reinterpret_cast<char*>(&be_val), 4);
            }
        }

        fileout.close();
        std::cout << "SEGY Saved: " << outputfile << " (" << num_traces << "x" << num_samples << ")" << std::endl;
    }

    // =============================================================================
    // 4. 重载 2：从 1D Flat vector 写入 (高性能，无拷贝)
    // =============================================================================
    void writeSegyFile2D(const std::vector<float>& flatData, int nTraces, int nSamples, const std::string& outputfile, float dt) {
        std::ofstream fileout(outputfile, std::ios::binary);
        if (!fileout.is_open()) return;

        // 1. Text Header
        std::vector<char> textHeader(3200, ' ');
        std::string       desc = "SEGY GENERATED BY SEISMIC-IO LIB (FLAT)";
        std::memcpy(textHeader.data(), desc.c_str(), desc.size());
        fileout.write(textHeader.data(), 3200);

        // 2. Binary Header
        std::vector<char> binHeader(400, 0);
        int16_t           interval_us = swap2byte(static_cast<int16_t>(dt * 1000000));
        std::memcpy(&binHeader[16], &interval_us, 2);

        int16_t ns_be = swap2byte(static_cast<int16_t>(nSamples));
        std::memcpy(&binHeader[20], &ns_be, 2);

        int16_t fmt_be = swap2byte(5); // IEEE
        std::memcpy(&binHeader[24], &fmt_be, 2);
        fileout.write(binHeader.data(), 400);

        // 3. Data
        std::vector<char> traceHeader(240, 0);
        for (int i = 0; i < nTraces; ++i) {
            std::fill(traceHeader.begin(), traceHeader.end(), 0);
            int32_t seq = swap4byte(i + 1);
            std::memcpy(&traceHeader[0], &seq, 4);
            std::memcpy(&traceHeader[114], &ns_be, 2);
            std::memcpy(&traceHeader[116], &interval_us, 2);
            fileout.write(traceHeader.data(), 240);

            size_t offset = (size_t)i * nSamples;
            for (int j = 0; j < nSamples; ++j) {
                float    val = flatData[offset + j];
                uint32_t val_int;
                std::memcpy(&val_int, &val, 4);
                val_int = swap4byte(val_int);
                fileout.write(reinterpret_cast<char*>(&val_int), 4);
            }
        }
        fileout.close();
    }

    // =============================================================================
    // 5. 【新增】：重载 3：专用于“地震道集数据采集（Shot Gather）”的高保真导出
    // =============================================================================
    void writeSegyFile2D(const std::vector<std::vector<float>>& dataArray,
        const std::vector<TraceMetadata>& metadata,
        const std::string& outputfile,
        float                                  dt) {
        std::ofstream fileout(outputfile, std::ios::binary);
        if (!fileout.is_open()) {
            std::cerr << "Error: Cannot create file " << outputfile << std::endl;
            return;
        }

        int num_traces = dataArray.size();
        if (num_traces == 0) return;
        int num_samples = dataArray[0].size();

        // 1. Text Header (3200 bytes)
        std::vector<char> textHeader(3200, ' ');
        std::string       desc = "C01 SEGY SHOT GATHER DATA EXPORTED BY ACQUISITION MODULE";
        std::memcpy(textHeader.data(), desc.c_str(), desc.size());
        fileout.write(textHeader.data(), 3200);

        // 2. Binary Header (400 bytes)
        std::vector<char> binHeader(400, 0);
        int16_t           interval_us = static_cast<int16_t>(dt * 1000000);
        int16_t           be_interval = swap2byte(interval_us);
        std::memcpy(&binHeader[16], &be_interval, 2);

        int16_t samples = static_cast<int16_t>(num_samples);
        int16_t be_samples = swap2byte(samples);
        std::memcpy(&binHeader[20], &be_samples, 2);

        int16_t format = swap2byte(5); // IEEE Float
        std::memcpy(&binHeader[24], &format, 2);
        fileout.write(binHeader.data(), 400);

        // 3. Trace Data
        std::vector<char> traceHeader(240, 0);

        for (int i = 0; i < num_traces; ++i) {
            std::fill(traceHeader.begin(), traceHeader.end(), 0);

            // 获取该道对应的采集物理元数据 (做防空钳位保护)
            TraceMetadata meta;
            if (i < static_cast<int>(metadata.size())) {
                meta = metadata[i];
            }
            else {
                // 如果外部没有传入足够的元数据，退化并设置本道序号为默认值
                meta.traceInRecord = i + 1;
                meta.fieldRecordNum = 1;
            }

            // -------------------------------------------------------------
            // A. 道头物理位置标准映射与字节序转换 (大端) [2]
            // -------------------------------------------------------------
            // Bytes 1-4: 线内道顺序号 [2]
            int32_t trace_seq = swap4byte(i + 1);
            std::memcpy(&traceHeader[0], &trace_seq, 4);

            // Bytes 9-12: 原始野外记录生产号 (Field Record / Shot ID) [2]
            int32_t f_record = swap4byte(meta.fieldRecordNum);
            std::memcpy(&traceHeader[8], &f_record, 4);

            // Bytes 13-16: 野外记录中的道序号 (Trace ID) [2]
            int32_t t_record = swap4byte(meta.traceInRecord);
            std::memcpy(&traceHeader[12], &t_record, 4);

            // Bytes 73-76: 震源物理 X 坐标 (Source X) [2]
            int32_t src_x = swap4byte(meta.sourceX);
            std::memcpy(&traceHeader[72], &src_x, 4);

            // Bytes 77-80: 震源物理 Y 坐标 (Source Y) [2]
            int32_t src_y = swap4byte(meta.sourceY);
            std::memcpy(&traceHeader[76], &src_y, 4);

            // Bytes 81-84: 检波点物理 X 坐标 (Group X) [2]
            int32_t grp_x = swap4byte(meta.groupX);
            std::memcpy(&traceHeader[80], &grp_x, 4);

            // Bytes 85-88: 检波点物理 Y 坐标 (Group Y) [2]
            int32_t grp_y = swap4byte(meta.groupY);
            std::memcpy(&traceHeader[84], &grp_y, 4);

            // Bytes 115-116: 单道采样点数
            std::memcpy(&traceHeader[114], &be_samples, 2);

            // Bytes 117-118: 单道采样率 (微秒)
            std::memcpy(&traceHeader[116], &be_interval, 2);

            // 写入 240 字节 Trace Header [2]
            fileout.write(traceHeader.data(), 240);

            // -------------------------------------------------------------
            // B. 写入实际道集波形振幅数据 (IEEE Big-Endian)
            // -------------------------------------------------------------
            for (int j = 0; j < num_samples; ++j) {
                float val = dataArray[i][j];

                uint32_t val_int;
                std::memcpy(&val_int, &val, 4);

                uint32_t be_val = swap4byte(val_int);
                fileout.write(reinterpret_cast<char*>(&be_val), 4);
            }
        }

        fileout.close();
        std::cout << "[Acquisition IO] SEGY Record Gather Saved: " << outputfile
            << " (" << num_traces << " traces x " << num_samples << " samples)" << std::endl;
    }

    // =============================================================================
    // 6. 文本格式导出
    // =============================================================================
    void writeTextFile2D(const std::vector<std::vector<float>>& dataArray, const std::string& outputfile) {
        std::ofstream fileout(outputfile);
        if (!fileout.is_open()) return;

        fileout.precision(6);
        fileout << std::fixed;

        for (size_t i = 0; i < dataArray.size(); i++) {
            for (size_t j = 0; j < dataArray[i].size(); j++) {
                fileout << dataArray[i][j];
                if (j < dataArray[i].size() - 1) fileout << ",";
            }
            fileout << "\n";
        }
        fileout.close();
    }

} // namespace SeismicIO