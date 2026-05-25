#version 430 core
out vec4 FragColor;
in vec2 TexCoords;

uniform sampler2D seisTexture;   // 地震波场颜色纹理 (CUDA零拷贝写入)
uniform sampler2D modelTexture;  // 离屏地质物性纹理 (R: Vp, G: Vs, B: Rho)
uniform bool useModelTexture;    // 是否启用非均匀速度模型纹理
uniform float uniformVp;         // 均匀介质下的纵波速度 (由 C++ 实时滑块传入)
uniform float uniformRho;        // 均匀介质下的密度 (由 C++ 实时滑块传入)

uniform vec2 simSize;   
uniform float viewZoom;
uniform float totalTime;
uniform float npml;              // PML 边界格数
uniform int waveStyle;           // 0: 熔岩, 1: 3D 釉面, 2: 经典红蓝
uniform int modelStyle;          // 背景风格: 0: 经典钛金, 1: 科学地质图, 2: 灰度Vp, 3: Viridis, 4: Cyber

uniform bool showGrid;        // <-- 新增：控制网格与矩阵点显示的开关

// 4 级色谱线性插值辅助函数
vec3 ramp4(float t, vec3 c1, vec3 c2, vec3 c3, vec3 c4) {
    if (t < 0.333) return mix(c1, c2, t * 3.0);
    if (t < 0.666) return mix(c2, c3, (t - 0.333) * 3.0);
    return mix(c3, c4, (t - 0.666) * 3.0);
}

// 物理 Vp 速度值提取
float calcVp(vec2 uv) {
    if (useModelTexture) {
        return texture(modelTexture, uv).r * 6000.0; // 映射至 0~6000 m/s
    }
    return uniformVp;
}

// 物理密度值提取
float calcRHO(vec2 uv) {
    if (useModelTexture) {
        return texture(modelTexture, uv).b * 3000.0; // 映射至 0~3000 kg/m3
    }
    return uniformRho;
}

// 抗锯齿网格自适应线宽函数
float getGridLine(float pos, float spacing, float pixelWidth) {
    float coord = pos / spacing;
    float df = max(fwidth(coord), 0.0001);
    float grid = abs(fract(coord - 0.5) - 0.5) / df;
    float line = 1.0 - min(grid / pixelWidth, 1.0);
    return line;
}

// 无分支 2D 矩形 SDF 距离函数
float sdBox(vec2 p, vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

// =============================================================================
// 【 核心重构：高度契合多地质图风格的自适应背景生成函数 】
// =============================================================================
vec3 getGeologicalBackground(vec2 uv, vec2 cellPos) {
    // 默认 Style 0: 经典钛金灰深色底
    vec3 bgCol = vec3(0.012, 0.015, 0.018); 
    
    float vp = calcVp(uv);
    float t = clamp((vp - 500.0) / 5500.0, 0.0, 1.0);
    
    // --- Geological Styles 切换 (保留经典核心样式) ---
    if (modelStyle == 0) {
        // Style 0: 经典钛金灰
        bgCol = vec3(0.012, 0.015, 0.018);
    }
    else if (modelStyle == 1) {
        // Style 1: 科学地质图 (Modern Geological Map)
        vec3 cWater = vec3(0.0, 0.22, 0.45);  // 深蓝水层/极软层
        vec3 cSed   = vec3(0.28, 0.55, 0.42); // 青灰沉积层
        vec3 cRock  = vec3(0.75, 0.65, 0.28); // 沙黄硬砂岩
        vec3 cHard  = vec3(0.68, 0.18, 0.12); // 红褐基岩/超硬层

        if (t < 0.333)       bgCol = mix(cWater, cSed, t * 3.0);
        else if (t < 0.666)  bgCol = mix(cSed, cRock, (t - 0.333) * 3.0);
        else                 bgCol = mix(cRock, cHard, (t - 0.666) * 3.0);

        // 邻域差分边缘检测 (非均匀介质下自动在层位边界绘制黑色科学剖分线)
        if (useModelTexture) {
            vec2 texelSize = 1.0 / simSize;
            float vp_right = calcVp(uv + vec2(texelSize.x, 0.0));
            float vp_up    = calcVp(uv + vec2(0.0, texelSize.y));
            float diff = abs(vp - vp_right) + abs(vp - vp_up);
            float edge = smoothstep(10.0, 50.0, diff);
            bgCol = mix(bgCol, vec3(0.05, 0.05, 0.08), edge * 0.4);
        }
    }
    else if (modelStyle == 2) {
        // Style 2: 经典灰度 Vp 强度图
        bgCol = vec3(t * 0.6 + 0.08);
    }
    else if (modelStyle == 3) {
        // Style 3: 经典高对比度 Viridis 色谱 (绿-黄-蓝)
        vec3 c1 = vec3(0.26, 0.00, 0.33); vec3 c2 = vec3(0.19, 0.40, 0.56);
        vec3 c3 = vec3(0.12, 0.63, 0.53); vec3 c4 = vec3(0.99, 0.90, 0.14);
        bgCol = ramp4(t, c1, c2, c3, c4);
    }
    else if (modelStyle == 4) {
        // Style 4: 赛博高能高对比度色谱
        vec3 c1 = vec3(0.02, 0.08, 0.15); vec3 c2 = vec3(0.00, 0.45, 0.55);
        vec3 c3 = vec3(0.30, 0.00, 0.60); vec3 c4 = vec3(0.90, 0.10, 0.50);
        bgCol = ramp4(t, c1, c2, c3, c4);
    }

    // =============================================================================
    // 【 核心优化 】PML 阻尼层斜条纹自适应光影融合 (Damping Stripes)
    // 采用“动态相乘变暗”代替“强行混合钛灰色”，让地质图边界自然变暗，杜绝钛灰色色块！
    // =============================================================================
    if (npml > 0.0) {
        float distToEdge = min(min(cellPos.x, simSize.x - cellPos.x), 
                               min(cellPos.y, simSize.y - cellPos.y));

        if (distToEdge < npml) {
            // 斜向阻尼线光栅计算
            float stripe = step(0.5, fract((cellPos.x + cellPos.y) * 0.15));
            float fadeFactor = (0.25 + 0.15 * stripe) * smoothstep(0.0, npml, npml - distToEdge);
            
            // 关键：不混合 vec3(0.06)，而是直接将当前的 bgCol 与 bgCol * 0.15 混合，实现高雅的物理变暗
            bgCol = mix(bgCol, bgCol * 0.15, fadeFactor); 
        }
    }

    return bgCol;
}

void main() {
    // 1. 垂直翻转纹理坐标 Y (地表顶端对齐)
    vec2 flippedTexCoords = vec2(TexCoords.x, 1.0 - TexCoords.y);

    // 2. 采样地震波场颜色
    vec4 texColor = texture(seisTexture, flippedTexCoords);

    // 3. 将纹理坐标映射到实际网格单位
    vec2 cellPos = flippedTexCoords * simSize;

    // 4. 计算高度定制化的地质/阻尼层融合背景 (100% 物理真实)
    vec3 baseBG = getGeologicalBackground(flippedTexCoords, cellPos);

    // =============================================================================
    // 【 5. 逆向物理重构：解构振幅并计算 3D 凹凸起伏 】
    // =============================================================================
    float v = 0.0;
    if (texColor.r > texColor.b) {
        v = 1.0 - texColor.g;      // 正振幅
    } else {
        v = -(1.0 - texColor.g);   // 负振幅
    }

    // 偏导数重构 3D 凹凸表面法线
    float dx = dFdx(v) * 12.0; 
    float dy = dFdy(v) * 12.0;
    vec3 normal = normalize(vec3(-dx, -dy, 1.0));
    float gradLen = length(vec2(dx, dy));

    // =============================================================================
    // 【 6. 多重渲染风格计算 (Styles) 】
    // =============================================================================
    vec3 waveColor = baseBG;
    float waveIntensity = 0.0;

    // --- Style 0: Magma (发光熔岩能量场) ---
    if (waveStyle == 0) {
        float energy = 1.0 - exp(-abs(v) * 8.0);
        
        vec3 c0 = baseBG;                 // 熔岩消隐后，无缝化为地质图底图
        vec3 c1 = vec3(0.25, 0.05, 0.35); // 深紫色
        vec3 c2 = vec3(0.70, 0.15, 0.40); // 紫红色
        vec3 c3 = vec3(0.95, 0.50, 0.15); // 亮橙色
        vec3 c4 = vec3(1.0, 1.0, 1.0);    // 纯白核心
        
        vec3 col;
        if (energy < 0.25) {
            col = mix(c0, c1, energy * 4.0);
        } else if (energy < 0.5) {
            col = mix(c1, c2, (energy - 0.25) * 4.0);
        } else if (energy < 0.75) {
            col = mix(c2, c3, (energy - 0.5) * 4.0);
        } else {
            col = mix(c3, c4, (energy - 0.75) * 4.0);
        }
        
        waveColor = col;
        waveIntensity = smoothstep(0.03, 0.35, energy);
    }
    // --- Style 1: Deep Coolwarm (3D 釉面起伏冷暖色) ---
    else if (waveStyle == 1) {
        float scaled_v = clamp(v * 2.5, -1.0, 1.0);
    
        vec3 col_warm = vec3(0.85, 0.05, 0.1);   // 浓郁宝石红
        vec3 col_cool = vec3(0.05, 0.25, 0.85);  // 深邃皇家蓝
        
        // 【核心修复】：中性过渡色直接绑定 baseBG，消隐时完美沉入地层，拒绝脏灰色色块！
        vec3 col_mid  = baseBG;                  

        vec3 base;
        float intensity = pow(abs(scaled_v), 0.45); 
        if (scaled_v > 0.0) {
            base = mix(col_mid, col_warm, intensity);
        } else {
            base = mix(col_mid, col_cool, intensity);
        }

        // 乘法漫反射阴影，营造 3D 立体波峰起伏
        vec3 lightDir = normalize(vec3(-0.4, 0.5, 0.8));
        float shading = dot(normal, lightDir) * 0.25 + 0.95; 
        vec3 finalCol = base * shading;

        // 釉面高光反射
        float spec = pow(max(dot(normal, vec3(0,0,1)), 0.0), 64.0);
        finalCol += vec3(1.0) * spec * 0.25 * abs(scaled_v);

        // Alpha 阻尼消隐控制
        float alpha = smoothstep(0.01, 0.1, abs(scaled_v) + gradLen * 0.05);
    
        waveColor = finalCol;
        waveIntensity = clamp(alpha, 0.0, 1.0);
    }
    // --- Style 2: Default (经典荧光红蓝) ---
    else {
        waveColor = texColor.rgb;
        waveIntensity = clamp(1.0 - texColor.g, 0.0, 1.0);
        if (waveIntensity > 0.02) {
            waveColor *= 1.3; 
        }
    }

    // 最终混合：无波场区域完全渲染为高保真的地层地质图，有波场区域叠加霓虹波纹
    vec3 finalColor = mix(baseBG, waveColor, waveIntensity);

    // =============================================================================
    // 7. 【物理双重网格绘制】
    // =============================================================================
    float pxToGridUnits = fwidth(cellPos.x); 

    // A. 次网格 (白色)
    float subLineX = getGridLine(cellPos.x, 1.0, 1.0);
    float subLineY = getGridLine(cellPos.y, 1.0, 1.0);
    float gridSub = max(subLineX, subLineY);
    float opacitySub = 0.0196 * smoothstep(1.5, 0.4, pxToGridUnits); 

    // B. 主网格 (白色)
    float mainInterval = 5.0;
    float mainLineX = getGridLine(cellPos.x, mainInterval, 1.5);
    float mainLineY = getGridLine(cellPos.y, mainInterval, 1.5);
    float gridMain = max(mainLineX, mainLineY);
    float opacityMain = 0.0588 * smoothstep(15.0, 4.0, pxToGridUnits); 

    // C. 主网格十字交点圆点 (青色)
    vec2 mainCoord = cellPos / mainInterval;
    vec2 nearestIntersection = round(mainCoord);
    vec2 offsetToIntersection = (mainCoord - nearestIntersection) * mainInterval;
    float distToIntersection = length(offsetToIntersection);
    float distInPixels = distToIntersection / max(pxToGridUnits, 0.0001);
    
    float dotRadius = 3.0;
    float dotAA = 1.0;
    float gridDot = smoothstep(dotRadius + dotAA, dotRadius - dotAA, distInPixels);
    float opacityDot = 0.137 * smoothstep(15.0, 4.0, pxToGridUnits); 

    // 核心修改：仅在 showGrid 为 true 时，才向画面混合网格与圆点，否则保持纯净地质背景
    if (showGrid) {
        finalColor = mix(finalColor, vec3(1.0, 1.0, 1.0), gridSub * opacitySub);   
        finalColor = mix(finalColor, vec3(1.0, 1.0, 1.0), gridMain * opacityMain); 
        finalColor = mix(finalColor, vec3(0.0, 1.0, 1.0), gridDot * opacityDot);   
    }   

    // =============================================================================
    // 8. 绘制 PML 吸收边界荧光线
    // =============================================================================
    if (npml > 0.0) {
        vec2 boxCenter = simSize * 0.5;
        vec2 halfExtents = (simSize * 0.5) - vec2(npml);
        float distToPml = abs(sdBox(cellPos - boxCenter, halfExtents));

        float pmlLineThickness = 1.5; 
        float pmlAA = fwidth(distToPml);
        float pmlBorder = smoothstep(pmlAA * pmlLineThickness, 0.0, distToPml - 0.05);
        
        vec3 pmlLineColor = vec3(0.5, 1.0, 0.5); 
        float topFade = smoothstep(0.0, npml + 2.0, cellPos.y);
        
        finalColor = mix(finalColor, pmlLineColor, pmlBorder * 0.65 * topFade);
    }

    // =============================================================================
    // 9. 绘制最外层物理边界框 (白框)
    // =============================================================================
    float dOuterLeft   = abs(cellPos.x - 1.0);
    float dOuterRight  = abs(cellPos.x - (simSize.x - 1.0));
    float dOuterBottom = abs(cellPos.y - 1.0);
    float dOuterTop    = abs(cellPos.y - (simSize.y - 1.0));

    float distToOuter = 1e6;
    if(cellPos.x >= 0.9 && cellPos.x <= simSize.x - 0.9) {
        distToOuter = min(distToOuter, dOuterBottom);
        distToOuter = min(distToOuter, dOuterTop);
    }
    if(cellPos.y >= 0.9 && cellPos.y <= simSize.y - 0.9) {
        distToOuter = min(distToOuter, dOuterLeft);
        distToOuter = min(distToOuter, dOuterRight);
    }

    float outerAA = fwidth(distToOuter);
    float outerBorder = smoothstep(outerAA * 1.5, 0.0, distToOuter - 0.05);
    finalColor = mix(finalColor, vec3(1.0, 1.0, 1.0), outerBorder * 0.4); 

    // 12. 输出结果
    FragColor = vec4(finalColor, 0.8); 
}