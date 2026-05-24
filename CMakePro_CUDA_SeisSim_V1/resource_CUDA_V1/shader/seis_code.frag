#version 430 core
out vec4 FragColor;
in vec2 TexCoords;

uniform sampler2D seisTexture; // 地震波场纹理
uniform vec2 simSize;   
uniform float viewZoom;
uniform float totalTime;
uniform float npml;           // PML 边界格数
uniform int waveStyle;        // 0: Magma (熔岩发光), 1: Deep Coolwarm (3D 釉面冷暖), 2: Default (荧光红蓝)

// =============================================================================
// 【高级屏幕空间抗锯齿网格绘制函数】
// =============================================================================
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

void main() {
    // 1. 基础背景 (深色钛金灰)
    vec3 baseBG = vec3(0.012, 0.015, 0.018); 
    
    // 2. 垂直翻转纹理坐标 Y (对齐地表)
    vec2 flippedTexCoords = vec2(TexCoords.x, 1.0 - TexCoords.y);

    // 3. 采样原始地震波场颜色
    vec4 texColor = texture(seisTexture, flippedTexCoords);

    // 4. 将纹理坐标映射到实际网格单位
    vec2 cellPos = flippedTexCoords * simSize;

    // =============================================================================
    // 【 5. 核心黑科技：解构波值并重构 3D 法线与坡度 】
    // =============================================================================
    // A. 逆向重构有符号波场幅值 v ∈ [-1.0, 1.0]
    float v = 0.0;
    if (texColor.r > texColor.b) {
        v = 1.0 - texColor.g;      // 正振幅 (红色系)
    } else {
        v = -(1.0 - texColor.g);   // 负振幅 (蓝色系)
    }

    // B. 利用屏幕空间偏导数计算 3D 表面凹凸法线 (12.0 乘数控制凹凸立体感)
    float dx = dFdx(v) * 12.0; 
    float dy = dFdy(v) * 12.0;
    vec3 normal = normalize(vec3(-dx, -dy, 1.0));
    float gradLen = length(vec2(dx, dy));

    // =============================================================================
    // 【 6. 多重渲染风格切换 (Styles) 】
    // =============================================================================
    vec3 waveColor = baseBG;
    float waveIntensity = 0.0;

    // --- Style 0: Magma (发光熔岩能量场) ---
    if (waveStyle == 0) {
        // 计算平滑指数能量
        float energy = 1.0 - exp(-abs(v) * 8.0);
        
        // 5 级发光色阶
        vec3 c0 = baseBG;                 // 深钛金背景底色
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
        vec3 col_mid  = baseBG;                  // 完美过渡至钛金背景

        vec3 base;
        float intensity = pow(abs(scaled_v), 0.45); 
        if (scaled_v > 0.0) {
            base = mix(col_mid, col_warm, intensity);
        } else {
            base = mix(col_mid, col_cool, intensity);
        }

        // 乘法漫反射阴影，营造 3D 起伏感 [1.1.3, 1.2.7]
        vec3 lightDir = normalize(vec3(-0.4, 0.5, 0.8));
        float shading = dot(normal, lightDir) * 0.25 + 0.95; 
        vec3 finalCol = base * shading;

        // 釉面高光 specular 反光效果
        float spec = pow(max(dot(normal, vec3(0,0,1)), 0.0), 64.0);
        finalCol += vec3(1.0) * spec * 0.25 * abs(scaled_v);

        // Alpha 控制，掩盖背景
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

    // 混合波场到背景上
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

    finalColor = mix(finalColor, vec3(1.0, 1.0, 1.0), gridSub * opacitySub);   
    finalColor = mix(finalColor, vec3(1.0, 1.0, 1.0), gridMain * opacityMain); 
    finalColor = mix(finalColor, vec3(0.0, 1.0, 1.0), gridDot * opacityDot);   

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
        
        vec3 pmlLineColor = vec3(0.5,1,0.5); 
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

    // 10. 后期扫描线
    float scanline = sin(TexCoords.y * 1080.0 + totalTime * 4.0) * 0.01;
    finalColor += scanline;

    // 11. 后期暗角
    float vignette = distance(TexCoords, vec2(0.5));
    finalColor *= smoothstep(1.3, 0.45, vignette);

    FragColor = vec4(finalColor, 0.8); 
}