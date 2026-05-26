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
uniform int waveStyle;           // 0: Magma, 1~12 对应标准科学色谱
uniform int modelStyle;          // 背景地质图风格: 0: 经典钛金, 1: 科学地质图, 2: 灰度Vp, 3: 跟随波场风格
uniform bool showGrid;

uniform float gridSpacing;       // 网格基础间隔 (如 50.0，对应 UI 的 50m 划分)
uniform float gridOpacity;       // 网格整体不透明度系数 (0.0 ~ 1.0 调节亮度)

// =============================================================================
// 【第一级：零依赖底层几何与差值辅助函数】
// =============================================================================
float getGridLine(float pos, float spacing, float pixelWidth) {
    float coord = pos / spacing;
    float df = max(fwidth(coord), 1e-5);
    float dist = abs(coord - round(coord));
    return 1.0 - smoothstep(pixelWidth, pixelWidth + 1.0, dist / df);
}

// 无分支 2D 矩形 SDF 距离函数 (无外部依赖，优先置顶)
float sdBox(vec2 p, vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

// =============================================================================
// 【第二级：12 套标准的科学色谱单向定义】
// =============================================================================
vec3 getJet(float t) {
    vec3 c0 = vec3(0.0, 0.0, 0.5), c1 = vec3(0.0, 0.5, 1.0), c2 = vec3(0.5, 1.0, 0.5), c3 = vec3(1.0, 0.75, 0.0), c4 = vec3(0.5, 0.0, 0.0);
    if (t < 0.25) return mix(c0, c1, t * 4.0);
    if (t < 0.5) return mix(c1, c2, (t - 0.25) * 4.0);
    if (t < 0.75) return mix(c2, c3, (t - 0.5) * 4.0);
    return mix(c3, c4, (t - 0.75) * 4.0);
}

vec3 getTerrain(float t) {
    vec3 c0 = vec3(0.0, 0.235, 0.667), c1 = vec3(0.0, 0.706, 0.863), c2 = vec3(0.941, 0.941, 0.471), c3 = vec3(0.196, 0.784, 0.196), c4 = vec3(0.471, 0.392, 0.235), c5 = vec3(0.98, 0.98, 0.98);
    if (t < 0.25) return mix(c0, c1, t * 4.0);
    if (t < 0.45) return mix(c1, c2, (t - 0.25) / 0.20);
    if (t < 0.60) return mix(c2, c3, (t - 0.45) / 0.15);
    if (t < 0.80) return mix(c3, c4, (t - 0.60) / 0.20);
    return mix(c4, c5, (t - 0.80) / 0.20);
}

vec3 getSeismic(float t) {
    vec3 c0 = vec3(0.000, 0.000, 0.706), c1 = vec3(0.88, 0.88, 0.88), c2 = vec3(0.706, 0.000, 0.000); // 更稳定的 phase center
    if (t < 0.5) return mix(c0, c1, t * 2.0);
    return mix(c1, c2, (t - 0.5) * 2.0);
}

vec3 getGrayscale(float t) {
    vec3 c0 = vec3(0.12), c1 = vec3(0.50), c2 = vec3(0.88);
    if (t < 0.5) return mix(c0, c1, t * 2.0);
    return mix(c1, c2, (t - 0.5) * 2.0);
}

vec3 getHot(float t) {
    vec3 c0 = vec3(0.0, 0.0, 0.0), c1 = vec3(1.0, 0.0, 0.0), c2 = vec3(1.0, 1.0, 0.0), c3 = vec3(1.0, 1.0, 1.0);
    if (t < 0.333) return mix(c0, c1, t * 3.0);
    if (t < 0.666) return mix(c1, c2, (t - 0.333) * 3.0);
    return mix(c2, c3, (t - 0.666) * 3.0);
}

vec3 getCold(float t) {
    vec3 c0 = vec3(0.0, 0.0, 0.0), c1 = vec3(0.0, 0.0, 1.0), c2 = vec3(0.0, 1.0, 1.0), c3 = vec3(1.0, 1.0, 1.0);
    if (t < 0.333) return mix(c0, c1, t * 3.0);
    if (t < 0.666) return mix(c1, c2, (t - 0.333) * 3.0);
    return mix(c2, c3, (t - 0.666) * 3.0);
}

vec3 getCoolwarm(float t) {
    vec3 c0 = vec3(0.230, 0.299, 0.754), c1 = vec3(0.554, 0.690, 0.996), c2 = vec3(0.865, 0.865, 0.865), c3 = vec3(0.957, 0.598, 0.477), c4 = vec3(0.706, 0.016, 0.150);
    if (t < 0.25) return mix(c0, c1, t * 4.0);
    if (t < 0.5) return mix(c1, c2, (t - 0.25) * 4.0);
    if (t < 0.75) return mix(c2, c3, (t - 0.5) * 4.0);
    return mix(c3, c4, (t - 0.75) * 4.0);
}

vec3 getViridis(float t) {
    vec3 c0 = vec3(0.267, 0.004, 0.329), c1 = vec3(0.282, 0.137, 0.455), c2 = vec3(0.251, 0.404, 0.541), c3 = vec3(0.208, 0.718, 0.475), c4 = vec3(0.561, 0.843, 0.267), c5 = vec3(0.85, 0.88, 0.18); // 压高亮
    if (t < 0.2) return mix(c0, c1, t * 5.0);
    if (t < 0.4) return mix(c1, c2, (t - 0.2) * 5.0);
    if (t < 0.6) return mix(c2, c3, (t - 0.4) * 5.0);
    if (t < 0.8) return mix(c3, c4, (t - 0.6) * 5.0);
    return mix(c4, c5, (t - 0.8) * 5.0);
}

vec3 getPlasma(float t) {
    vec3 c0 = vec3(0.051, 0.031, 0.529), c1 = vec3(0.416, 0.0, 0.659), c2 = vec3(0.733, 0.216, 0.329), c3 = vec3(0.976, 0.557, 0.035), c4 = vec3(0.941, 0.976, 0.129);
    if (t < 0.25) return mix(c0, c1, t * 4.0);
    if (t < 0.50) return mix(c1, c2, (t - 0.25) * 4.0);
    if (t < 0.75) return mix(c2, c3, (t - 0.50) * 4.0);
    return mix(c3, c4, (t - 0.75) * 4.0);
}

vec3 getTurbo(float t) {
    vec3 c0 = vec3(0.190, 0.071, 0.232), c1 = vec3(0.276, 0.421, 0.891), c2 = vec3(0.158, 0.735, 0.923); // deep purple, blue, cyan
    vec3 c3 = vec3(0.120, 0.780, 0.520); // green (降低亮度)
    vec3 c4 = vec3(0.580, 0.820, 0.180); // yellow-green (降低黄区过曝)
    vec3 c5 = vec3(0.930, 0.720, 0.180), c6 = vec3(0.980, 0.420, 0.120), c7 = vec3(0.480, 0.015, 0.015); // orange, red-orange, deep red
    if (t < 0.10) return mix(c0, c1, t * 10.0);
    if (t < 0.20) return mix(c1, c2, (t - 0.10) * 10.0);
    if (t < 0.35) return mix(c2, c3, (t - 0.20) / 0.15);
    if (t < 0.50) return mix(c3, c4, (t - 0.35) / 0.15);
    if (t < 0.70) return mix(c4, c5, (t - 0.50) / 0.20);
    if (t < 0.85) return mix(c5, c6, (t - 0.70) / 0.15);
    return mix(c6, c7, (t - 0.85) / 0.15);
}

vec3 getInferno(float t) {
    vec3 c0 = vec3(0.0, 0.0, 0.016), c1 = vec3(0.341, 0.063, 0.427), c2 = vec3(0.733, 0.216, 0.329), c3 = vec3(0.976, 0.557, 0.035), c4 = vec3(0.988, 1.0, 0.643);
    if (t < 0.25) return mix(c0, c1, t * 4.0);
    if (t < 0.50) return mix(c1, c2, (t - 0.25) * 4.0);
    if (t < 0.75) return mix(c2, c3, (t - 0.50) * 4.0);
    return mix(c3, c4, (t - 0.75) * 4.0);
}

vec3 getPuOr(float t) {
    vec3 c0 = vec3(0.498, 0.231, 0.031), c1 = vec3(0.775, 0.518, 0.235), c2 = vec3(0.992, 0.878, 0.714);
    vec3 c3 = vec3(0.92, 0.92, 0.90);
    vec3 c4 = vec3(0.847, 0.855, 0.922), c5 = vec3(0.592, 0.554, 0.745), c6 = vec3(0.329, 0.153, 0.533);
    if (t < 0.166) return mix(c0, c1, t * 6.0);
    if (t < 0.333) return mix(c1, c2, (t - 0.166) * 6.0);
    if (t < 0.5) return mix(c2, c3, (t - 0.333) * 6.0);
    if (t < 0.666) return mix(c3, c4, (t - 0.5) * 6.0);
    if (t < 0.833) return mix(c4, c5, (t - 0.666) * 6.0);
    return mix(c5, c6, (t - 0.833) * 6.0);
}

// =============================================================================
// 【 核心重构：高度复用的全局 13 套标准科学色谱映射函数库 】 [1.2.7]
// =============================================================================
vec3 getColormap(int style, float t, vec3 midCol) {
    // Diverging Scientific Wavefield Maps
    if (style == 0) return getSeismic(t);
    if (style == 1) return getCoolwarm(t);
    if (style == 2) return getPuOr(t);
    if (style == 3) return getGrayscale(t);
    // Sequential Energy Maps
    if (style == 4) return getInferno(t);
    if (style == 5) return getPlasma(t);
    if (style == 6) return getViridis(t);
    if (style == 7) return getTurbo(t);
    if (style == 8) { // Magma Glow (经典熔岩发光)
        vec3 c0 = midCol, c1 = vec3(0.25, 0.05, 0.35), c2 = vec3(0.70, 0.15, 0.40), c3 = vec3(0.95, 0.50, 0.15), c4 = vec3(1.0, 1.0, 1.0);
        if (t < 0.25) return mix(c0, c1, t * 4.0);
        if (t < 0.5) return mix(c1, c2, (t - 0.25) * 4.0);
        if (t < 0.75) return mix(c2, c3, (t - 0.5) * 4.0);
        return mix(c3, c4, (t - 0.75) * 4.0);
    }
    // Legacy / Experimental
    if (style == 9) return getJet(t);
    if (style == 10) return getHot(t);
    if (style == 11) return getCold(t);
    // Specialized
    if (style == 12) return getTerrain(t);
    return midCol;
}

// =============================================================================
// 【第三级：物性取值与地层解析函数】
// =============================================================================
float calcVp(vec2 uv) {
    if (useModelTexture) return texture(modelTexture, uv).r * 6000.0; // 映射至 0~6000 m/s
    return uniformVp;
}

float calcRHO(vec2 uv) {
    if (useModelTexture) return texture(modelTexture, uv).b * 3000.0; // 映射至 0~3000 kg/m3
    return uniformRho;
}

// =============================================================================
// 【 核心重构：高度集成的地质背景生成函数 】
// =============================================================================
vec3 getGeologicalBackground(vec2 uv, vec2 cellPos) {
    // Velocity -> normalized scalar
    float vp = calcVp(uv);
    float t = clamp((vp - 500.0) / 5500.0, 0.0, 1.0);

    // Default Background
    vec3 bgCol = vec3(0.012, 0.015, 0.018);

    // Style 0: Dark Titanium
    if (modelStyle == 0) {
        bgCol = vec3(0.012, 0.015, 0.018);
    }
    // Style 1: Geological Map
    else if (modelStyle == 1) {
        vec3 cWater = vec3(0.0, 0.22, 0.45), cSed = vec3(0.28, 0.55, 0.42), cRock = vec3(0.75, 0.65, 0.28), cHard = vec3(0.68, 0.18, 0.12);
        if (t < 0.333) {
            bgCol = mix(cWater, cSed, t * 3.0);
        } else if (t < 0.666) {
            bgCol = mix(cSed, cRock, (t - 0.333) * 3.0);
        } else {
            bgCol = mix(cRock, cHard, (t - 0.666) * 3.0);
        }
        // Structural edge enhancement
        if (useModelTexture) {
            vec2 texelSize = 1.0 / simSize;
            float vp_right = calcVp(uv + vec2(texelSize.x, 0.0));
            float vp_up = calcVp(uv + vec2(0.0, texelSize.y));
            float diff = abs(vp - vp_right) + abs(vp - vp_up);
            float edge = smoothstep(10.0, 50.0, diff);
            bgCol = mix(bgCol, vec3(0.05, 0.05, 0.08), edge * 0.35);
        }
    }
    // Style 2: Grayscale Velocity
    else if (modelStyle == 2) {
        bgCol = vec3(mix(0.18, 0.82, t));
    }
    // Style 3: Sync With Active Wave Colormap
    else if (modelStyle == 3) {
        bgCol = getColormap(waveStyle, t, vec3(0.015, 0.018, 0.022));
        // Background desaturation 避免背景和波场竞争
        bgCol = mix(vec3(dot(bgCol, vec3(0.299, 0.587, 0.114))), bgCol, 0.55);
        // Dark scientific attenuation
        bgCol *= 0.38;
    }
    // Style 4: Pure White Scientific Background (Soft scientific white, 避免纯 1.0 白导致刺眼)
    else if (modelStyle == 4) {
        bgCol = mix(vec3(0.965), vec3(0.90), t * 0.35);
        // Very soft structural variation
        if (useModelTexture) {
            vec2 texelSize = 1.0 / simSize;
            float vp_right = calcVp(uv + vec2(texelSize.x, 0.0));
            float vp_up = calcVp(uv + vec2(0.0, texelSize.y));
            float diff = abs(vp - vp_right) + abs(vp - vp_up);
            float edge = smoothstep(10.0, 50.0, diff);
            bgCol = mix(bgCol, vec3(0.82, 0.84, 0.88), edge * 0.12);
        }
    }

    // PML Darkening
    if (npml > 0.0) {
        float distToEdge = min(min(cellPos.x, simSize.x - cellPos.x), min(cellPos.y, simSize.y - cellPos.y));
        if (distToEdge < npml) {
            float stripe = step(0.5, fract((cellPos.x + cellPos.y) * 0.15));
            float fade = smoothstep(0.0, npml, npml - distToEdge);
            if (modelStyle == 4) { // 白背景需要单独处理
                bgCol = mix(bgCol, vec3(0.82, 0.84, 0.88), fade * (0.35 + 0.10 * stripe));
            } else {
                bgCol = mix(bgCol, bgCol * 0.15, fade * (0.25 + 0.15 * stripe));
            }
        }
    }
    return bgCol;
}

// =============================================================================
// 【第四级：主渲染 main() 函数，完全复用并精简，实现极致性能】
// =============================================================================
void main() {
    // 1. 垂直翻转纹理坐标 Y (地表居顶对齐)
    vec2 flippedTexCoords = vec2(TexCoords.x, 1.0 - TexCoords.y);

    // 2. 采样地震波场颜色
    vec4 texColor = texture(seisTexture, flippedTexCoords);

    // 3. 将纹理坐标映射到实际网格单位
    vec2 cellPos = flippedTexCoords * simSize;

    // 4. 计算自适应地质背景底色
    vec3 baseBG = getGeologicalBackground(flippedTexCoords, cellPos);

    // 5. 逆向物理重构：解构振幅值 v
    float v = (texColor.r > texColor.b) ? (1.0 - texColor.g) : -(1.0 - texColor.g);

    // 偏导数重构 3D 凹凸表面法线 (仅用于保留着色，彻底舍弃高能耗 specular 光照)
    float dx = dFdx(v) * 12.0, dy = dFdy(v) * 12.0;
    vec3 normal = normalize(vec3(-dx, -dy, 1.0));
    float gradLen = length(vec2(dx, dy));

    // =============================================================================
    // 6. 多重渲染风格计算 (重构：统一的双极性对称映射)
    // =============================================================================
    vec3 waveColor;
    float waveIntensity;
    bool isDiverging = waveStyle == 0 || waveStyle == 1 || waveStyle == 2 || waveStyle == 3 || waveStyle == 7;

    if (isDiverging) { // Signed Wavefield Rendering
        float signedV = tanh(v * 2.5);
        float t = clamp(signedV * 0.5 + 0.5, 0.0, 1.0);
        waveColor = getColormap(waveStyle, t, baseBG);
        waveIntensity = pow(smoothstep(0.001, 0.035, abs(v)), 0.75); // Diverging: 保留低振幅相位连续性
    } else { // Energy Rendering
        float energy = tanh(abs(v) * 2.5);
        waveColor = getColormap(waveStyle, energy, baseBG);
        waveIntensity = pow(smoothstep(0.01, 0.12, abs(v)), 0.95); // Sequential: 更强调主能量
    }

    if (isDiverging) { // Diverging Signed Wavefield (原逻辑保留)
        float signedV = tanh(v * 2.5);
        float t = clamp(signedV * 0.5 + 0.5, 0.0, 1.0);
        waveColor = getColormap(waveStyle, t, baseBG);
        waveIntensity = smoothstep(0.01, 0.1, abs(v));
    } else { // Sequential Energy Colormaps (原逻辑保留)
        float energy = tanh(abs(v) * 2.5);
        waveColor = getColormap(waveStyle, energy, baseBG);
        waveIntensity = smoothstep(0.01, 0.1, abs(v));
    }

    // 最终混合：无波场区域完全保留高保真地层地质图，有波场区域叠加标准波谱
    vec3 finalColor = mix(baseBG, waveColor, waveIntensity);
    float pxToGridUnits = max(fwidth(cellPos.x), fwidth(cellPos.y));

    // 7. 【物理双重网格绘制】
    // Adaptive Scientific Grid Rendering
    // =============================================================================
    // 【重构后的网格自适应绘制逻辑】
    // =============================================================================
    if (showGrid) {
        // 基于传入的 gridSpacing 进行 LOD 缩放计算
        float baseSpacing = max(5.0, gridSpacing);
        float lod = pow(2.0, floor(log2(max(pxToGridUnits * (50.0 / baseSpacing), 1e-5))));
        
        float mainInterval = baseSpacing * max(1.0, lod);
        float subInterval = mainInterval * 0.2; // 细分网格自适应为 5 等分，不再死锁在 1.0

        // Colors
        vec3 subGridCol = vec3(0.72), mainGridCol = vec3(1.0), dotCol = vec3(0.5, 0.95, 0.5);

        // Sub Grid
        float subLineX = getGridLine(cellPos.x, subInterval, 0.7);
        float subLineY = getGridLine(cellPos.y, subInterval, 0.7);
        float gridSub = max(subLineX, subLineY);
        float opacitySub = 0.018 * gridOpacity * (1.0 - smoothstep(0.4, 1.5, pxToGridUnits));

        // Main Grid
        float mainLineX = getGridLine(cellPos.x, mainInterval, 1.0);
        float mainLineY = getGridLine(cellPos.y, mainInterval, 1.0);
        float gridMain = max(mainLineX, mainLineY);
        float opacityMain = 0.055 * gridOpacity * (1.0 - smoothstep(4.0, 15.0, pxToGridUnits));

        // Grid Dots
        vec2 mainCoord = cellPos / mainInterval;
        vec2 nearestIntersection = round(mainCoord);
        vec2 offsetToIntersection = (mainCoord - nearestIntersection) * mainInterval;
        float dist2 = dot(offsetToIntersection, offsetToIntersection);
        float dist2Pixels = dist2 / max(pxToGridUnits * pxToGridUnits, 1e-6);
        float dotRadius = 2.5, dotAA = 2.0;
        float gridDot = 1.0 - smoothstep(dotRadius * dotRadius, (dotRadius + dotAA) * (dotRadius + dotAA), dist2Pixels);
        float opacityDot = 0.16 * gridOpacity * (1.0 - smoothstep(4.0, 15.0, pxToGridUnits));

        // Additive Blend
        finalColor += subGridCol * gridSub * opacitySub;
        finalColor += mainGridCol * gridMain * opacityMain;
        finalColor += dotCol * gridDot * opacityDot;
        finalColor = clamp(finalColor, 0.0, 1.0);
    }

    // 8. 绘制 PML 吸收边界荧光线 (采用最外层 1D 线性距离算法，彻底解决缩放闪烁)
    if (npml > 0.0) {
        float dPmlLeft = abs(cellPos.x - npml), dPmlRight = abs(cellPos.x - (simSize.x - npml));
        float dPmlBottom = abs(cellPos.y - (simSize.y - npml)), dPmlTop = abs(cellPos.y - npml);
        float distToPml = 1e6;
        if (cellPos.x >= (npml - 0.1) && cellPos.x <= simSize.x - (npml - 0.1)) {
            distToPml = min(distToPml, dPmlBottom);
            distToPml = min(distToPml, dPmlTop);
        }
        if (cellPos.y >= (npml - 0.1) && cellPos.y <= simSize.y - (npml - 0.1)) {
            distToPml = min(distToPml, dPmlLeft);
            distToPml = min(distToPml, dPmlRight);
        }
        float pmlLineThickness = 1.5, pmlAA = fwidth(distToPml);
        float pmlBorder = smoothstep(pmlAA * pmlLineThickness, 0.0, distToPml - 0.05);
        vec3 pmlLineColor = vec3(0.5, 1.0, 0.5); 
        float topFade = smoothstep(0.0, npml + 2.0, cellPos.y);
        finalColor = mix(finalColor, pmlLineColor, pmlBorder * 0.65 * topFade);
    }

    // 9. 绘制最外层物理边界框 (白框)
    float dOuterLeft = abs(cellPos.x - 1.0), dOuterRight = abs(cellPos.x - (simSize.x - 1.0));
    float dOuterBottom = abs(cellPos.y - 1.0), dOuterTop = abs(cellPos.y - (simSize.y - 1.0));
    float distToOuter = 1e6;
    if (cellPos.x >= 0.9 && cellPos.x <= simSize.x - 0.9) {
        distToOuter = min(distToOuter, dOuterBottom);
        distToOuter = min(distToOuter, dOuterTop);
    }
    if (cellPos.y >= 0.9 && cellPos.y <= simSize.y - 0.9) {
        distToOuter = min(distToOuter, dOuterLeft);
        distToOuter = min(distToOuter, dOuterRight);
    }
    float outerAA = fwidth(distToOuter);
    float outerBorder = smoothstep(outerAA * 1.5, 0.0, distToOuter - 0.05);
    finalColor = mix(finalColor, vec3(1.0, 1.0, 1.0), outerBorder * 0.4); 

    FragColor = vec4(finalColor, 0.8); 
}