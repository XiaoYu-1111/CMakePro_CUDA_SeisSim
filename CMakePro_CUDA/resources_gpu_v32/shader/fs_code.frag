#version 430 core
out vec4 FragColor;
in vec2 TexCoords;

uniform sampler2D lifeTexture;
uniform vec3 cellColor;
uniform vec2 simSize;   // 模拟网格大小 (如 1920, 1080)
uniform float viewZoom; // 当前缩放倍数

void main() {
    float heat = texture(lifeTexture, TexCoords).r;
    
    // --- [1. 动态网格线计算] ---
    // 根据模拟坐标计算当前像素所在的网格位置
    vec2 gridPos = TexCoords * simSize;
    vec2 gridLine = fract(gridPos); // 取得 0.0-1.0 的小数部分
    
    // 只有在放大到一定程度时才显示网格，防止缩小视角时网格挤成一团
    float gridAlpha = smoothstep(0.5, 1.5, viewZoom * 0.1); 
    // 计算线条粗细 (反比于缩放，保持视觉厚度恒定)
    float thickness = 0.05 / viewZoom; 
    
    float isGrid = 0.0;
    if (gridLine.x < thickness || gridLine.y < thickness) {
        isGrid = 0.12 * gridAlpha; // 网格强度
    }

    // --- [2. 颜色混合逻辑] ---
    // 基础背景色：深青色/墨绿色 (对标 CPU 版本的底色)
    vec3 backgroundColor = vec3(0.02, 0.08, 0.07);
    
    // 细胞发光效果
    vec3 neonColor = mix(backgroundColor, cellColor, heat);
    if(heat > 0.1) {
        neonColor += cellColor * heat * 0.4; // 核心区域增亮
    }
    
    // 叠加网格线
    vec3 finalColor = neonColor + vec3(1.0, 1.0, 1.0) * isGrid;

    // 输出颜色，给一点 Alpha 通道方便和背景 DrawList 混合
    FragColor = vec4(finalColor, 1.0);
}