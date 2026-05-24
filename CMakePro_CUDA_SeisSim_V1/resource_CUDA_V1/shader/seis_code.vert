#version 430 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoords;

out vec2 TexCoords;

uniform vec2 viewOffset;
uniform float viewZoom;
uniform vec2 winSize;
uniform vec2 simSize;

void main() {
    TexCoords = aTexCoords;
    
    // 1. 计算窗口与模拟网格的宽高比，防止图像被拉伸变形
    float winAspect = winSize.x / winSize.y;
    float simAspect = simSize.x / simSize.y;
    vec2 aspectCorr = vec2(1.0, 1.0);
    if (winAspect > simAspect) {
        aspectCorr.x = winAspect / simAspect;
    } else {
        aspectCorr.y = simAspect / winAspect;
    }

    // 2. 应用视口平移与缩放 (与 C++ 鼠标交互完全匹配)
    vec2 pos = (aPos - viewOffset) * viewZoom / aspectCorr;
    gl_Position = vec4(pos, 0.0, 1.0);
}