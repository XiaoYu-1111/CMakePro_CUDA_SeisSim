#version 430 core

// 输入：顶点位置 (x, y, z) 和 纹理坐标 (u, v)
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoords;

// 输出：传给片段着色器的纹理坐标
out vec2 TexCoords;

void main() {
    // 将顶点位置直接输出到裁剪空间 (NDC)
    // 对于全屏矩形，aPos 的范围通常是 (-1, -1) 到 (1, 1)
    gl_Position = vec4(aPos, 1.0);
    
    // 传递纹理坐标
    TexCoords = aTexCoords;
}