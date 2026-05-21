#version 430 core
out vec4 FragColor;
in vec2 TexCoords;

uniform float u_time;
uniform vec2 u_resolution;
uniform vec2 u_center;
uniform float u_zoom;
uniform int u_iters;
uniform bool u_enableScanlines;
uniform int u_styleMode; 

uniform int u_fractalType; // 0: Mandelbrot, 1: Julia, 2: Burning Ship, 3: Mandelbar
uniform vec2 u_juliaSeed;  // Julia 集合需要的种子点 (例如 -0.8, 0.156)

// --- 复数除法 (修复平滑法线的核心) ---
vec2 complex_div(vec2 a, vec2 b) {
    float denom = dot(b, b);
    return vec2(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / denom;
}

// --- ACES 电影级色调映射 ---
vec3 ACESFilm(vec3 x) {
    float a = 2.51f; float b = 0.03f; float c = 2.43f; float d = 0.59f; float e = 0.14f;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}

void main() {
    float aspect = u_resolution.x / u_resolution.y;
    vec2 uv = (TexCoords - 0.5) * vec2(aspect, 1.0);
    vec2 c = uv * u_zoom + u_center;

    // --- 动态参数计算：缩放深度驱动器 ---
    // z_level 从 0 (初始) 增加到约 25 (极深)
    float z_level = -log2(u_zoom); 
    float time_flow = u_time * 0.3;

    // --- [新增逻辑] 专门为 Burning Ship 反转 Y 轴 ---
    // u_fractalType 为 2 代表 Burning Ship
    if (u_fractalType == 2) {
        // 反转 c 的 Y 分量，让船看起来是正着的
        c.y = -c.y; 
    }
    // --- 分型迭代 ---
    vec2 z = vec2(0.0);

    // Julia 模式下，z 随像素变，c 是固定种子
    if(u_fractalType == 1) {
        z = c; 
        c = u_juliaSeed;
    }

    vec2 dz = vec2(0.0);
    float iter = 0.0;
    const float BAILOUT = 1e8;

    for(int i = 0; i < u_iters; i++) {
        // --- 核心算法分支 ---
        if(u_fractalType == 2) { 
            // Burning Ship 特有的绝对值逻辑
            z = abs(z); 
        }
        
        // 计算导数 (DE)
        if(u_fractalType == 3) {
            // Mandelbar 的导数：dz = 2 * conj(z) * conj(dz) + 1
            dz = 2.0 * vec2(z.x * dz.x + z.y * dz.y, z.x * dz.y - z.y * dz.x) + vec2(1.0, 0.0);
            // Mandelbar 公式：z = conj(z)^2 + c
            z = vec2(z.x * z.x - z.y * z.y, -2.0 * z.x * z.y) + c;
        } else {
            // 标准 Mandelbrot/Julia/Burning Ship 的导数
            dz = 2.0 * vec2(z.x * dz.x - z.y * dz.y, z.x * dz.y + z.y * dz.x) + vec2(1.0, 0.0);
            // 标准公式：z = z^2 + c
            z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        }

        if(dot(z, z) > BAILOUT) break;
        iter += 1.0;
    }

    vec3 col = vec3(0.0);
    float pixel_size = u_zoom / u_resolution.y;

    if(iter >= float(u_iters)) {
        // 内部区域：随深度加深的黑曜石核心
        // --- 修改后 ---
        // 基础底色提亮，并随深度(z_level)产生微弱的色彩起伏
        vec3 abyss_base = (u_styleMode == 0) ? vec3(0.04, 0.05, 0.07) : vec3(0.06, 0.04, 0.08);
        col = abyss_base * (1.0 + z_level * 0.1); 
        // 加入微弱的金属拉丝感纹理，让内部不那么死板
        col *= 1.0 + 0.1 * sin(z.x * 100.0 + z.y * 100.0);
    } 
    else {
        float r = length(z);
        float sn = iter - log2(log2(r)) + 4.0;
        
        // --- 1. 动态结构演变 (Machined Grooves Evolution) ---
        // 缩放越深，条纹频率随之产生相位漂移
        float band_freq = 4.0 + sin(z_level * 0.2) * 2.0; 
        float phase = fract(sn * band_freq - time_flow);
        float wave = smoothstep(0.0, 0.1, phase) * (1.0 - smoothstep(0.8, 1.0, phase));

        // --- 2. 动态 3D 法线与环境交互 ---
        vec2 v = complex_div(z, dz);
        // 深度越深，表面越显得锐利(bump减小)
        float bump = 0.015 / (1.0 + z_level * 0.1); 
        vec3 normal = normalize(vec3(v.x, v.y, bump)); 

        // --- 3. 动态光照模型 ---
        // 光源位置随深度和时间旋转，模拟无人机探测灯扫射
        float light_ang = time_flow + z_level * 0.5;
        vec3 light_main = normalize(vec3(cos(light_ang), sin(light_ang), 1.0)); 
        vec3 light_fill = normalize(vec3(-light_main.x, -light_main.y, 0.5));
        vec3 view_dir   = vec3(0.0, 0.0, 1.0);
        vec3 half_dir   = normalize(light_main + view_dir);

        float NdotL = clamp(dot(normal, light_main), 0.0, 1.0);
        float NdotV = clamp(dot(normal, view_dir), 0.0, 1.0);
        float spec  = pow(clamp(dot(normal, half_dir), 0.0, 1.0), 64.0 + z_level * 10.0);

        // --- 4. 材质动态变色 (Adaptive Color Shift) ---
        vec3 albedo, rim_col;
        
        // 计算动态色偏：深度越深，钛金属越呈现“烧蓝”或“电光金”
        vec3 shift_color = 0.5 + 0.5 * cos(vec3(0.0, 0.4, 0.8) + z_level * 0.3);
        
        if (u_styleMode == 0) {
            // 风格 0：航空钛 (随深度增加冷光质感)
            albedo = mix(vec3(0.2, 0.22, 0.25), vec3(0.1, 0.15, 0.3), clamp(z_level/20.0, 0.0, 1.0));
            rim_col = mix(vec3(0.0, 0.7, 1.0), vec3(0.5, 0.0, 1.0), clamp(z_level/25.0, 0.0, 1.0));
        } else {
            // 风格 1：极端环境钛 (随深度产生高温灼烧色)
            albedo = vec3(0.05);
            rim_col = shift_color * 1.5;
        }

        // 最终合成
        //col = albedo * (NdotL * 0.8 + 0.2); // 基础漫反射
        col = albedo * (NdotL * 0.6 + 0.4);
        col += vec3(1.0) * spec * (0.5 + z_level * 0.1); // 随深度增强的高光
        col += rim_col * pow(1.0 - NdotV, 4.0) * 0.8; // 边缘光

        // 距离估计 AO
        float de = 0.5 * log(r) * r / length(dz);
        float ao = smoothstep(0.0, pixel_size * (20.0 + z_level), de);
        // --- 修改后 ---
        // 即使在最深的沟壑里，也保留至少 25% 的亮度
        float softened_ao = mix(0.25, 1.0, ao); 
        float softened_wave = mix(0.3, 1.0, wave);
        col *= softened_ao * softened_wave;

        // 边界高能线条
        float edge_glow = smoothstep(pixel_size * 5.0, 0.0, de);
        col += rim_col * edge_glow * (1.0 + z_level * 0.2); 
    }

    // --- 后期处理：动态感增强 ---
    vec2 p = TexCoords - 0.5;
    // 缩放越深，暗角越重，模拟望远镜/显微镜感
    // --- 修改后 ---
// 降低系数 1.5 -> 0.8，并限制 z_level 的影响，防止深处全黑
float vignette = 1.1 - dot(p, p) * (0.8 + clamp(z_level * 0.02, 0.0, 0.5));
    col *= clamp(vignette, 0.0, 1.0);

    // 动态色差 (Chromatic Aberration)：随深度增加，画面边缘越模糊/溢色
    float ca = dot(p, p) * (0.01 + z_level * 0.002);
    vec3 out_col;
    out_col.r = col.r * (1.0 + ca);
    out_col.g = col.g;
    out_col.b = col.b * (1.0 - ca);

    // 扫描线
    if(u_enableScanlines) {
        out_col *= 0.96 + 0.04 * sin(gl_FragCoord.y * 1.5 + u_time);
    }

    // 胶片颗粒
    float noise = fract(sin(dot(TexCoords, vec2(12.9898, 78.233))) * 43758.5453);
    out_col += (noise - 0.5) * 0.03;

    FragColor = vec4(ACESFilm(out_col), 1.0);
}