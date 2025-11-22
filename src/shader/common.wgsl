struct CameraUniform {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    time: f32,
    day_progress: f32,
    weather_offset: f32,
    cloud_type: f32,
    lightning_intensity: f32,
    lightning_pos: vec3<f32>,
    _pad1: f32,
    lightning_color: vec3<f32>,
    _pad2: f32,
    wind: vec2<f32>,
    rain: f32,
    _pad3: f32,
}
@group(0) @binding(0) var<uniform> camera: CameraUniform;

fn hash(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix( hash(i + vec3<f32>(0.0,0.0,0.0)),
                        hash(i + vec3<f32>(1.0,0.0,0.0)), u.x),
                   mix( hash(i + vec3<f32>(0.0,1.0,0.0)),
                        hash(i + vec3<f32>(1.0,1.0,0.0)), u.x), u.y),
               mix(mix( hash(i + vec3<f32>(0.0,0.0,1.0)),
                        hash(i + vec3<f32>(1.0,0.0,1.0)), u.x),
                   mix( hash(i + vec3<f32>(0.0,1.0,1.0)),
                        hash(i + vec3<f32>(1.0,1.0,1.0)), u.x), u.y), u.z);
}

fn noise2d(p: vec2<f32>) -> f32 {
    return noise(vec3<f32>(p.x, 0.0, p.y));
}

fn fbm_fast(p: vec3<f32>) -> f32 {
    var val = 0.0; var amp = 0.5; var pos = p;
    val += noise(pos) * amp; pos *= 2.02; amp *= 0.5;
    val += noise(pos) * amp;
    return val;
}

// Used by terrain gen
fn fbm(p: vec2<f32>) -> f32 {
    var v = 0.0; var a = 0.5;
    var pos = p;
    let rot = mat2x2<f32>(cos(0.5), sin(0.5), -sin(0.5), cos(0.5));
    for (var i = 0; i < 3; i++) {
        v += a * noise2d(pos);
        pos = rot * pos * 2.0 + vec2<f32>(100.0);
        a *= 0.5;
    }
    return v;
}

fn dither(frag_coord: vec2<f32>) -> f32 {
    let magic = vec3<f32>(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(frag_coord, magic.xy)));
}

// CENTRALIZED TERRAIN HEIGHT
// Used by: Shader Gen (Compute), Voxel (Wetness check), Rain (Occlusion)
fn get_terrain_height(p: vec2<f32>) -> f32 {
    let temp = noise2d(p * 0.001);
    let h_noise = fbm(p * 0.005);
    let mount = smoothstep(0.6, 0.8, temp);
    return mix(60.0 + h_noise * 20.0, 60.0 + h_noise * 80.0, mount);
}
