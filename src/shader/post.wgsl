struct Out { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }
@group(0) @binding(0) var t_scene: texture_2d<f32>;
@group(0) @binding(1) var s_scene: sampler;

@vertex fn vs_main(@builtin(vertex_index) idx: u32) -> Out {
    var out: Out;
    var uvs = array<vec2<f32>,3>(vec2<f32>(0., 2.), vec2<f32>(0., 0.), vec2<f32>(2., 0.));
    out.uv = uvs[idx];
    out.pos = vec4<f32>(out.uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv.y = 1.0 - out.uv.y;
    return out;
}

@fragment fn fs_main(in: Out) -> @location(0) vec4<f32> {
    let col = textureSample(t_scene, s_scene, in.uv);
    let mapped = col.rgb / (col.rgb + vec3<f32>(1.0));
    let sharp = mapped * 1.1 + 0.05;
    return vec4<f32>(sharp, 1.0);
}
