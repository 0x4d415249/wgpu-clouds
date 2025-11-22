@group(1) @binding(0) var t_diffuse: texture_2d<f32>;
@group(1) @binding(1) var s_diffuse: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) bounds: vec4<f32>,
    @location(4) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) bounds: vec4<f32>,
    @location(2) color: vec4<f32>,
    @location(3) normal: vec3<f32>,
    @location(4) world_pos: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = vec4<f32>(in.position, 1.0);
    out.clip_position = camera.view_proj * world_pos;
    out.uv = in.uv;
    out.bounds = in.bounds;
    out.color = in.color;
    out.normal = in.normal;
    out.world_pos = in.position;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let min_uv = in.bounds.xy;
    let max_uv = in.bounds.zw;
    let tile_size = max_uv - min_uv;
    let tiled_uv = min_uv + fract(in.uv) * tile_size;
    let tex_color = textureSample(t_diffuse, s_diffuse, tiled_uv);

    if (tex_color.a < 0.5) { discard; }

    let env = get_environment_light();
    let diffuse = max(dot(in.normal, env[0]), 0.0);
    let l_vec = camera.lightning_pos - in.world_pos;
    let l_dist_sq = dot(l_vec, l_vec);
    let l_att = 1.0 / (1.0 + l_dist_sq * 0.00005);
    let lightning_light = camera.lightning_color * camera.lightning_intensity * 50.0 * l_att;

    let lighting = env[2] + (env[1] * diffuse) + lightning_light;
    let final_color = tex_color.rgb * in.color.rgb * lighting;

    let dist = distance(in.world_pos, camera.camera_pos);
    let fog_factor = 1.0 - exp(-dist * 0.002);
    let view_dir = normalize(in.world_pos - camera.camera_pos);
    let w_data = get_regional_weather(in.world_pos.xz);
    let fog_color = get_sky_color(view_dir, get_sun_pos(), w_data.x);
    let result = mix(final_color, fog_color, fog_factor);

    return vec4<f32>(result, 1.0);
}
