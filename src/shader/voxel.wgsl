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
    var tex_color = textureSample(t_diffuse, s_diffuse, tiled_uv);

    if (tex_color.a < 0.5) { discard; }

    let w_data = get_regional_weather(in.world_pos.xz);

    // --- INTEGRATED WETNESS LOGIC ---
    let terrain_h = get_terrain_height(in.world_pos.xz);

    // Improved check: Block must be near the surface top to be wet.
    // If we are more than 3 blocks below the terrain surface, we assume we are in a cave.
    // This effectively masks out wetness for deep caves even if "get_terrain_height" returns the mountain top.
    let height_diff = terrain_h - in.world_pos.y;
    let is_surface_layer = height_diff < 4.0 && height_diff > -2.0;

    let rain_amt = select(0.0, camera.rain * w_data.y, is_surface_layer);

    let wetness_darkening = mix(1.0, 0.6, rain_amt * 0.8);
    tex_color = vec4<f32>(tex_color.rgb * wetness_darkening, tex_color.a);

    // --- Reflections ---
    let view_dir = normalize(in.world_pos - camera.camera_pos);
    let reflect_dir = reflect(view_dir, normalize(in.normal));

    let sun_pos = get_sun_pos();
    let sky_ref = get_sky_color(reflect_dir, sun_pos, w_data.x);

    let fresnel = pow(1.0 - max(dot(-view_dir, in.normal), 0.0), 3.0);
    let puddle_mask = max(in.normal.y, 0.0);
    let reflect_strength = rain_amt * (0.3 + 0.5 * fresnel) * puddle_mask;

    tex_color = vec4<f32>(mix(tex_color.rgb, sky_ref, reflect_strength), tex_color.a);

    // Splash Overlay (Puddles)
    if (in.normal.y > 0.9 && rain_amt > 0.1) {
        let splash_speed = camera.time * 15.0;
        let splash_uv = in.world_pos.xz * 4.0;
        let n = noise(vec3<f32>(splash_uv.x, splash_uv.y, splash_speed));
        if (n > 0.85) {
            tex_color = mix(tex_color, vec4<f32>(0.8, 0.9, 1.0, 1.0), 0.4);
        }
    }

    let env = get_environment_light();
    let diffuse = max(dot(in.normal, env[0]), 0.0);
    let l_vec = camera.lightning_pos - in.world_pos;
    let l_dist_sq = dot(l_vec, l_vec);
    let l_att = 1.0 / (1.0 + l_dist_sq * 0.00005);
    let lightning_light = camera.lightning_color * camera.lightning_intensity * 50.0 * l_att;

    let lighting = env[2] + (env[1] * diffuse * 1.2) + lightning_light;
    let final_color = tex_color.rgb * in.color.rgb * lighting;

    // Further reduced fog density
    let dist = distance(in.world_pos, camera.camera_pos);
    let fog_factor = 1.0 - exp(-(max(dist - 80.0, 0.0)) * 0.001);
    let fog_color = get_sky_color(view_dir, sun_pos, w_data.x);

    let color_rgb = mix(final_color, fog_color, fog_factor);

    let puddle_ref = max(in.normal.y, 0.0) * rain_amt * 0.9;
    let reflect_val = puddle_ref;

    return vec4<f32>(color_rgb, reflect_val);
}
