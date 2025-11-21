// --- UNIFORMS ---
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

@group(1) @binding(0) var t_diffuse: texture_2d<f32>;
@group(1) @binding(1) var s_diffuse: sampler;

// --- UTILS ---
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
    var val = 0.0;
    var amp = 0.5;
    var pos = p;
    val += noise(pos) * amp; pos *= 2.02; amp *= 0.5;
    val += noise(pos) * amp;
    return val;
}

fn dither(frag_coord: vec2<f32>) -> f32 {
    let magic = vec3<f32>(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(frag_coord, magic.xy)));
}

// --- WEATHER ---
fn get_regional_weather(world_xz: vec2<f32>) -> vec2<f32> {
    let scale = 0.0003;
    let scroll = vec2<f32>(camera.time * 2.0, 0.0);
    let sample_pos = (world_xz + scroll) * scale;

    var w = noise2d(sample_pos);
    w += noise2d(sample_pos * 2.0) * 0.5;
    w = w / 1.5;

    w = clamp(w + camera.weather_offset, 0.0, 1.0);

    let rain_amt = smoothstep(0.6, 0.8, w);
    return vec2<f32>(w, rain_amt);
}

// --- SKYBOX LOGIC ---
fn get_sun_pos() -> vec3<f32> {
    let angle = camera.day_progress * 6.28318;
    return normalize(vec3<f32>(sin(angle), cos(angle) * 0.8, -0.4));
}

fn get_moon_pos() -> vec3<f32> { return -get_sun_pos(); }

fn get_environment_light() -> mat3x3<f32> {
    let sun_dir = get_sun_pos();
    let sun_h = sun_dir.y;

    var dir = sun_dir;
    var color = vec3<f32>(0.0);
    var ambient = vec3<f32>(0.0);

    if (sun_h > -0.2) {
        dir = sun_dir;
        if (sun_h > 0.1) {
            color = vec3<f32>(1.0, 0.98, 0.9);
            ambient = vec3<f32>(0.6, 0.8, 1.0);
        } else {
            let t = (sun_h + 0.2) / 0.3;
            color = mix(vec3<f32>(1.0, 0.3, 0.0), vec3<f32>(1.0, 0.9, 0.7), t);
            ambient = mix(vec3<f32>(0.2, 0.1, 0.3), vec3<f32>(0.6, 0.7, 0.9), t);
        }
        let fade = smoothstep(-0.2, 0.0, sun_h);
        color *= fade;
        ambient *= fade;
    } else {
        dir = get_moon_pos();
        let moon_h = dir.y;
        color = vec3<f32>(0.4, 0.5, 0.7) * 0.2;
        ambient = vec3<f32>(0.02, 0.02, 0.05);
        let fade = smoothstep(-0.2, 0.0, moon_h);
        color *= fade;
    }

    let storm_factor = clamp(camera.weather_offset + 0.2, 0.0, 1.0);
    color = mix(color, color * 0.1, storm_factor);
    ambient = mix(ambient, ambient * 0.3, storm_factor);

    return mat3x3<f32>(dir, color, ambient);
}

// UPDATED: Returns only the atmospheric gradient (No Stars)
fn get_sky_color(view_dir: vec3<f32>, sun_dir: vec3<f32>, weather: f32) -> vec3<f32> {
    let sun_h = sun_dir.y;

    let day_z = vec3<f32>(0.1, 0.4, 0.9); let day_h = vec3<f32>(0.6, 0.8, 1.0);
    let set_z = vec3<f32>(0.2, 0.1, 0.4); let set_h = vec3<f32>(0.9, 0.5, 0.1);
    let night_z = vec3<f32>(0.01, 0.01, 0.04); let night_h = vec3<f32>(0.02, 0.03, 0.08);

    var sky_z = day_z;
    var sky_h = day_h;

    if (sun_h < 0.2 && sun_h > -0.2) {
        let t = smoothstep(-0.2, 0.2, sun_h);
        sky_z = mix(night_z, mix(set_z, day_z, t), t);
        sky_h = mix(night_h, mix(set_h, day_h, t), t);
    } else if (sun_h <= -0.2) {
        sky_z = night_z;
        sky_h = night_h;
    }

    let storm_dark = mix(1.0, 0.15, weather);
    sky_z *= storm_dark;
    sky_h *= storm_dark;

    let horizon = pow(1.0 - max(view_dir.y, 0.0), 3.0);
    return mix(sky_z, sky_h, horizon);
}

// NEW: Calculates just the stars
fn get_stars(view_dir: vec3<f32>, sun_h: f32, weather: f32) -> vec3<f32> {
    if (sun_h < 0.1 && weather < 0.8) {
        let p = view_dir * 150.0;
        let star_noise = hash(floor(p));
        let star_vis = smoothstep(0.997, 1.0, star_noise);
        let star_int = (1.0 - smoothstep(-0.1, 0.1, sun_h)) * (1.0 - weather);
        return vec3<f32>(star_vis * star_int);
    }
    return vec3<f32>(0.0);
}

// --- CLOUD HELPERS ---
fn intersect_slab(ro: vec3<f32>, rd: vec3<f32>, y_min: f32, y_max: f32) -> vec2<f32> {
    let inv_dir_y = 1.0 / (rd.y + 0.00001);
    let t0 = (y_min - ro.y) * inv_dir_y;
    let t1 = (y_max - ro.y) * inv_dir_y;
    return vec2<f32>(min(t0, t1), max(t0, t1));
}

fn get_cloud_density(p: vec3<f32>, local_weather: f32) -> f32 {
    let c_type = camera.cloud_type;
    let wind = vec3<f32>(camera.wind.x, 0.0, camera.wind.y) * camera.time * 10.0;
    let pos_a = (p + wind) * 0.015;
    let noise_a = fbm_fast(pos_a);
    let pos_b = (p + wind) * vec3<f32>(0.005, 0.02, 0.005);
    let noise_b = fbm_fast(pos_b);
    let n = mix(noise_a, noise_b, c_type);
    let coverage = mix(0.65, 0.35, local_weather);
    return smoothstep(coverage, coverage + 0.25, n);
}

fn get_light_transmittance(pos: vec3<f32>, sun_dir: vec3<f32>, weather: f32) -> f32 {
    let steps = 3;
    let step_size = 15.0;
    var density = 0.0;
    for (var i = 0; i < steps; i++) {
        let sample_pos = pos + sun_dir * (f32(i) * step_size);
        density += get_cloud_density(sample_pos, weather);
    }
    return exp(-density * 1.2);
}

// --- MAIN PIPELINE ---
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

    // FOG CALCULATION
    let dist = distance(in.world_pos, camera.camera_pos);
    let fog_factor = 1.0 - exp(-dist * 0.002);
    let view_dir = normalize(in.world_pos - camera.camera_pos);
    let w_data = get_regional_weather(in.world_pos.xz);

    // FIX: Use gradient only (no stars) for fog
    let fog_color = get_sky_color(view_dir, get_sun_pos(), w_data.x);
    let result = mix(final_color, fog_color, fog_factor);

    return vec4<f32>(result, 1.0);
}

// --- SKYBOX PIPELINE ---
struct SkyboxVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_skybox(@builtin(vertex_index) v_idx: u32) -> SkyboxVertexOutput {
    var out: SkyboxVertexOutput;
    var uvs = array<vec2<f32>, 3>(vec2<f32>(0.0, 0.0), vec2<f32>(2.0, 0.0), vec2<f32>(0.0, 2.0));
    let uv = uvs[v_idx];
    out.uv = uv;
    out.clip_position = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 1.0, 1.0);
    return out;
}

@fragment
fn fs_skybox(in: SkyboxVertexOutput) -> @location(0) vec4<f32> {
    let ndc = vec4<f32>(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0, 1.0, 1.0);
    let world_pos_hom = camera.inv_view_proj * ndc;
    let view_dir = normalize(world_pos_hom.xyz / world_pos_hom.w - camera.camera_pos);
    let rnd = dither(in.clip_position.xy);

    let sun_pos = get_sun_pos();
    let env = get_environment_light();
    let look_target = camera.camera_pos.xz + view_dir.xz * 1000.0;
    let w_data = get_regional_weather(look_target);
    let sky_weather = w_data.x;

    // 1. Base Sky Gradient
    var color = get_sky_color(view_dir, sun_pos, sky_weather);

    // 2. Add Stars (Before Clouds, so clouds cover them)
    color += get_stars(view_dir, sun_pos.y, sky_weather);

    // 3. Lightning Ambient
    let lightning_flash = camera.lightning_color * camera.lightning_intensity * 0.3;
    color += lightning_flash * 0.1;

    // 4. Cloud Raymarching
    let plane_noise = noise(camera.camera_pos * 0.001 + view_dir * 10.0) * 30.0;
    let c_bottom = mix(140.0, 90.0, sky_weather) + plane_noise * 0.5;
    let c_top    = mix(170.0, 250.0, sky_weather) + plane_noise;

    if (view_dir.y > -0.1 || camera.camera_pos.y > 100.0) {
        let hit = intersect_slab(camera.camera_pos, view_dir, c_bottom, c_top);
        let t_min = hit.x;
        let t_max = hit.y;

        if (t_max > 0.0 && t_max > t_min) {
            let t_start = max(0.0, t_min);
            if (t_start < 4000.0) {
                let steps = 40;
                let march_dist = min(t_max, 4000.0) - t_start;
                let step_size = march_dist / f32(steps);
                var t = t_start + step_size * rnd;
                var total_trans = 1.0;
                var acc_color = vec3<f32>(0.0);
                let den_scale = mix(0.02, 0.15, sky_weather);
                let cos_theta = dot(view_dir, env[0]);
                let phase = 0.6 + 0.4 * pow(0.5 * (cos_theta + 1.0), 8.0);
                let ambient_base = env[2] + lightning_flash * 0.2;

                for (var i = 0; i < steps; i++) {
                    if (total_trans < 0.01) { break; }
                    let pos = camera.camera_pos + view_dir * t;
                    let local_w = get_regional_weather(pos.xz).x;
                    let h = (pos.y - c_bottom) / (c_top - c_bottom);
                    let h_fade = smoothstep(0.0, 0.2, h) * smoothstep(1.0, 0.8, h);
                    let d = get_cloud_density(pos, local_w);
                    let loc_den = d * h_fade;

                    if (loc_den > 0.001) {
                        let step_od = loc_den * step_size * den_scale;
                        let step_trans = exp(-step_od);
                        let step_op = 1.0 - step_trans;
                        let sun_shadow = get_light_transmittance(pos, env[0], local_w);
                        let powder = 1.0 - exp(-loc_den * 2.0);
                        let direct = env[1] * 2.0 * sun_shadow * phase * (0.5 + powder);
                        let l_vec = camera.lightning_pos - pos;
                        let l_att = 1.0 / (1.0 + dot(l_vec, l_vec) * 0.00003);
                        let point_light = camera.lightning_color * camera.lightning_intensity * 150.0 * l_att;
                        let amb_grad = mix(mix(0.6, 0.02, local_w), 1.0, h);
                        let light_res = mix(ambient_base * amb_grad, direct, 0.7) + point_light;

                        acc_color += light_res * step_op * total_trans;
                        total_trans *= step_trans;
                    }
                    t += step_size;
                }
                // 5. Blend Sky/Stars with Clouds
                // Stars are already in 'color', so multiplying by total_trans hides them behind clouds
                let fog = 1.0 - exp(-t_start * 0.0003);
                color = color * total_trans + mix(acc_color, color, fog);
            }
        }
    }

    // 6. Sun/Moon Disks (Draw on top of clouds? Optional. usually yes for bloom, no for realism)
    if (env[0].y > 0.0) {
        let sun_d = dot(view_dir, env[0]);
        color += env[1] * smoothstep(0.9995, 0.9998, sun_d) * 5.0;
    } else {
        let moon_dir = get_moon_pos();
        color += vec3<f32>(0.8, 0.9, 1.0) * smoothstep(0.9985, 0.999, dot(view_dir, moon_dir)) * 5.0;
    }

    // Rainbow
    if (w_data.y > 0.1 && sun_pos.y > 0.0) {
        let bow_angle = dot(view_dir, -sun_pos);
        let diff = bow_angle - 0.74;
        if (abs(diff) < 0.04) {
            let t = (diff / 0.04) * 0.5 + 0.5;
            let bow = vec3<f32>(
                smoothstep(0.4, 0.6, t) - smoothstep(0.8, 1.0, t),
                smoothstep(0.2, 0.4, t) - smoothstep(0.6, 0.8, t),
                smoothstep(0.0, 0.2, t) - smoothstep(0.4, 0.6, t)
            );
            color += bow * 0.3 * w_data.y * sun_pos.y;
        }
    }

    // Tonemap
    color = color / (color + vec3<f32>(1.0));
    return vec4<f32>(pow(color, vec3<f32>(1.0 / 2.2)), 1.0);
}
