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
    lightning_color: vec3<f32>,
    wind: vec2<f32>,
    rain: f32,
}
@group(0) @binding(0) var<uniform> camera: CameraUniform;

@group(1) @binding(0) var t_diffuse: texture_2d<f32>;
@group(1) @binding(1) var s_diffuse: sampler;

// --- PARTICLE STORAGE (Restored) ---
struct Particle {
    pos: vec4<f32>, // xyz = position, w = scale
    vel: vec4<f32>, // xyz = velocity
}
@group(2) @binding(0) var<storage, read_write> particles: array<Particle>;

// --- VERTEX OUTPUTS ---
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct RainVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) alpha: f32,
}

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

fn dither(frag_coord: vec2<f32>) -> f32 {
    let magic = vec3<f32>(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(frag_coord, magic.xy)));
}

// --- WEATHER LOGIC ---
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

// --- COMPUTE SHADER: RAIN PHYSICS (Restored) ---
@compute @workgroup_size(64)
fn cs_rain(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= arrayLength(&particles)) { return; }

    var p = particles[idx];

    // Physics: Gravity + Wind
    let wind_force = vec3<f32>(camera.wind.x * 20.0, 0.0, camera.wind.y * 20.0);
    let fall_vel = p.vel.y; // Base falling speed stored in Y
    let velocity = vec3<f32>(wind_force.x, fall_vel, wind_force.z);

    let dt = 0.016;
    p.pos += vec4<f32>(velocity * dt, 0.0);

    // Wrapping Logic (Infinite Rain Field)
    let range_h = 60.0;
    let range_y = 40.0;
    let center = camera.camera_pos;
    let dist = p.pos.xyz - center;

    // Wrap X/Z
    if (dist.x > range_h) { p.pos.x -= range_h * 2.0; }
    if (dist.x < -range_h) { p.pos.x += range_h * 2.0; }
    if (dist.z > range_h) { p.pos.z -= range_h * 2.0; }
    if (dist.z < -range_h) { p.pos.z += range_h * 2.0; }

    // Wrap Y (Vertical)
    if (dist.y < -range_y) { p.pos.y += range_y * 2.0; }
    if (dist.y > range_y) { p.pos.y -= range_y * 2.0; }

    particles[idx] = p;
}

// --- VERTEX SHADER: RAIN RENDERING ---
@vertex
fn vs_rain(
    @builtin(vertex_index) v_idx: u32,
    @builtin(instance_index) i_idx: u32
) -> RainVertexOutput {
    var out: RainVertexOutput;

    // Read from Storage Buffer
    let p = particles[i_idx];
    let center = p.pos.xyz;
    let scale = p.pos.w;

    // 1. Weather & Masking Check
    let w_data = get_regional_weather(center.xz);
    let local_rain_intensity = w_data.y;

    // Cloud height check
    let c_bottom = mix(130.0, 90.0, w_data.x);
    let height_mask = 1.0 - smoothstep(c_bottom - 10.0, c_bottom, center.y);

    // 2. Billboard Geometry
    let wind_force = vec3<f32>(camera.wind.x * 20.0, 0.0, camera.wind.y * 20.0);
    let vel = vec3<f32>(wind_force.x, p.vel.y, wind_force.z);
    let dir = normalize(vel);

    let w = 0.03 * scale;
    let h = 0.8 * scale;

    var pos = vec3<f32>(0.0);
    var uv = vec2<f32>(0.0);

    // Align to velocity and view
    let view_vec = normalize(center - camera.camera_pos);
    let right = normalize(cross(view_vec, dir));

    if (v_idx == 0u) { pos = -right * w + dir * h; uv = vec2<f32>(0.0, 0.0); }
    if (v_idx == 1u) { pos =  right * w + dir * h; uv = vec2<f32>(1.0, 0.0); }
    if (v_idx == 2u) { pos = -right * w - dir * h; uv = vec2<f32>(0.0, 1.0); }
    if (v_idx == 3u) { pos =  right * w - dir * h; uv = vec2<f32>(1.0, 1.0); }

    let world_pos = center + pos;
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.uv = uv;

    // 3. Fade Logic
    let d = distance(center, camera.camera_pos);
    let dist_fade = 1.0 - smoothstep(40.0, 60.0, d);

    out.alpha = camera.rain * local_rain_intensity * height_mask * dist_fade;

    return out;
}

@fragment
fn fs_rain(in: RainVertexOutput) -> @location(0) vec4<f32> {
    if (in.alpha < 0.01) { discard; }
    let x_fade = 1.0 - abs(in.uv.x * 2.0 - 1.0);
    let y_fade = 1.0 - abs(in.uv.y * 2.0 - 1.0);

    // Lit by lightning
    let lightning = camera.lightning_color * camera.lightning_intensity * 0.5;
    let col = vec3<f32>(0.7, 0.8, 0.9) + lightning;

    return vec4<f32>(col, in.alpha * x_fade * y_fade * 0.6);
}

// --- MAIN SCENE ---

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    var uvs = array<vec2<f32>, 3>(vec2<f32>(0.0, 0.0), vec2<f32>(2.0, 0.0), vec2<f32>(0.0, 2.0));
    let uv = uvs[in_vertex_index];
    let pos = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    out.uv = uv;
    out.clip_position = vec4<f32>(pos, 0.0, 1.0);
    return out;
}

fn get_sun_pos() -> vec3<f32> {
    let angle = camera.day_progress * 6.28318;
    return normalize(vec3<f32>(sin(angle), cos(angle) * 0.8, -0.4));
}

fn get_sun_light_color(sun_y: f32) -> vec3<f32> {
    if (sun_y > 0.15) { return vec3<f32>(1.0, 0.98, 0.92); }
    if (sun_y > -0.15) {
        let t = (sun_y + 0.15) / 0.3;
        return mix(vec3<f32>(1.0, 0.4, 0.1), vec3<f32>(1.0, 0.98, 0.92), t);
    }
    return vec3<f32>(0.02, 0.02, 0.05);
}

fn get_moon_pos() -> vec3<f32> { return -get_sun_pos(); }

fn get_environment_light() -> mat3x3<f32> {
    let sun_dir = get_sun_pos();
    let sun_h = sun_dir.y;

    var dir = sun_dir;
    var color = vec3<f32>(0.0);
    var ambient = vec3<f32>(0.0);

    if (sun_h > -0.2) {
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
    let storm_factor = camera.weather_offset; // Approx
    color = mix(color, color * 0.1, storm_factor);
    ambient = mix(ambient, ambient * 0.2, storm_factor);

    return mat3x3<f32>(dir, color, ambient);
}

fn get_sky_color(view_dir: vec3<f32>, sun_dir: vec3<f32>, weather: f32) -> vec3<f32> {
    let sun_h = sun_dir.y;
    let day_z = vec3<f32>(0.1, 0.4, 0.9); let day_h = vec3<f32>(0.6, 0.8, 1.0);
    let set_z = vec3<f32>(0.2, 0.1, 0.4); let set_h = vec3<f32>(0.9, 0.5, 0.1);
    let night_z = vec3<f32>(0.01, 0.01, 0.04); let night_h = vec3<f32>(0.02, 0.03, 0.08);

    var sky_z = day_z; var sky_h = day_h;
    if (sun_h < 0.2 && sun_h > -0.2) {
        let t = smoothstep(-0.2, 0.2, sun_h);
        sky_z = mix(night_z, mix(set_z, day_z, t), t);
        sky_h = mix(night_h, mix(set_h, day_h, t), t);
    } else if (sun_h <= -0.2) { sky_z = night_z; sky_h = night_h; }

    let storm_dark = mix(1.0, 0.15, weather);
    sky_z *= storm_dark; sky_h *= storm_dark;

    let horizon = pow(1.0 - max(view_dir.y, 0.0), 3.0);
    var color = mix(sky_z, sky_h, horizon);

    if (sun_h < 0.1 && weather < 0.8) {
        let p = view_dir * 150.0;
        let star_noise = hash(floor(p));
        let star_vis = smoothstep(0.997, 1.0, star_noise);
        let star_int = (1.0 - smoothstep(-0.1, 0.1, sun_h)) * (1.0 - weather);
        color += vec3<f32>(star_vis * star_int);
    }
    return color;
}

fn intersect_slab(ro: vec3<f32>, rd: vec3<f32>, y_min: f32, y_max: f32) -> vec2<f32> {
    let inv_dir_y = 1.0 / (rd.y + 0.00001);
    let t0 = (y_min - ro.y) * inv_dir_y;
    let t1 = (y_max - ro.y) * inv_dir_y;
    return vec2<f32>(min(t0, t1), max(t0, t1));
}

fn fbm_fast(p: vec3<f32>) -> f32 {
    var val = 0.0; var amp = 0.5; var pos = p;
    val += noise(pos) * amp; pos *= 2.02; amp *= 0.5;
    val += noise(pos) * amp;
    return val;
}

fn get_cloud_density(p: vec3<f32>, local_weather: f32) -> f32 {
    let c_type = camera.cloud_type;
    let wind = vec3<f32>(camera.wind.x, 0.0, camera.wind.y) * camera.time * 10.0;
    let pos_a = (p + wind) * 0.015; let noise_a = fbm_fast(pos_a);
    let pos_b = (p + wind) * vec3<f32>(0.005, 0.02, 0.005); let noise_b = fbm_fast(pos_b);
    let n = mix(noise_a, noise_b, c_type);
    let coverage = mix(0.70, 0.35, local_weather);
    return smoothstep(coverage, coverage + 0.25, n);
}

fn get_light_transmittance(pos: vec3<f32>, sun_dir: vec3<f32>, weather: f32) -> f32 {
    let steps = 3; let step_size = 15.0; var density = 0.0;
    for (var i = 0; i < steps; i++) {
        let sample_pos = pos + sun_dir * (f32(i) * step_size);
        density += get_cloud_density(sample_pos, weather);
    }
    return exp(-density * 1.2);
}

@fragment
fn fs_clouds(in: VertexOutput) -> @location(0) vec4<f32> {
    let ndc = vec4<f32>(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0, 1.0, 1.0);
    let world_pos_hom = camera.inv_view_proj * ndc;
    let view_dir = normalize(world_pos_hom.xyz / world_pos_hom.w - camera.camera_pos);
    let rnd = dither(in.clip_position.xy);

    let sun_pos = get_sun_pos();
    let env = get_environment_light();
    let light_dir = env[0];
    let light_col = env[1];
    let ambient_col = env[2];

    let look_target = camera.camera_pos.xz + view_dir.xz * 1000.0;
    let w_data = get_regional_weather(look_target);
    let sky_weather = w_data.x;

    var color = get_sky_color(view_dir, sun_pos, sky_weather);

    let lightning_flash = camera.lightning_color * camera.lightning_intensity * 0.3;
    color += lightning_flash * 0.1;

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
                let cos_theta = dot(view_dir, light_dir);
                let phase = 0.6 + 0.4 * pow(0.5 * (cos_theta + 1.0), 8.0);
                let ambient_base = ambient_col + lightning_flash * 0.2;

                for (var i = 0; i < steps; i++) {
                    if (total_trans < 0.01) { break; }

                    let pos = camera.camera_pos + view_dir * t;
                    let local_w_data = get_regional_weather(pos.xz);
                    let local_w = local_w_data.x;

                    let h = (pos.y - c_bottom) / (c_top - c_bottom);
                    let h_fade = smoothstep(0.0, 0.2, h) * smoothstep(1.0, 0.8, h);

                    let d = get_cloud_density(pos, local_w);
                    let loc_den = d * h_fade;

                    if (loc_den > 0.001) {
                        let step_od = loc_den * step_size * den_scale;
                        let step_trans = exp(-step_od);
                        let step_op = 1.0 - step_trans;
                        let sun_shadow = get_light_transmittance(pos, light_dir, local_w);
                        let powder = 1.0 - exp(-loc_den * 2.0);
                        let direct = light_col * 2.0 * sun_shadow * phase * (0.5 + powder);
                        let l_vec = camera.lightning_pos - pos;
                        let l_dist_sq = dot(l_vec, l_vec);
                        let l_att = 1.0 / (1.0 + l_dist_sq * 0.00003);
                        let point_light = camera.lightning_color * camera.lightning_intensity * 150.0 * l_att;
                        let amb_min = mix(0.6, 0.02, local_w);
                        let amb_grad = mix(amb_min, 1.0, h);
                        let ambient = ambient_base * amb_grad;
                        let light_res = mix(ambient, direct, 0.7) + point_light;
                        acc_color += light_res * step_op * total_trans;
                        total_trans *= step_trans;
                    }
                    t += step_size;
                }
                let fog = 1.0 - exp(-t_start * 0.0003);
                color = color * total_trans + mix(acc_color, color, fog);
            }
        }
    }

    if (light_dir.y > 0.0) {
        let sun_d = dot(view_dir, light_dir);
        let sun_disk = smoothstep(0.9995, 0.9998, sun_d);
        color += light_col * sun_disk * 5.0;
    } else {
        let moon_dir = get_moon_pos();
        let moon_d = dot(view_dir, moon_dir);
        let moon_disk = smoothstep(0.9985, 0.999, moon_d);
        color += vec3<f32>(0.8, 0.9, 1.0) * moon_disk * 5.0;
    }

    if (w_data.y > 0.1 && sun_pos.y > 0.0) {
        let anti_sun = -sun_pos;
        let bow_angle = dot(view_dir, anti_sun);
        let diff = bow_angle - 0.74;
        if (abs(diff) < 0.04) {
            let t = (diff / 0.04) * 0.5 + 0.5;
            let r = smoothstep(0.4, 0.6, t) - smoothstep(0.8, 1.0, t);
            let g = smoothstep(0.2, 0.4, t) - smoothstep(0.6, 0.8, t);
            let b = smoothstep(0.0, 0.2, t) - smoothstep(0.4, 0.6, t);
            let bow = vec3<f32>(r, g, b);
            color += bow * 0.3 * w_data.y * sun_pos.y;
        }
    }

    color = color / (color + vec3<f32>(1.0));
    color = pow(color, vec3<f32>(1.0 / 2.2));
    let luma = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    color = mix(vec3<f32>(luma), color, 1.2);
    return vec4<f32>(color, 1.0);
}

@fragment
fn fs_blit(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_diffuse, s_diffuse, in.uv);
}
