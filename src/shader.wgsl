struct CameraUniform {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    time: f32,
    day_progress: f32,
    weather_offset: f32, // Replaces global 'weather'
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

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct RainVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) alpha: f32,
}

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 0.0), vec2<f32>(2.0, 0.0), vec2<f32>(0.0, 2.0)
    );
    let uv = uvs[in_vertex_index];
    let pos = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    out.uv = uv;
    out.clip_position = vec4<f32>(pos, 0.0, 1.0);
    return out;
}

// --- NOISE ---
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

// --- WEATHER MAP ---
fn get_regional_weather(world_xz: vec2<f32>) -> vec2<f32> {
    let scale = 0.0003;
    let scroll = vec2<f32>(camera.time * 2.0, 0.0); // Weather moves slowly
    let sample_pos = (world_xz + scroll) * scale;

    var w = noise2d(sample_pos);
    w += noise2d(sample_pos * 2.0) * 0.5;
    w = w / 1.5;

    // Apply offset (Controlled by player/logic)
    w = clamp(w + camera.weather_offset, 0.0, 1.0);

    // Rain threshold
    let rain_amt = smoothstep(0.6, 0.8, w);
    return vec2<f32>(w, rain_amt);
}

// --- RAIN PARTICLES ---
@vertex
fn vs_rain(@builtin(vertex_index) v_idx: u32, @location(0) i_offset: vec3<f32>) -> RainVertexOutput {
    var out: RainVertexOutput;

    let box_size = vec3<f32>(80.0, 50.0, 80.0);
    let half_box = box_size * 0.5;
    let fall_speed = 50.0;
    let y_drop = camera.time * fall_speed;

    // 1. World Position
    let world_pos_base = i_offset + camera.camera_pos;
    let wind_drift = vec3<f32>(camera.wind.x, 0.0, camera.wind.y) * camera.time * 10.0;

    var pos_temp = i_offset;
    pos_temp.y -= y_drop;
    pos_temp.x += wind_drift.x;
    pos_temp.z += wind_drift.z;

    let x_rel = (pos_temp.x % box_size.x + box_size.x + half_box.x) % box_size.x - half_box.x;
    let y_rel = (pos_temp.y % box_size.y + box_size.y + half_box.y) % box_size.y - half_box.y;
    let z_rel = (pos_temp.z % box_size.z + box_size.z + half_box.z) % box_size.z - half_box.z;

    let particle_world_pos = camera.camera_pos + vec3<f32>(x_rel, y_rel, z_rel);

    // 2. Check Local Weather
    let weather_data = get_regional_weather(particle_world_pos.xz);
    let rain_intensity = weather_data.y;

    // 3. Cloud Masking (Approximate cloud base)
    let c_bottom = mix(130.0, 90.0, weather_data.x);
    let height_mask = 1.0 - smoothstep(c_bottom - 10.0, c_bottom, particle_world_pos.y);

    // 4. Geometry
    let w = 0.03; let h = 0.8;
    var pos = vec3<f32>(0.0);
    var uv = vec2<f32>(0.0);

    if (v_idx == 0u) { pos = vec3<f32>(-w,  h, 0.0); uv = vec2<f32>(0.0, 0.0); }
    if (v_idx == 1u) { pos = vec3<f32>( w,  h, 0.0); uv = vec2<f32>(1.0, 0.0); }
    if (v_idx == 2u) { pos = vec3<f32>(-w, -h, 0.0); uv = vec2<f32>(0.0, 1.0); }
    if (v_idx == 3u) { pos = vec3<f32>( w, -h, 0.0); uv = vec2<f32>(1.0, 1.0); }

    let wind_vel = vec3<f32>(camera.wind.x * 20.0, -fall_speed, camera.wind.y * 20.0);
    let streak_dir = normalize(wind_vel);
    let view_vec = normalize(particle_world_pos - camera.camera_pos);
    let right_vec = normalize(cross(view_vec, streak_dir));

    let vert_offset = right_vec * pos.x + streak_dir * (pos.y * 1.5);
    let final_pos = particle_world_pos + vert_offset;

    out.clip_position = camera.view_proj * vec4<f32>(final_pos, 1.0);
    out.uv = uv;

    let dist_fade = 1.0 - smoothstep(30.0, 50.0, length(vec3<f32>(x_rel, y_rel, z_rel)));

    // Global rain switch * local intensity * height mask * distance
    out.alpha = camera.rain * rain_intensity * height_mask * dist_fade;
    return out;
}

@fragment
fn fs_rain(in: RainVertexOutput) -> @location(0) vec4<f32> {
    if (in.alpha < 0.01) { discard; }
    let x_fade = 1.0 - abs(in.uv.x * 2.0 - 1.0);
    let y_fade = 1.0 - abs(in.uv.y * 2.0 - 1.0);
    return vec4<f32>(0.7, 0.8, 0.9, in.alpha * x_fade * y_fade * 0.7);
}

// --- MAIN FRAGMENT ---

fn dither(frag_coord: vec2<f32>) -> f32 {
    let magic = vec3<f32>(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(frag_coord, magic.xy)));
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

    // Key: Use local weather for coverage
    let coverage = mix(0.75, 0.30, local_weather);

    return smoothstep(coverage, coverage + 0.25, n);
}

fn get_light_transmittance(pos: vec3<f32>, sun_dir: vec3<f32>, local_weather: f32) -> f32 {
    let steps = 3;
    let step_size = 15.0;
    var density = 0.0;
    for (var i = 0; i < steps; i++) {
        let sample_pos = pos + sun_dir * (f32(i) * step_size);
        density += get_cloud_density(sample_pos, local_weather);
    }
    return exp(-density * 1.2);
}

@fragment
fn fs_clouds(in: VertexOutput) -> @location(0) vec4<f32> {
    let ndc = vec4<f32>(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0, 1.0, 1.0);
    let world_pos_hom = camera.inv_view_proj * ndc;
    let world_pos = world_pos_hom.xyz / world_pos_hom.w;
    let view_dir = normalize(world_pos - camera.camera_pos);
    let rnd = dither(in.clip_position.xy);

    let sun_pos = get_sun_pos();
    let sun_col = get_sun_light_color(sun_pos.y);

    // Sample Weather at horizon for Sky Color
    let look_target = camera.camera_pos.xz + view_dir.xz * 1000.0;
    let weather_data = get_regional_weather(look_target);
    let sky_weather = weather_data.x;

    // Sky
    let col_day_z = vec3<f32>(0.1, 0.4, 0.9);
    let col_day_h = vec3<f32>(0.6, 0.8, 1.0);
    let col_set_z = vec3<f32>(0.2, 0.1, 0.4);
    let col_set_h = vec3<f32>(0.9, 0.5, 0.1);
    let col_night_z = vec3<f32>(0.01, 0.01, 0.04);
    let col_night_h = vec3<f32>(0.02, 0.03, 0.08);

    var sky_z = col_day_z;
    var sky_h = col_day_h;
    let sun_h = sun_pos.y;

    if (sun_h < 0.2 && sun_h > -0.2) {
        let t = smoothstep(-0.2, 0.2, sun_h);
        sky_z = mix(col_night_z, mix(col_set_z, col_day_z, t), t);
        sky_h = mix(col_night_h, mix(col_set_h, col_day_h, t), t);
    } else if (sun_h <= -0.2) {
        sky_z = col_night_z; sky_h = col_night_h;
    }

    let storm_dark = mix(1.0, 0.1, sky_weather);
    sky_z *= storm_dark;
    sky_h *= storm_dark;

    let horizon = pow(1.0 - max(view_dir.y, 0.0), 3.0);
    var color = mix(sky_z, sky_h, horizon);

    let lightning_flash = camera.lightning_color * camera.lightning_intensity * 0.3;
    color += lightning_flash * 0.1;

    // Clouds
    // Use local weather at camera for bounds, or blend?
    // Let's use the sampled weather along ray for more dynamic feel
    let c_bottom = mix(130.0, 90.0, sky_weather);
    let c_top    = mix(170.0, 250.0, sky_weather);

    if (view_dir.y > -0.1 || camera.camera_pos.y > 100.0) {
        let hit = intersect_slab(camera.camera_pos, view_dir, c_bottom, c_top);
        let t_min = hit.x;
        let t_max = hit.y;

        if (t_max > 0.0 && t_max > t_min) {
            let t_start = max(0.0, t_min);
            let dist_limit = 4000.0;

            if (t_start < dist_limit) {
                let steps = 40;
                let march_dist = min(t_max, dist_limit) - t_start;
                let step_size = march_dist / f32(steps);

                var t = t_start + step_size * rnd;
                var total_trans = 1.0;
                var acc_color = vec3<f32>(0.0);

                let cos_theta = dot(view_dir, sun_pos);
                let phase = 0.6 + 0.4 * pow(0.5 * (cos_theta + 1.0), 8.0);

                let ambient_base = mix(sky_z, sky_h, 0.5) * 0.8 + lightning_flash * 0.2;

                for (var i = 0; i < steps; i++) {
                    if (total_trans < 0.01) { break; }

                    let pos = camera.camera_pos + view_dir * t;

                    // Sample Regional Weather per voxel
                    let local_w_data = get_regional_weather(pos.xz);
                    let local_w = local_w_data.x;

                    let h = (pos.y - c_bottom) / (c_top - c_bottom);
                    let h_fade = smoothstep(0.0, 0.2, h) * smoothstep(1.0, 0.8, h);

                    // Pass local weather to density function
                    let d = get_cloud_density(pos, local_w);
                    let loc_den = d * h_fade;

                    if (loc_den > 0.001) {
                        let den_scale = mix(0.02, 0.20, local_w);
                        let step_od = loc_den * step_size * den_scale;
                        let step_trans = exp(-step_od);
                        let step_op = 1.0 - step_trans;

                        let sun_shadow = get_light_transmittance(pos, sun_pos, local_w);
                        let powder = 1.0 - exp(-loc_den * 2.0);

                        let direct = sun_col * 2.0 * sun_shadow * phase * (0.5 + powder);

                        // Lightning
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
                let c_final = mix(acc_color, color, fog);
                color = color * total_trans + c_final;
            }
        }
    }

    // Rainbow
    if (weather_data.y > 0.1 && sun_pos.y > 0.0) {
        let anti_sun = -sun_pos;
        let bow_angle = dot(view_dir, anti_sun);
        let diff = bow_angle - 0.74;
        if (abs(diff) < 0.04) {
            let t = (diff / 0.04) * 0.5 + 0.5;
            let r = smoothstep(0.4, 0.6, t) - smoothstep(0.8, 1.0, t);
            let g = smoothstep(0.2, 0.4, t) - smoothstep(0.6, 0.8, t);
            let b = smoothstep(0.0, 0.2, t) - smoothstep(0.4, 0.6, t);
            let bow = vec3<f32>(r, g, b);
            color += bow * 0.3 * weather_data.y * sun_pos.y;
        }
    }

    // Sun
    let sun_d = dot(view_dir, sun_pos);
    let sun_disk = smoothstep(0.9995, 0.9998, sun_d);
    color += sun_col * sun_disk * 5.0;

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
