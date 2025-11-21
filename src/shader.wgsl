// --- GROUP 0: UNIFORMS ---
struct CameraUniform {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    time: f32,
    day_progress: f32,
    weather: f32,
    cloud_type: f32,
}
@group(0) @binding(0) var<uniform> camera: CameraUniform;

// --- GROUP 1: UPSCALING (Texture & Sampler) ---
@group(1) @binding(0) var t_diffuse: texture_2d<f32>;
@group(1) @binding(1) var s_diffuse: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
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

// --- NOISE FUNCTIONS ---

fn dither(frag_coord: vec2<f32>) -> f32 {
    let magic = vec3<f32>(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(frag_coord, magic.xy)));
}

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

fn fbm_fast(p: vec3<f32>) -> f32 {
    var val = 0.0;
    var amp = 0.5;
    var pos = p;
    val += noise(pos) * amp; pos *= 2.02; amp *= 0.5;
    val += noise(pos) * amp;
    return val;
}

// --- CLOUD SHAPING ---

fn get_cloud_density(p: vec3<f32>) -> f32 {
    let weather = camera.weather;
    let c_type = camera.cloud_type;

    let wind = vec3<f32>(camera.time * (0.3 + weather * 0.5), 0.0, 0.0);

    // Cumulus
    let scale_a = 0.015;
    let pos_a = (p + wind) * scale_a;
    let noise_a = fbm_fast(pos_a);

    // Stratus
    let scale_b = vec3<f32>(0.005, 0.02, 0.005);
    let pos_b = (p + wind) * scale_b;
    let noise_b = fbm_fast(pos_b);

    let n = mix(noise_a, noise_b, c_type);

    let coverage = mix(0.65, 0.35, weather);
    let density = smoothstep(coverage, coverage + 0.25, n);

    return density;
}

// --- ATMOSPHERE ---

fn get_sun_dir() -> vec3<f32> {
    let angle = camera.day_progress * 6.28318;
    return normalize(vec3<f32>(sin(angle), cos(angle) * 0.8, -0.4));
}

fn get_sun_light_color(sun_y: f32) -> vec3<f32> {
    if (sun_y > 0.15) { return vec3<f32>(1.0, 0.98, 0.92); }
    if (sun_y > -0.15) {
        let t = (sun_y + 0.15) / 0.3;
        return mix(vec3<f32>(1.0, 0.5, 0.1), vec3<f32>(1.0, 0.98, 0.92), t);
    }
    return vec3<f32>(0.02, 0.02, 0.05);
}

fn intersect_slab(ro: vec3<f32>, rd: vec3<f32>, y_min: f32, y_max: f32) -> vec2<f32> {
    let inv_dir_y = 1.0 / (rd.y + 0.00001); // Prevent div by zero
    let t0 = (y_min - ro.y) * inv_dir_y;
    let t1 = (y_max - ro.y) * inv_dir_y;
    return vec2<f32>(min(t0, t1), max(t0, t1));
}

// --- PASS 1: CLOUD RAYMARCH ---
@fragment
fn fs_clouds(in: VertexOutput) -> @location(0) vec4<f32> {
    let ndc = vec4<f32>(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0, 1.0, 1.0);
    let world_pos_hom = camera.inv_view_proj * ndc;
    let world_pos = world_pos_hom.xyz / world_pos_hom.w;
    let view_dir = normalize(world_pos - camera.camera_pos);

    let rnd = dither(in.clip_position.xy);

    let sun_dir = get_sun_dir();
    let sun_height = sun_dir.y;
    let sun_col = get_sun_light_color(sun_height);

    // Sky Gradient
    let col_day_z = vec3<f32>(0.15, 0.45, 0.85);
    let col_day_h = vec3<f32>(0.65, 0.80, 0.95);

    let col_set_z = vec3<f32>(0.25, 0.15, 0.45);
    let col_set_h = vec3<f32>(0.95, 0.55, 0.25);

    let col_night_z = vec3<f32>(0.01, 0.01, 0.04);
    let col_night_h = vec3<f32>(0.02, 0.03, 0.08);

    var sky_z = col_day_z;
    var sky_h = col_day_h;

    if (sun_height < 0.2 && sun_height > -0.2) {
        let t = smoothstep(-0.2, 0.2, sun_height);
        sky_z = mix(col_night_z, mix(col_set_z, col_day_z, t), t);
        sky_h = mix(col_night_h, mix(col_set_h, col_day_h, t), t);
    } else if (sun_height <= -0.2) {
        sky_z = col_night_z;
        sky_h = col_night_h;
    }

    let horizon_curve = pow(1.0 - max(view_dir.y, 0.0), 3.0);
    var color = mix(sky_z, sky_h, horizon_curve);

    // Clouds
    let c_bottom = mix(130.0, 100.0, camera.weather);
    let c_top    = mix(170.0, 250.0, camera.weather);

    // Optimization: Don't trace ground
    if (view_dir.y > -0.1 || camera.camera_pos.y > 100.0) {
        let hit = intersect_slab(camera.camera_pos, view_dir, c_bottom, c_top);
        let t_min = hit.x;
        let t_max = hit.y;

        if (t_max > 0.0 && t_max > t_min) {
            let t_start = max(0.0, t_min);
            let dist_limit = 3000.0; // Further draw distance

            if (t_start < dist_limit) {
                let steps = 32;
                let march_dist = min(t_max, dist_limit) - t_start;
                let step_size = march_dist / f32(steps);

                var t = t_start + step_size * rnd;
                var total_transmittance = 1.0;
                var acc_color = vec3<f32>(0.0);

                // Density multiplier (Beer's law scaler)
                let density_scale = mix(0.02, 0.05, camera.weather);

                let cos_theta = dot(view_dir, sun_dir);
                // HG Phase function for silver lining
                let phase = 0.6 + 0.4 * pow(0.5 * (cos_theta + 1.0), 8.0);

                let ambient = mix(sky_z, sky_h, 0.5) * 0.8;

                for (var i = 0; i < steps; i++) {
                    if (total_transmittance < 0.01) { break; }

                    let pos = camera.camera_pos + view_dir * t;

                    let h = (pos.y - c_bottom) / (c_top - c_bottom);
                    // Parabolic fade for top/bottom softness
                    let h_fade = smoothstep(0.0, 0.2, h) * smoothstep(1.0, 0.8, h);

                    let d = get_cloud_density(pos);
                    let local_density = d * h_fade;

                    if (local_density > 0.001) {
                        // Calculate opacity for this step using Beer-Lambert: 1 - e^(-density * step)
                        // This prevents black artifacts when step size is large.
                        let step_optical_depth = local_density * step_size * density_scale;
                        let step_transmittance = exp(-step_optical_depth);
                        let step_opacity = 1.0 - step_transmittance;

                        // Lighting
                        let sun_access = smoothstep(0.0, 1.0, h + 0.2); // +0.2 prevents black bottoms
                        let powder = 1.0 - exp(-local_density * 2.0); // Darkens centers, brightens edges

                        let direct = sun_col * 1.8 * sun_access * phase * (0.5 + powder);

                        // Ensure ambient is never 0 to prevent black artifacts
                        let light_result = mix(ambient + vec3<f32>(0.05), direct, 0.6);

                        // Accumulate: Color + Transmittance
                        acc_color += light_result * step_opacity * total_transmittance;
                        total_transmittance *= step_transmittance;
                    }
                    t += step_size;
                }

                // Distance Fog
                let fog = 1.0 - exp(-t_start * 0.0003);
                let cloud_final = mix(acc_color, color, fog);

                // Alpha blend: ScreenColor * Transmittance + CloudColor
                color = color * total_transmittance + cloud_final;
            }
        }
    }

    // Sun
    let sun_dot = dot(view_dir, sun_dir);
    let sun_disk = smoothstep(0.9995, 0.9998, sun_dot);
    let sun_bloom = pow(max(sun_dot, 0.0), 60.0) * 0.6;
    color += sun_col * (sun_disk + sun_bloom);

    // Tone Mapping
    color = color / (color + vec3<f32>(1.0));
    color = pow(color, vec3<f32>(1.0 / 2.2));

    // Saturation Boost
    let luma = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    let sat_color = mix(vec3<f32>(luma), color, 1.2);

    return vec4<f32>(sat_color, 1.0);
}

// --- PASS 2: BLIT (Upscale) ---
@fragment
fn fs_blit(in: VertexOutput) -> @location(0) vec4<f32> {
    // Simply sample the texture generated in Pass 1
    // The sampler handles the bilinear filtering (upscaling)
    return textureSample(t_diffuse, s_diffuse, in.uv);
}
