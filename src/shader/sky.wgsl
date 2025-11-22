@group(2) @binding(0) var t_depth: texture_depth_2d;

struct SkyboxVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_sky(@builtin(vertex_index) v_idx: u32) -> SkyboxVertexOutput {
    var out: SkyboxVertexOutput;
    var uvs = array<vec2<f32>, 3>(vec2<f32>(0.0, 0.0), vec2<f32>(2.0, 0.0), vec2<f32>(0.0, 2.0));
    let uv = uvs[v_idx];
    out.uv = uv;
    out.clip_position = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    return out;
}

// --- SKY COLORS (Ported from skyColors.glsl) ---
// Simplified for WGSL and specific use case
fn get_sky_gradient(view_dir: vec3<f32>, sun_pos: vec3<f32>, weather: f32) -> vec3<f32> {
    let sun_factor = max(dot(normalize(sun_pos), vec3<f32>(0.0, 1.0, 0.0)), 0.0);
    let rain_factor = weather;
    
    // Base colors (approximate from GLSL)
    let day_up = vec3<f32>(0.1, 0.4, 0.8);
    let day_middle = vec3<f32>(0.4, 0.6, 0.9);
    let day_down = vec3<f32>(0.7, 0.8, 0.9);
    
    let sunset_up = vec3<f32>(0.2, 0.1, 0.3);
    let sunset_middle = vec3<f32>(0.7, 0.3, 0.1);
    let sunset_down = vec3<f32>(0.9, 0.6, 0.2);
    
    let night_up = vec3<f32>(0.0, 0.0, 0.02);
    let night_middle = vec3<f32>(0.02, 0.02, 0.05);
    let night_down = vec3<f32>(0.05, 0.05, 0.1);
    
    // Mix based on sun height (day/sunset/night)
    // Simplified mixing logic
    let is_day = smoothstep(-0.2, 0.2, sun_pos.y);
    let is_sunset = 1.0 - abs(sun_pos.y * 5.0); // Peak at horizon
    let is_sunset_clamped = clamp(is_sunset, 0.0, 1.0);
    
    var up = mix(night_up, day_up, is_day);
    var middle = mix(night_middle, day_middle, is_day);
    var down = mix(night_down, day_down, is_day);
    
    // Apply sunset
    up = mix(up, sunset_up, is_sunset_clamped);
    middle = mix(middle, sunset_middle, is_sunset_clamped);
    down = mix(down, sunset_down, is_sunset_clamped);
    
    // Apply rain/weather
    let rain_color = vec3<f32>(0.1, 0.12, 0.15);
    up = mix(up, rain_color, rain_factor * 0.8);
    middle = mix(middle, rain_color, rain_factor * 0.8);
    down = mix(down, rain_color, rain_factor * 0.8);
    
    // Gradient mixing based on view direction
    let v_dot_u = view_dir.y;
    let horizon = smoothstep(-0.1, 0.3, v_dot_u);
    let zenith = smoothstep(0.3, 1.0, v_dot_u);
    
    var sky = mix(down, middle, horizon);
    sky = mix(sky, up, zenith);
    
    return sky;
}

// --- AURORA ---
fn get_aurora(view_dir: vec3<f32>, sun_h: f32) -> vec3<f32> {
    // Aurora is usually in the north, but let's make it cover the sky a bit
    // Map view_dir to a plane
    if (view_dir.y < 0.0) { return vec3<f32>(0.0); }
    
    let t = camera.time * 0.5;
    let pos = vec2<f32>(view_dir.x / view_dir.y, view_dir.z / view_dir.y) * 0.5;
    
    // FBM for aurora bands
    var noise_val = 0.0;
    var p = pos + vec2<f32>(t * 0.1, t * 0.05);
    
    noise_val += noise2d(p * 2.0) * 0.5;
    noise_val += noise2d(p * 4.0 + vec2<f32>(t * 0.2, 0.0)) * 0.25;
    noise_val += noise2d(p * 8.0) * 0.125;
    
    // Shape the aurora
    // We want bands that fade out at horizon and zenith
    let horizon_fade = smoothstep(0.0, 0.2, view_dir.y);
    let zenith_fade = 1.0 - smoothstep(0.8, 1.0, view_dir.y);
    
    let intensity = smoothstep(0.4, 0.8, noise_val) * horizon_fade * zenith_fade;
    
    // Colors (Green/Purple/Blue)
    let color_a = vec3<f32>(0.0, 1.0, 0.5); // Green
    let color_b = vec3<f32>(0.5, 0.0, 1.0); // Purple
    
    // Mix colors based on noise or position
    let color_mix = smoothstep(0.3, 0.7, noise2d(p * 1.5));
    let aurora_col = mix(color_a, color_b, color_mix);
    
    // Fade in/out based on night depth
    let night_factor = smoothstep(-0.2, -0.5, sun_h);
    
    return aurora_col * intensity * night_factor * 0.5; // Scale brightness
}


// --- CLOUDS (Ported from reimaginedClouds.glsl) ---
// hash and noise are inherited from common.wgsl

fn fbm_clouds(p: vec3<f32>) -> f32 {
    var val = 0.0; var amp = 0.5; var pos = p;
    val += noise(pos) * amp; pos *= 2.02; amp *= 0.5;
    val += noise(pos) * amp; pos *= 2.03; amp *= 0.5;
    val += noise(pos) * amp;
    return val;
}

fn get_volumetric_cloud_density(pos: vec3<f32>, weather: f32) -> f32 {
    let time_offset = vec3<f32>(camera.time * 5.0, 0.0, camera.time * 2.0);
    let p = (pos + time_offset) * 0.002; // Scale
    
    var n = fbm_clouds(p);
    
    // Shape
    let coverage = mix(0.4, 0.7, weather); // Increase coverage with rain
    let density = smoothstep(1.0 - coverage, 1.0 - coverage + 0.1, n);
    
    return density;
}

// intersect_slab is inherited from weather.wgsl

@fragment
fn fs_sky(in: SkyboxVertexOutput) -> @location(0) vec4<f32> {
    let ndc = vec4<f32>(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0, 1.0, 1.0);
    let world_pos_hom = camera.inv_view_proj * ndc;
    let view_dir = normalize(world_pos_hom.xyz / world_pos_hom.w - camera.camera_pos);
    let rnd = dither(in.clip_position.xy);

    let depth_val = textureLoad(t_depth, vec2<i32>(in.clip_position.xy), 0);
    
    // Depth Reconstruction
    let geom_ndc = vec4<f32>(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0, depth_val, 1.0);
    let geom_pos_hom = camera.inv_view_proj * geom_ndc;
    let geom_pos = geom_pos_hom.xyz / geom_pos_hom.w;
    let geom_dist = distance(camera.camera_pos, geom_pos);
    let is_sky = depth_val >= 1.0;

    let sun_pos = get_sun_pos();
    let env = get_environment_light();
    let look_target = camera.camera_pos.xz + view_dir.xz * 1000.0;
    let w_data = get_regional_weather(look_target);
    let sky_weather = w_data.x;

    // --- Draw Sky Background ---
    var color = vec3<f32>(0.0);
    
    // Use new gradient function
    color = get_sky_gradient(view_dir, sun_pos, sky_weather);
    
    if (is_sky) {
        // Stars
        color += get_stars(view_dir, sun_pos.y, sky_weather);
        
        // Sun/Moon Glare
        let sun_dot = max(dot(view_dir, normalize(sun_pos)), 0.0);
        let moon_dot = max(dot(view_dir, normalize(-sun_pos)), 0.0);
        
        let sun_glare = pow(sun_dot, 200.0) * 10.0; // Simple glare
        let moon_glare = pow(moon_dot, 100.0) * 2.0;
        
        color += vec3<f32>(1.0, 0.9, 0.7) * sun_glare;
        color += vec3<f32>(0.5, 0.6, 0.8) * moon_glare;
        
        // Aurora Borealis
        // Only visible at night and if weather is clear (or slightly rainy/snowy in some biomes, but let's stick to clear night)
        if (sun_pos.y < -0.2 && sky_weather < 0.5) {
             let aurora = get_aurora(view_dir, sun_pos.y);
             color += aurora;
        }
    }

    // --- Volumetric Clouds ---
    let c_bottom = 120.0;
    let c_top = 220.0;
    
    var total_trans = 1.0;
    
    if (view_dir.y > 0.0 || camera.camera_pos.y > c_bottom) {
        let hit = intersect_slab(camera.camera_pos, view_dir, c_bottom, c_top);
        let t_min = hit.x; 
        let t_max = hit.y;

        if (t_max > 0.0 && t_max > t_min) {
            let t_start = max(0.0, t_min);
            let t_end = min(t_max, min(geom_dist, 4000.0));
            
            if (t_end > t_start) {
                let steps = 32;
                let step_size = (t_end - t_start) / f32(steps);
                var t = t_start + step_size * rnd;
                var acc_color = vec3<f32>(0.0);
                
                let sun_dir = normalize(sun_pos);
                let ambient = env[2];
                let light_col = env[1];
                
                for (var i = 0; i < steps; i++) {
                    if (total_trans < 0.01) { break; }
                    let pos = camera.camera_pos + view_dir * t;
                    
                    let den = get_volumetric_cloud_density(pos, sky_weather);
                    
                    if (den > 0.01) {
                        let step_od = den * step_size * 0.01;
                        let step_trans = exp(-step_od);
                        
                        // Lighting
                        // Simple directional lighting for clouds
                        let light_sample_pos = pos + sun_dir * 10.0;
                        let light_den = get_volumetric_cloud_density(light_sample_pos, sky_weather);
                        let shadow = exp(-light_den * 2.0); // Soft shadow
                        
                        let cloud_light = ambient + light_col * shadow * 2.0;
                        
                        acc_color += cloud_light * (1.0 - step_trans) * total_trans;
                        total_trans *= step_trans;
                    }
                    
                    t += step_size;
                }
                
                // Blend clouds
                color = color * total_trans + acc_color;
            }
        }
    }

    let final_alpha = select(1.0 - total_trans, 1.0, is_sky);
    return vec4<f32>(color, final_alpha);
}

