fn get_regional_weather(world_xz: vec2<f32>) -> vec2<f32> {
    let scale = 0.0003;
    let scroll = vec2<f32>(camera.time * 2.0, 0.0);
    let sample_pos = (world_xz + scroll) * scale;

    // Base large-scale weather pattern
    var w = noise2d(sample_pos);
    w += noise2d(sample_pos * 2.0) * 0.5;
    w = w / 1.5;
    w = clamp(w + camera.weather_offset, 0.0, 1.0);

    // --- Rain Waves ---
    // Create smaller, faster moving waves to break up the "boxy" look
    // We mix high-frequency scrolling noise into the rain threshold
    let wave_scroll = vec2<f32>(camera.time * 15.0, camera.time * 5.0);
    let wave_noise = noise2d((world_xz + wave_scroll) * 0.02);

    // Modulate the rain threshold with the wave
    // This creates a transition zone where rain comes and goes in waves
    let rain_base = smoothstep(0.5, 0.9, w);
    let rain_variation = rain_base * (0.7 + 0.3 * wave_noise);

    return vec2<f32>(w, rain_variation);
}

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
        if (sun_h > 0.1) {
            color = vec3<f32>(1.0, 0.98, 0.9);
            ambient = vec3<f32>(0.6, 0.8, 1.0);
        } else {
            let t = (sun_h + 0.2) / 0.3;
            color = mix(vec3<f32>(1.0, 0.3, 0.0), vec3<f32>(1.0, 0.9, 0.7), t);
            ambient = mix(vec3<f32>(0.2, 0.1, 0.3), vec3<f32>(0.6, 0.7, 0.9), t);
        }
        let fade = smoothstep(-0.2, 0.0, sun_h);
        color *= fade; ambient *= fade;
    } else {
        dir = get_moon_pos();
        color = vec3<f32>(0.4, 0.5, 0.7) * 0.2;
        ambient = vec3<f32>(0.02, 0.02, 0.05);
        let fade = smoothstep(-0.2, 0.0, dir.y);
        color *= fade;
    }

    // --- Storm Lighting Fix ---
    // Made storm lighting significantly brighter so you can see effects
    let storm = clamp(camera.weather_offset + 0.2, 0.0, 1.0);

    // Instead of multiplying by 0.1 (very dark), we multiply by 0.3
    color = mix(color, color * 0.3, storm);

    // Increased ambient light during storms from 0.3 to 0.6 so voxels aren't black
    ambient = mix(ambient, ambient * 0.6, storm);

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

    // Lightened the storm sky slightly so it's less oppressive
    let storm_dark = mix(1.0, 0.3, weather);
    sky_z *= storm_dark; sky_h *= storm_dark;
    let horizon = pow(1.0 - max(view_dir.y, 0.0), 3.0);
    return mix(sky_z, sky_h, horizon);
}

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
    let pos_a = (p + wind) * 0.015; let noise_a = fbm_fast(pos_a);
    let pos_b = (p + wind) * vec3<f32>(0.005, 0.02, 0.005); let noise_b = fbm_fast(pos_b);
    let n = mix(noise_a, noise_b, c_type);
    let coverage = mix(0.65, 0.35, local_weather);
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
