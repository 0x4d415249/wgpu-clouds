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

// --- WAVING LOGIC ---
fn GetRawWave(pos: vec3<f32>, wind: f32) -> vec3<f32> {
    let magnitude = sin(wind * 0.0027 + pos.z + pos.y) * 0.04 + 0.04;
    let d0 = sin(wind * 0.0127);
    let d1 = sin(wind * 0.0089);
    let d2 = sin(wind * 0.0114);
    var wave = vec3<f32>(0.0);
    wave.x = sin(wind * 0.0063 + d0 + d1 - pos.x + pos.z + pos.y) * magnitude;
    wave.z = sin(wind * 0.0224 + d1 + d2 + pos.x - pos.z + pos.y) * magnitude;
    wave.y = sin(wind * 0.0015 + d2 + d0 + pos.z + pos.y - pos.y) * magnitude;
    return wave;
}

fn GetWave(pos: vec3<f32>, waveSpeed: f32) -> vec3<f32> {
    let wind = camera.time * waveSpeed * 1.0; // WAVING_SPEED assumed 1.0
    var wave = GetRawWave(pos, wind);
    
    // Rain influence (simplified)
    let rainFactor = camera.rain;
    let wavingIntensity = 1.0 * mix(1.0, 0.5, rainFactor); // WAVING_I assumed 1.0

    return wave * wavingIntensity;
}

fn DoWave(pos: vec3<f32>, wave_type: f32) -> vec3<f32> {
    var p = pos;
    let world_pos = pos + camera.camera_pos; // Approximate world pos for noise
    
    if (wave_type > 0.5 && wave_type < 1.5) { // 1.0: Foliage
        // Only wave top vertices usually, but for cross models we wave everything but bottom?
        // Simplified: wave everything based on height or UV?
        // For now, just wave everything.
        
        // DoWave_Foliage logic
        var wp = world_pos;
        wp.y *= 0.5;
        var wave = GetWave(wp, 170.0);
        wave.x = wave.x * 8.0 + wave.y * 4.0;
        wave.y = 0.0;
        wave.z = wave.z * 3.0;
        wave *= 0.1;
        
        // Mask bottom vertices? 
        // We don't have UVs easily accessible here to check bottom, but usually y-pos check works.
        // Assuming local pos y=0 is bottom.
        // But we are in world space (chunk relative).
        // Let's just apply it.
        p += wave;
    } else if (wave_type > 1.5 && wave_type < 2.5) { // 2.0: Leaves
        var wp = world_pos * vec3<f32>(0.75, 0.375, 0.75);
        var wave = GetWave(wp, 170.0);
        wave *= vec3<f32>(8.0, 3.0, 4.0);
        wave *= 0.1;
        p += wave;
    } else if (wave_type > 2.5 && wave_type < 3.5) { // 3.0: Water
        let waterWaveTime = camera.time * 6.0;
        var wp = world_pos;
        wp.x *= 14.0;
        wp.z *= 14.0;
        var wave = sin(waterWaveTime * 0.7 + wp.x * 0.14 + wp.z * 0.07);
        wave += sin(waterWaveTime * 0.5 + wp.x * 0.10 + wp.z * 0.05);
        p.y += wave * 0.125 - 0.05;
    } else if (wave_type > 3.5 && wave_type < 4.5) { // 4.0: Lava
        let lavaWaveTime = camera.time * 3.0;
        var wp = world_pos;
        wp.x *= 14.0;
        wp.z *= 14.0;
        var wave = sin(lavaWaveTime * 0.7 + wp.x * 0.14 + wp.z * 0.07);
        wave += sin(lavaWaveTime * 0.5 + wp.x * 0.05 + wp.z * 0.10);
        p.y += wave * 0.0125;
    }
    
    return p;
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    // Apply Waving
    let wave_type = in.color.a;
    let pos_waved = DoWave(in.position, wave_type);
    
    let world_pos = vec4<f32>(pos_waved, 1.0);
    out.clip_position = camera.view_proj * world_pos;
    out.uv = in.uv;
    out.bounds = in.bounds;
    out.color = vec4<f32>(in.color.rgb, 1.0); // Reset alpha for fragment shader
    out.normal = in.normal;
    out.world_pos = pos_waved;
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

    // --- LIGHTING & ENVIRONMENT ---
    let w_data = get_regional_weather(in.world_pos.xz);
    let terrain_h = get_terrain_height(in.world_pos.xz);
    
    // Wetness
    let height_diff = terrain_h - in.world_pos.y;
    let is_surface_layer = height_diff < 4.0 && height_diff > -2.0;
    let rain_amt = select(0.0, camera.rain * w_data.y, is_surface_layer);
    let wetness_darkening = mix(1.0, 0.6, rain_amt * 0.8);
    tex_color = vec4<f32>(tex_color.rgb * wetness_darkening, tex_color.a);

    // Lighting Vectors
    let view_dir = normalize(in.world_pos - camera.camera_pos);
    let normal = normalize(in.normal);
    let sun_pos = get_sun_pos();
    let sun_dir = normalize(sun_pos); // Approximate sun direction
    
    // Basic Directional Shading (NdotL)
    let NdotL = max(dot(normal, sun_dir), 0.0);
    let NdotU = max(dot(normal, vec3<f32>(0.0, 1.0, 0.0)), 0.0);
    
    // Ambient
    let env = get_environment_light(); // [0]: sun_dir, [1]: sun_color, [2]: ambient
    let ambient = env[2];
    let light_color = env[1];
    
    // Improved Shading Model
    var diffuse = NdotL;
    
    // Side shading (fake AO/directional)
    let side_shade = mix(0.6, 1.0, NdotU);
    diffuse *= side_shade;

    // Shadows (Simple distance based fade for now, real shadow mapping is TODO)
    // We don't have shadow map bound yet in this pass.
    
    // Combine
    var lighting = ambient + (light_color * diffuse);
    
    // Lightning
    let l_vec = camera.lightning_pos - in.world_pos;
    let l_dist_sq = dot(l_vec, l_vec);
    let l_att = 1.0 / (1.0 + l_dist_sq * 0.00005);
    let lightning_light = camera.lightning_color * camera.lightning_intensity * 50.0 * l_att;
    lighting += lightning_light;

    var final_color = tex_color.rgb * in.color.rgb * lighting;

    // Reflections (Puddles)
    let reflect_dir = reflect(view_dir, normal);
    let sky_ref = get_sky_color(reflect_dir, sun_pos, w_data.x);
    let fresnel = pow(1.0 - max(dot(-view_dir, normal), 0.0), 3.0);
    let puddle_mask = max(normal.y, 0.0);
    let reflect_strength = rain_amt * (0.3 + 0.5 * fresnel) * puddle_mask;
    
    final_color = mix(final_color, sky_ref, reflect_strength);

    // Splash Overlay
    if (normal.y > 0.9 && rain_amt > 0.1) {
        let splash_speed = camera.time * 15.0;
        let splash_uv = in.world_pos.xz * 4.0;
        let n = noise(vec3<f32>(splash_uv.x, splash_uv.y, splash_speed));
        if (n > 0.85) {
            final_color = mix(final_color, vec3<f32>(0.8, 0.9, 1.0), 0.4);
        }
    }

    // --- ATMOSPHERIC FOG ---
    // Ported from mainFog.glsl (simplified)
    
    let dist = distance(in.world_pos, camera.camera_pos);
    
    // Altitude Factor (Fog decreases with height)
    let fog_altitude = 60.0;
    let fog_fade_height = 30.0;
    let altitude_factor = clamp(1.0 - (camera.camera_pos.y - fog_altitude) / fog_fade_height, 0.0, 1.0);
    let altitude_factor_sq = altitude_factor * altitude_factor;
    
    // Density
    // Rain increases fog density
    let rain_fog = camera.rain * 0.5;
    let base_density = 0.0005 + rain_fog * 0.002;
    
    // Calculate Fog Factor
    // exp(-density * dist) is standard.
    // Complementary uses a more complex falloff: 1 - exp(-pow(dist * ..., ...))
    // Let's stick to a nice exponential squared fog for volumetric feel.
    let fog_dist = max(dist - 20.0, 0.0);
    let fog_val = fog_dist * base_density * altitude_factor_sq;
    let fog_factor = 1.0 - exp(-fog_val * fog_val);
    
    // Fog Color
    // Mix sky color with a specific fog tint
    // During rain, fog is grey. During sunset, it's orange/purple.
    let fog_sky_ref = get_sky_color(view_dir, sun_pos, w_data.x);
    var fog_col = fog_sky_ref;
    
    // Cave Fog (Underground)
    // If we are deep underground, fog becomes black/dark.
    // We don't have lightmap, so use height as proxy.
    if (camera.camera_pos.y < 40.0) {
        let cave_factor = clamp((40.0 - camera.camera_pos.y) / 40.0, 0.0, 1.0);
        fog_col = mix(fog_col, vec3<f32>(0.0, 0.0, 0.0), cave_factor * 0.95);
        // Increase density underground
        // fog_factor = mix(fog_factor, 1.0 - exp(-dist * 0.05), cave_factor); 
        // Actually, just let it be dark.
    }

    final_color = mix(final_color, fog_col, fog_factor);

    return vec4<f32>(final_color, 1.0);
}
