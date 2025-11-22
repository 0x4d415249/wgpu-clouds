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
    if (is_sky) {
        color = get_sky_color(view_dir, sun_pos, sky_weather);
        color += get_stars(view_dir, sun_pos.y, sky_weather);
        let lightning_flash = camera.lightning_color * camera.lightning_intensity * 0.3;
        color += lightning_flash * 0.1;
        if (env[0].y > 0.0) {
             color += env[1] * smoothstep(0.9995, 0.9998, dot(view_dir, env[0])) * 5.0;
        } else {
             color += vec3<f32>(0.8, 0.9, 1.0) * smoothstep(0.9985, 0.999, dot(view_dir, get_moon_pos())) * 5.0;
        }
        let w_data_near = get_regional_weather(camera.camera_pos.xz);
        if (w_data_near.y > 0.1 && sun_pos.y > 0.0) {
            let bow_angle = dot(view_dir, -sun_pos);
            let diff = bow_angle - 0.74;
            if (abs(diff) < 0.04) {
                let t = (diff / 0.04) * 0.5 + 0.5;
                let bow = vec3<f32>(
                    smoothstep(0.4, 0.6, t) - smoothstep(0.8, 1.0, t),
                    smoothstep(0.2, 0.4, t) - smoothstep(0.6, 0.8, t),
                    smoothstep(0.0, 0.2, t) - smoothstep(0.4, 0.6, t)
                );
                color += bow * 0.3 * w_data_near.y * sun_pos.y;
            }
        }
    }

    // --- Volumetric Clouds ---
    let plane_noise = noise(camera.camera_pos * 0.001 + view_dir * 10.0) * 30.0;
    let c_bottom = mix(140.0, 90.0, sky_weather) + plane_noise * 0.5;
    let c_top    = mix(170.0, 250.0, sky_weather) + plane_noise;

    var total_trans = 1.0;

    if (view_dir.y > -0.5 || camera.camera_pos.y > 50.0) {
        let hit = intersect_slab(camera.camera_pos, view_dir, c_bottom, c_top);
        let t_min = hit.x; let t_max = hit.y;

        if (t_max > 0.0 && t_max > t_min) {
            let t_start = max(0.0, t_min);
            // We clamp the end of the ray to the geometry distance.
            // This naturally handles "softness" because if the geometry is
            // very close to the start of the cloud, the ray is short,
            // accumulating less density -> more transparent.
            let t_end = min(t_max, min(geom_dist, 4000.0));

            // Crucial check: Only render if the geometry is actually BEHIND the cloud start.
            // If t_end < t_start, the mountain is closer than the cloud layer, so we shouldn't draw clouds on it.
            if (t_end > t_start) {
                let steps = 40;
                let step_size = (t_end - t_start) / f32(steps);
                var t = t_start + step_size * rnd;
                var acc_color = vec3<f32>(0.0);
                let den_scale = mix(0.02, 0.15, sky_weather);
                let phase = 0.6 + 0.4 * pow(0.5 * (dot(view_dir, env[0]) + 1.0), 8.0);
                let ambient_base = env[2];

                for (var i = 0; i < steps; i++) {
                    if (total_trans < 0.01) { break; }
                    let pos = camera.camera_pos + view_dir * t;

                    // REMOVED: The artificial "depth_fade" that was causing the gap/cutoff.
                    // Now we just integrate density purely based on position.

                    let local_w = get_regional_weather(pos.xz).x;
                    let h = (pos.y - c_bottom) / (c_top - c_bottom);
                    let h_fade = smoothstep(0.0, 0.2, h) * smoothstep(1.0, 0.8, h);
                    let loc_den = get_cloud_density(pos, local_w) * h_fade;

                    if (loc_den > 0.001) {
                        let step_od = loc_den * step_size * den_scale;
                        let step_trans = exp(-step_od);
                        let sun_shadow = get_light_transmittance(pos, env[0], local_w);
                        let direct = env[1] * 2.0 * sun_shadow * phase * (0.5 + (1.0 - exp(-loc_den * 2.0)));
                        let l_vec = camera.lightning_pos - pos;
                        let l_att = 1.0 / (1.0 + dot(l_vec, l_vec) * 0.00003);
                        let point_light = camera.lightning_color * camera.lightning_intensity * 150.0 * l_att;

                        let light_res = mix(ambient_base * mix(mix(0.6, 0.02, local_w), 1.0, h), direct, 0.7) + point_light;

                        acc_color += light_res * (1.0 - step_trans) * total_trans;
                        total_trans *= step_trans;
                    }
                    t += step_size;
                }

                // Distance fog for the clouds themselves (fades them into the sky color far away)
                let fog = 1.0 - exp(-t_start * 0.0003);
                color = color * total_trans + mix(acc_color, color, fog);
            }
        }
    }

    // Output Composition
    // If looking at Sky (is_sky): Alpha = 1.0 (We replace the black background completely)
    // If looking at Voxel (!is_sky): Alpha = 1.0 - total_trans (This represents the opacity of the clouds)
    let final_alpha = select(1.0 - total_trans, 1.0, is_sky);

    return vec4<f32>(pow(color, vec3<f32>(1.0 / 2.2)), final_alpha);
}
