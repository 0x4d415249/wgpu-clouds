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

struct Out { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }

@group(0) @binding(0) var t_scene: texture_2d<f32>;
@group(0) @binding(1) var s_scene: sampler;
@group(0) @binding(2) var t_depth: texture_depth_2d;
@group(1) @binding(0) var<uniform> camera: CameraUniform;

@vertex fn vs_main(@builtin(vertex_index) idx: u32) -> Out {
    var out: Out;
    var uvs = array<vec2<f32>,3>(vec2<f32>(0., 2.), vec2<f32>(0., 0.), vec2<f32>(2., 0.));
    out.uv = uvs[idx];
    out.pos = vec4<f32>(out.uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv.y = 1.0 - out.uv.y;
    return out;
}

fn get_view_pos(screen_uv: vec2<f32>) -> vec3<f32> {
    let depth = textureSample(t_depth, s_scene, screen_uv);
    let ndc = vec4<f32>(screen_uv.x * 2.0 - 1.0, 1.0 - screen_uv.y * 2.0, depth, 1.0);
    let world_pos_h = camera.inv_view_proj * ndc;
    return world_pos_h.xyz / world_pos_h.w;
}

@fragment fn fs_main(in: Out) -> @location(0) vec4<f32> {
    let tex = textureSample(t_scene, s_scene, in.uv);
    var col = tex.rgb;
    let reflectivity = tex.a;

    // SSR
    if (reflectivity > 0.01) {
        let world_pos = get_view_pos(in.uv);

        // Improved Normal Reconstruction
        let dx = dpdx(world_pos);
        let dy = dpdy(world_pos);
        let normal = normalize(cross(dx, dy));

        let view_dir = normalize(world_pos - camera.camera_pos);
        let reflect_dir = reflect(view_dir, normal);

        var hit_color = vec3<f32>(0.0);
        var hit = false;

        // Increased steps for better reach, smaller stride for precision
        let steps = 60;
        // Jitter the start position to reduce banding artifacts
        let jitter = fract(sin(dot(in.uv, vec2<f32>(12.9898, 78.233))) * 43758.5453);
        var ray_pos = world_pos + reflect_dir * 0.5;

        for (var i = 0; i < steps; i++) {
            ray_pos += reflect_dir * (0.5 + jitter * 0.1);

            let clip_pos = camera.view_proj * vec4<f32>(ray_pos, 1.0);
            if (clip_pos.w <= 0.0) { continue; }

            let ndc = clip_pos.xyz / clip_pos.w;
            let screen_uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);

            if (screen_uv.x < 0.0 || screen_uv.x > 1.0 || screen_uv.y < 0.0 || screen_uv.y > 1.0) {
                break;
            }

            let surface_world_pos = get_view_pos(screen_uv);

            // Robust depth check:
            // Ray must be *behind* the surface (ray distance > surface distance)
            // But not *too* far behind (thickness check), to avoid reflecting objects behind walls.
            let dist_cam_to_surface = distance(camera.camera_pos, surface_world_pos);
            let dist_cam_to_ray = distance(camera.camera_pos, ray_pos);

            if (dist_cam_to_ray > dist_cam_to_surface && dist_cam_to_ray < dist_cam_to_surface + 1.5) {
                hit_color = textureSample(t_scene, s_scene, screen_uv).rgb;

                // Edge Fade: Fade out reflections near the screen edges
                let border_dist = min(
                    min(screen_uv.x, 1.0 - screen_uv.x),
                    min(screen_uv.y, 1.0 - screen_uv.y)
                );
                let edge_alpha = smoothstep(0.0, 0.1, border_dist);

                hit_color *= edge_alpha;
                hit = true;
                break;
            }
        }

        if (hit) {
            // Mix reflection based on reflectivity
            col = mix(col, hit_color, reflectivity * 0.5);
        }
    }

    // --- TONE MAPPING & SATURATION ---
    // Filmic Tone Mapping (ACES approximation for better contrast/vibrancy)
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    // FIXED: Clamp boundaries must be vec3
    let mapped = clamp((col * (a * col + vec3<f32>(b))) / (col * (c * col + vec3<f32>(d)) + vec3<f32>(e)), vec3<f32>(0.0), vec3<f32>(1.0));

    // Saturation Boost
    let luma = dot(mapped, vec3<f32>(0.2126, 0.7152, 0.0722));
    let vibrance = 1.2; // Increase for more colorful look
    let final_col = mix(vec3<f32>(luma), mapped, vibrance);

    // Slight Gamma Correction (ACES already does some, but a little more helps darkness)
    return vec4<f32>(pow(final_col, vec3<f32>(1.0 / 1.1)), 1.0);
}
