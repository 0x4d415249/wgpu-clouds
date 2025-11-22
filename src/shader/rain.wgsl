struct Particle { pos: vec4<f32>, vel: vec4<f32> }
@group(2) @binding(0) var<storage, read_write> particles: array<Particle>;

@compute @workgroup_size(64)
fn cs_rain(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= arrayLength(&particles)) { return; }

    var p = particles[idx];
    let wind_force = vec3<f32>(camera.wind.x * 20.0, 0.0, camera.wind.y * 20.0);
    let fall_vel = p.vel.y;
    let velocity = vec3<f32>(wind_force.x, fall_vel, wind_force.z);

    p.pos += vec4<f32>(velocity * 0.016, 0.0);

    let range_h = 60.0;
    let range_y = 40.0;
    let center = camera.camera_pos;
    let dist = p.pos.xyz - center;

    if (dist.x > range_h) { p.pos.x -= range_h * 2.0; }
    if (dist.x < -range_h) { p.pos.x += range_h * 2.0; }
    if (dist.z > range_h) { p.pos.z -= range_h * 2.0; }
    if (dist.z < -range_h) { p.pos.z += range_h * 2.0; }
    if (dist.y < -range_y) { p.pos.y += range_y * 2.0; }
    if (dist.y > range_y) { p.pos.y -= range_y * 2.0; }

    particles[idx] = p;
}

struct RainVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) alpha: f32,
}

@vertex
fn vs_rain(@builtin(vertex_index) v_idx: u32, @builtin(instance_index) i_idx: u32) -> RainVertexOutput {
    var out: RainVertexOutput;
    let p = particles[i_idx];
    let center = p.pos.xyz;
    let scale = p.pos.w;

    let w_data = get_regional_weather(center.xz);
    let c_bottom = mix(130.0, 90.0, w_data.x);
    let height_mask = 1.0 - smoothstep(c_bottom - 10.0, c_bottom, center.y);

    let wind_force = vec3<f32>(camera.wind.x * 20.0, 0.0, camera.wind.y * 20.0);
    let vel = vec3<f32>(wind_force.x, p.vel.y, wind_force.z);
    let dir = normalize(vel);

    let w = 0.03 * scale;
    let h = 0.8 * scale;
    var pos = vec3<f32>(0.0);
    var uv = vec2<f32>(0.0);

    let view_vec = normalize(center - camera.camera_pos);
    let right = normalize(cross(view_vec, dir));

    if (v_idx == 0u) { pos = -right * w + dir * h; uv = vec2<f32>(0.0, 0.0); }
    if (v_idx == 1u) { pos =  right * w + dir * h; uv = vec2<f32>(1.0, 0.0); }
    if (v_idx == 2u) { pos = -right * w - dir * h; uv = vec2<f32>(0.0, 1.0); }
    if (v_idx == 3u) { pos =  right * w - dir * h; uv = vec2<f32>(1.0, 1.0); }

    let world_pos = center + pos;
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.uv = uv;

    let d = distance(center, camera.camera_pos);
    let dist_fade = 1.0 - smoothstep(40.0, 60.0, d);
    out.alpha = camera.rain * w_data.y * height_mask * dist_fade;
    return out;
}

@fragment
fn fs_rain(in: RainVertexOutput) -> @location(0) vec4<f32> {
    if (in.alpha < 0.01) { discard; }
    let x_fade = 1.0 - abs(in.uv.x * 2.0 - 1.0);
    let y_fade = 1.0 - abs(in.uv.y * 2.0 - 1.0);
    let lightning = camera.lightning_color * camera.lightning_intensity * 0.5;
    let col = vec3<f32>(0.7, 0.8, 0.9) + lightning;
    return vec4<f32>(col, in.alpha * x_fade * y_fade * 0.6);
}
