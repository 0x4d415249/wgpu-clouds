use crate::data::GameRegistry;

pub fn generate_wgsl(registry: &GameRegistry) -> String {
    let mut s = String::new();

    s.push_str(r#"
    const PI: f32 = 3.14159265359;

    struct CameraUniform {
        view_proj: mat4x4<f32>,
        inv_view_proj: mat4x4<f32>,
        camera_pos: vec3<f32>,
        time: f32,
        screen_size: vec2<f32>,
        render_scale: f32,
        _pad: f32,
    }
    @group(0) @binding(0) var<uniform> camera: CameraUniform;
    @group(1) @binding(0) var t_diffuse: texture_2d<f32>;
    @group(1) @binding(1) var s_diffuse: sampler;
    @group(2) @binding(0) var t_depth: texture_depth_2d;

    fn hash(p: vec3<f32>) -> f32 {
        var p3 = fract(p * 0.1031);
        p3 += dot(p3, p3.yzx + 33.33);
        return fract((p3.x + p3.y) * p3.z);
    }

    fn noise(p: vec3<f32>) -> f32 {
        let i = floor(p);
        let f = fract(p);
        let u = f * f * (3.0 - 2.0 * f);
        return mix(mix(mix( hash(i + vec3<f32>(0,0,0)), hash(i + vec3<f32>(1,0,0)), u.x),
                       mix( hash(i + vec3<f32>(0,1,0)), hash(i + vec3<f32>(1,1,0)), u.x), u.y),
                   mix(mix( hash(i + vec3<f32>(0,0,1)), hash(i + vec3<f32>(1,0,1)), u.x),
                       mix( hash(i + vec3<f32>(0,1,1)), hash(i + vec3<f32>(1,1,1)), u.x), u.y), u.z);
    }

    fn noise2d(p: vec2<f32>) -> f32 { return noise(vec3<f32>(p.x, 0.0, p.y)); }
    "#);

    s.push_str("\n// --- BLOCK IDS ---\n");
    for (name, def) in &registry.blocks {
        let raw_name = name.to_uppercase().replace(":", "_");
        let const_name = format!("BLOCK_{}", raw_name);
        s.push_str(&format!(
            "const {}: u32 = {}u;\n",
            const_name, def.numeric_id
        ));
    }

    let get_id_str = |lookup: &str| -> String {
        let key = format!("maricraft:{}", lookup);
        if registry.blocks.contains_key(&key) {
            format!("BLOCK_MARICRAFT_{}", lookup.to_uppercase())
        } else {
            "1u".to_string()
        }
    };

    let id_grass = get_id_str("grass");
    let id_dirt = get_id_str("dirt");
    let id_stone = get_id_str("stone");
    let id_water = get_id_str("water");

    s.push_str(&format!(
        r#"
    struct GenParams {{ chunk_pos: vec3<i32>, seed: u32 }}
    @group(0) @binding(0) var<uniform> gen_params: GenParams;
    @group(0) @binding(1) var<storage, read_write> block_buffer: array<u32>;

    @compute @workgroup_size(4, 4, 4)
    fn cs_generate(@builtin(global_invocation_id) id: vec3<u32>) {{
        if (id.x >= 32u || id.y >= 32u || id.z >= 32u) {{ return; }}
        let wx = f32(gen_params.chunk_pos.x * 32) + f32(id.x);
        let wy = f32(gen_params.chunk_pos.y * 32) + f32(id.y);
        let wz = f32(gen_params.chunk_pos.z * 32) + f32(id.z);

        let scale_base = 0.005;
        let h_base = noise2d(vec2<f32>(wx, wz) * scale_base) * 40.0 + 60.0;
        let h_detail = noise2d(vec2<f32>(wx, wz) * 0.05) * 5.0;
        let height = h_base + h_detail;

        let cave_n = noise(vec3<f32>(wx, wy, wz) * 0.04);
        var blk = 0u;

        if (wy <= height) {{
            if (cave_n > 0.65) {{ blk = 0u; }}
            else {{
                if (wy >= height - 1.0) {{ blk = {}; }}
                else if (wy >= height - 4.0) {{ blk = {}; }}
                else {{ blk = {}; }}
            }}
        }} else if (wy < 55.0) {{
            blk = {};
        }}

        let idx = id.x + id.z * 32u + id.y * 1024u;
        block_buffer[idx] = blk;
    }}
    "#,
        id_grass, id_dirt, id_stone, id_water
    ));

    s.push_str(
        r#"
    struct VertexInput {
        @location(0) pos: vec3<f32>,
        @location(1) norm: vec3<f32>,
        @location(2) uv: vec2<f32>,
        @location(3) bounds: vec4<f32>,
        @location(4) color: vec4<f32>,
    };
    struct VertexOutput {
        @builtin(position) clip_pos: vec4<f32>,
        @location(0) uv: vec2<f32>,
        @location(1) bounds: vec4<f32>,
        @location(2) color: vec4<f32>,
        @location(3) world_pos: vec3<f32>,
        @location(4) norm: vec3<f32>,
    };

    @vertex
    fn vs_main(in: VertexInput) -> VertexOutput {
        var out: VertexOutput;
        let wp = vec4<f32>(in.pos, 1.0);
        out.clip_pos = camera.view_proj * wp;
        out.uv = in.uv;
        out.bounds = in.bounds;
        out.color = in.color;
        out.world_pos = in.pos;
        out.norm = in.norm;
        return out;
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
        let min_uv = in.bounds.xy;
        let max_uv = in.bounds.zw;
        let tile_size = max_uv - min_uv;
        let tiled_uv = min_uv + fract(in.uv) * tile_size;
        let tex = textureSample(t_diffuse, s_diffuse, tiled_uv);
        if (tex.a < 0.5) { discard; }

        let sun_dir = normalize(vec3<f32>(0.2, 0.8, 0.3));
        let diffuse = max(dot(in.norm, sun_dir), 0.2);
        let light = min(diffuse + 0.4, 1.0);

        let dist = distance(in.world_pos, camera.camera_pos);
        let fog = 1.0 - exp(-dist * 0.0015);
        let sky_col = vec3<f32>(0.6, 0.7, 0.9);

        let final_rgb = mix(tex.rgb * in.color.rgb * light, sky_col, fog);
        return vec4<f32>(final_rgb, 1.0);
    }

    struct SkyOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }
    @vertex fn vs_sky(@builtin(vertex_index) idx: u32) -> SkyOut {
        var out: SkyOut;
        var uvs = array<vec2<f32>,3>(vec2<f32>(0.,0.), vec2<f32>(2.,0.), vec2<f32>(0.,2.));
        out.uv = uvs[idx];
        out.pos = vec4<f32>(out.uv * 2.0 - 1.0, 0.0, 1.0);
        return out;
    }

    @fragment
    fn fs_sky(in: SkyOut) -> @location(0) vec4<f32> {
        let ndc = vec4<f32>(in.uv * 2.0 - 1.0, 1.0, 1.0);
        let view_dir = normalize((camera.inv_view_proj * ndc).xyz - camera.camera_pos);

        let depth_raw = textureLoad(t_depth, vec2<i32>(in.pos.xy), 0);

        let z_n = 2.0 * depth_raw - 1.0;
        let z_e = 2.0 * 0.1 * 4000.0 / (4000.0 + 0.1 - z_n * (4000.0 - 0.1));
        let scene_dist = select(z_e, 10000.0, depth_raw == 1.0);

        var color = vec3<f32>(0.6, 0.7, 0.9); // Base Sky
        var alpha = 1.0;

        if (depth_raw < 1.0) {
            color = vec3<f32>(0.0);
            alpha = 0.0;
        }

        let cloud_y = 140.0;
        if (camera.camera_pos.y < cloud_y && view_dir.y > 0.0) {
            let t = (cloud_y - camera.camera_pos.y) / view_dir.y;

            if (t < scene_dist && t < 2000.0) {
                let steps = 8; // Lightweight loop
                let t_limit = min(scene_dist, 2000.0);
                let step_size = max(0.0, t_limit - t) / f32(steps);

                var cur_t = t;
                for (var i = 0; i < steps; i++) {
                    let pos = camera.camera_pos + view_dir * cur_t;
                    let den = noise2d(pos.xz * 0.01 + camera.time * 0.02);

                    if (den > 0.5) {
                        let cloud_val = smoothstep(0.5, 0.8, den);
                        let soft = smoothstep(0.0, 30.0, scene_dist - cur_t);
                        let cloud_col = vec3<f32>(1.0);
                        let contrib = cloud_val * 0.6 * soft * (1.0 / f32(steps));

                        if (depth_raw >= 1.0) {
                            color = mix(color, cloud_col, contrib);
                        } else {
                            color = mix(color, cloud_col, contrib);
                            alpha = max(alpha, contrib);
                        }
                    }
                    cur_t += step_size;
                }
            }
        }

        return vec4<f32>(color, alpha);
    }
    "#,
    );

    s
}
