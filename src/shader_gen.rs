use crate::data::GameRegistry;
use crate::shader;

pub fn generate_wgsl(registry: &GameRegistry) -> String {
    let mut s = String::new();

    s.push_str(shader::COMMON);
    s.push_str(shader::WEATHER);

    s.push_str("\n// --- BLOCK IDS ---\n");
    for (name, def) in &registry.blocks {
        let raw_name = name.to_uppercase().replace(":", "_");
        s.push_str(&format!(
            "const BLOCK_{}: u32 = {}u;\n",
            raw_name, def.numeric_id
        ));
    }

    // --- DYNAMIC COMPUTE SHADER ---
    let get_id = |lookup: &str| -> String {
        let key = format!("maricraft:{}", lookup);
        if registry.blocks.contains_key(&key) {
            format!("BLOCK_MARICRAFT_{}", lookup.to_uppercase())
        } else if lookup == "water" {
            "0u".to_string()
        } else {
            "1u".to_string()
        }
    };

    let b_grass = get_id("grass");
    let b_dirt = get_id("dirt");
    let b_stone = get_id("stone");
    let b_water = get_id("water");
    let b_bedrock = get_id("bedrock");
    let b_sand = get_id("sand");
    let b_snow = get_id("snow_block");
    let b_log = get_id("oak_log");
    let b_leaves = get_id("oak_leaves");

    // UPDATED: Uses get_terrain_height() from common.wgsl instead of hardcoded math
    s.push_str(&format!(
        r#"
    struct GenParams {{ chunk_pos: vec3<i32>, seed: u32 }}
    @group(0) @binding(0) var<uniform> gen_params: GenParams;
    @group(0) @binding(1) var<storage, read_write> block_buffer: array<u32>;

    @compute @workgroup_size(4, 4, 4)
    fn cs_generate(@builtin(global_invocation_id) id: vec3<u32>) {{
        if (id.x >= 64u || id.y >= 64u || id.z >= 64u) {{ return; }}

        let wx = f32(gen_params.chunk_pos.x * 64) + f32(id.x);
        let wy = f32(gen_params.chunk_pos.y * 64) + f32(id.y);
        let wz = f32(gen_params.chunk_pos.z * 64) + f32(id.z);

        let temp = noise2d(vec2<f32>(wx, wz) * 0.001);
        let humid = noise2d(vec2<f32>(wx, wz) * 0.001 + vec2<f32>(100.0, 0.0));

        let height = get_terrain_height(vec2<f32>(wx, wz));

        var blk = 0u;

        if (wy <= height) {{
            let cave = noise(vec3<f32>(wx, wy, wz) * 0.04);
            if (wy < 3.0) {{ blk = {}; }}
            else if (cave > 0.65) {{ blk = 0u; }}
            else {{
                if (temp > 0.6 && humid < 0.4) {{
                    if (wy >= height - 3.0) {{ blk = {}; }} else {{ blk = {}; }}
                }} else if (temp < 0.3) {{
                    if (wy >= height - 1.0) {{ blk = {}; }} else {{ blk = {}; }}
                }} else {{
                    if (wy >= height - 1.0) {{ blk = {}; }}
                    else if (wy >= height - 4.0) {{ blk = {}; }}
                    else {{ blk = {}; }}
                }}
            }}
        }} else if (wy < 50.0) {{
            blk = {};
        }}

        // Trees
        if (blk == 0u && wy > height && wy < height + 15.0) {{
             let tree = noise2d(vec2<f32>(wx, wz) * 0.05);
             if (temp > 0.3 && temp < 0.6 && tree > 0.75) {{
                 let tx = floor(wx/5.0)*5.0; let tz = floor(wz/5.0)*5.0;
                 if (distance(vec2<f32>(wx,wz), vec2<f32>(tx,tz)) < 0.1) {{
                     if (wy < height + 6.0) {{ blk = {}; }}
                 }}
                 let dist = distance(vec3<f32>(wx,wy,wz), vec3<f32>(tx, height+6.0, tz));
                 if (dist < 3.0 && dist > 0.5 && blk == 0u) {{ blk = {}; }}
             }}
        }}

        let idx = id.x + (id.z * 64u) + (id.y * 64u * 64u);
        block_buffer[idx] = blk;
    }}
    "#,
        b_bedrock,
        b_sand,
        b_stone,
        b_snow,
        b_dirt,
        b_grass,
        b_dirt,
        b_stone,
        b_water,
        b_log,
        b_leaves
    ));

    s.push_str(shader::VOXEL_RENDER);
    s.push_str(shader::SKY_RENDER);
    s.push_str(shader::RAIN);

    s
}
