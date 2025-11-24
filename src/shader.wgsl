// --- CONSTANTS ---
const CHUNK_SIZE: u32 = 64u;
const TOTAL_VOXELS_INTS: u32 = 65536u; // 64^3 / 4

struct GlobalUniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    time: f32,
};

struct ChunkUniforms {
    offset: vec3<i32>,
};

struct IndirectDrawArgs {
    index_count: atomic<u32>,
    instance_count: atomic<u32>,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
};

struct MeshMetadata {
    vertex_counter: atomic<u32>,
    index_counter: atomic<u32>,
};

struct PackedVertex {
    pos_data: u32,
    color_data: u32,
};

struct AtomicVoxelStorage {
    data: array<atomic<u32>>,
};

@group(0) @binding(0) var<uniform> global: GlobalUniforms;
@group(1) @binding(0) var<uniform> chunk_data: ChunkUniforms;

@group(2) @binding(0) var<storage, read_write> atomic_voxels: AtomicVoxelStorage;
@group(2) @binding(1) var<storage, read_write> draw_args: IndirectDrawArgs;
@group(2) @binding(2) var<storage, read_write> mesh_meta: MeshMetadata;

@group(3) @binding(0) var<storage, read_write> out_vertices: array<PackedVertex>;
@group(3) @binding(1) var<storage, read_write> out_indices: array<u32>;

// --- HELPERS ---
fn get_index(x: u32, y: u32, z: u32) -> u32 {
    return x + (z * CHUNK_SIZE) + (y * CHUNK_SIZE * CHUNK_SIZE);
}

fn set_voxel_atomic(x: u32, y: u32, z: u32, val: u32) {
    let idx = get_index(x, y, z);
    let array_idx = idx / 4u;
    let sub_idx = (idx % 4u) * 8u;
    let set_val = (val & 0xFFu) << sub_idx;
    atomicOr(&atomic_voxels.data[array_idx], set_val);
}

fn get_voxel_atomic(x: u32, y: u32, z: u32) -> u32 {
    if (x >= CHUNK_SIZE || y >= CHUNK_SIZE || z >= CHUNK_SIZE) { return 0u; }
    let idx = get_index(x, y, z);
    let array_idx = idx / 4u;
    let sub_idx = (idx % 4u) * 8u;
    let packed_val = atomicLoad(&atomic_voxels.data[array_idx]);
    return (packed_val >> sub_idx) & 0xFFu;
}

// --- CLEAR (Fast Reset) ---
@compute @workgroup_size(256)
fn clear_voxels(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= TOTAL_VOXELS_INTS) { return; }
    atomicStore(&atomic_voxels.data[idx], 0u);
}

// --- NOISE ---
fn hash3(p: vec3<u32>) -> f32 {
    var p3 = fract(vec3<f32>(p) * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn noise(x: vec3<f32>) -> f32 {
    let p = floor(x);
    let f = fract(x);
    let f2 = f*f*(3.0-2.0*f);
    return mix(mix(mix( hash3(vec3<u32>(p)), hash3(vec3<u32>(p + vec3(1.0,0.0,0.0))),f2.x),
                   mix( hash3(vec3<u32>(p + vec3(0.0,1.0,0.0))), hash3(vec3<u32>(p + vec3(1.0,1.0,0.0))),f2.x),f2.y),
               mix(mix( hash3(vec3<u32>(p + vec3(0.0,0.0,1.0))), hash3(vec3<u32>(p + vec3(1.0,0.0,1.0))),f2.x),
                   mix( hash3(vec3<u32>(p + vec3(0.0,1.0,1.0))), hash3(vec3<u32>(p + vec3(1.0,1.0,1.0))),f2.x),f2.y),f2.z);
}

fn fbm(p: vec3<f32>) -> f32 {
    var f = 0.0;
    f += 0.5000 * noise(p);
    f += 0.2500 * noise(p * 2.0);
    f += 0.1250 * noise(p * 4.0);
    return f;
}

// --- GENERATION ---
@compute @workgroup_size(8, 8, 8)
fn generate(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;

    if (x >= CHUNK_SIZE || y >= CHUNK_SIZE || z >= CHUNK_SIZE) { return; }

    let world_pos = vec3<f32>(f32(x), f32(y), f32(z)) + vec3<f32>(chunk_data.offset);
    let noise_pos = world_pos + vec3<f32>(100000.0);

    var block_type = 0u;
    let scale = 0.015;
    let height_val = fbm(vec3<f32>(noise_pos.x * scale, noise_pos.z * scale, 0.0)) * 64.0 + 20.0;
    let cave_val = fbm(noise_pos * 0.05 + vec3(23.0));

    if (world_pos.y < height_val) {
        if (cave_val > 0.55) { block_type = 0u; }
        else {
            if (world_pos.y < 5.0) { block_type = 4u; }
            else if (world_pos.y > height_val - 3.0) {
                if (world_pos.y > height_val - 1.0) { block_type = 1u; }
                else { block_type = 2u; }
            } else { block_type = 3u; }
        }
    }

    if (y == 0u) { block_type = 1u; }

    if (block_type != 0u) {
        set_voxel_atomic(x, y, z, block_type);
    }
}

// --- MESHING ---
@compute @workgroup_size(8, 8, 8)
fn mesh(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;

    if (x >= CHUNK_SIZE || y >= CHUNK_SIZE || z >= CHUNK_SIZE) { return; }

    let block = get_voxel_atomic(x, y, z);
    if (block == 0u) { return; }

    let dirs = array<vec3<i32>, 6>(
        vec3(1, 0, 0), vec3(-1, 0, 0),
        vec3(0, 1, 0), vec3(0, -1, 0),
        vec3(0, 0, 1), vec3(0, 0, -1)
    );

    for (var i = 0; i < 6; i++) {
        let n = dirs[i];
        let nx = u32(i32(x) + n.x);
        let ny = u32(i32(y) + n.y);
        let nz = u32(i32(z) + n.z);

        if (get_voxel_atomic(nx, ny, nz) == 0u) {
            let v_idx = atomicAdd(&mesh_meta.vertex_counter, 4u);
            let i_idx = atomicAdd(&draw_args.index_count, 6u);

            var dx = vec3<f32>(0.0); var dy = vec3<f32>(0.0);
            if (abs(n.y) > 0) { dx = vec3(0.0, 0.0, 1.0); dy = vec3(1.0, 0.0, 0.0); }
            else if (abs(n.x) > 0) { dx = vec3(0.0, 1.0, 0.0); dy = vec3(0.0, 0.0, 1.0); }
            else { dx = vec3(1.0, 0.0, 0.0); dy = vec3(0.0, 1.0, 0.0); }

            let center = vec3<f32>(f32(x), f32(y), f32(z)) + vec3<f32>(0.5) + vec3<f32>(n) * 0.5;

            // Colors
            var c = vec4<f32>(1.0);
            if (block == 1u) { c = vec4(0.2, 0.8, 0.2, 1.0); }
            else if (block == 2u) { c = vec4(0.5, 0.35, 0.1, 1.0); }
            else if (block == 3u) { c = vec4(0.6, 0.6, 0.6, 1.0); }
            else if (block == 4u) { c = vec4(0.1, 0.1, 0.1, 1.0); }
            let r = u32(c.r * 255.0);
            let g = u32(c.g * 255.0);
            let b = u32(c.b * 255.0);
            let a = u32(c.a * 255.0);
            let color_u32 = r | (g << 8u) | (b << 16u) | (a << 24u);

            let ao_byte = 255u;

            let offsets = array<vec2<f32>, 4>(
                vec2(-0.5, -0.5), vec2(0.5, -0.5), vec2(0.5, 0.5), vec2(-0.5, 0.5)
            );

            for (var k = 0; k < 4; k++) {
                let pos = center + dx * offsets[k].x + dy * offsets[k].y;
                let px = u32(pos.x + 0.1);
                let py = u32(pos.y + 0.1);
                let pz = u32(pos.z + 0.1);
                let packed_pos = px | (py << 8u) | (pz << 16u) | (ao_byte << 24u);
                out_vertices[v_idx + u32(k)] = PackedVertex(packed_pos, color_u32);
            }

            if (n.x > 0 || n.y > 0 || n.z > 0) {
                out_indices[i_idx + 0u] = v_idx + 0u;
                out_indices[i_idx + 1u] = v_idx + 1u;
                out_indices[i_idx + 2u] = v_idx + 2u;
                out_indices[i_idx + 3u] = v_idx + 0u;
                out_indices[i_idx + 4u] = v_idx + 2u;
                out_indices[i_idx + 5u] = v_idx + 3u;
            } else {
                out_indices[i_idx + 0u] = v_idx + 0u;
                out_indices[i_idx + 1u] = v_idx + 2u;
                out_indices[i_idx + 2u] = v_idx + 1u;
                out_indices[i_idx + 3u] = v_idx + 0u;
                out_indices[i_idx + 4u] = v_idx + 3u;
                out_indices[i_idx + 5u] = v_idx + 2u;
            }
        }
    }
}

// --- SKY SHADER ---
struct SkyOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_sky(@builtin(vertex_index) vertex_index: u32) -> SkyOutput {
    var pos = array<vec2<f32>, 3>(
        vec2(-1.0, -3.0),
        vec2(-1.0, 1.0),
        vec2(3.0, 1.0)
    );
    let xy = pos[vertex_index];
    var out: SkyOutput;
    out.clip_position = vec4<f32>(xy, 1.0, 1.0);
    out.uv = xy;
    return out;
}

@fragment
fn fs_sky(in: SkyOutput) -> @location(0) vec4<f32> {
    let clip = vec4<f32>(in.uv, 1.0, 1.0);
    let world_pos_h = global.inv_view_proj * clip;
    let world_pos = world_pos_h.xyz / world_pos_h.w;
    let dir = normalize(world_pos - global.camera_pos.xyz);

    let top_color = vec3<f32>(0.2, 0.4, 0.8);
    let horizon_color = vec3<f32>(0.8, 0.9, 1.0);
    let bottom_color = vec3<f32>(0.1, 0.1, 0.1);

    var t = dir.y;
    var sky = mix(horizon_color, top_color, pow(max(t, 0.0), 0.5));
    sky = mix(sky, bottom_color, max(-t, 0.0));

    let sun_dir = normalize(vec3<f32>(0.3, 0.6, 0.5));
    let sun = pow(max(dot(dir, sun_dir), 0.0), 200.0);
    let sun_color = vec3<f32>(1.0, 0.9, 0.6) * sun;

    return vec4<f32>(sky + sun_color, 1.0);
}

// --- VOXEL RENDER ---
struct VertexInput {
    @location(0) packed_pos: u32,
    @location(1) packed_color: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let lx = f32(in.packed_pos & 0xFFu);
    let ly = f32((in.packed_pos >> 8u) & 0xFFu);
    let lz = f32((in.packed_pos >> 16u) & 0xFFu);

    // Unpack Color
    let r = f32(in.packed_color & 0xFFu) / 255.0;
    let g = f32((in.packed_color >> 8u) & 0xFFu) / 255.0;
    let b = f32((in.packed_color >> 16u) & 0xFFu) / 255.0;
    let a = f32((in.packed_color >> 24u) & 0xFFu) / 255.0;

    let chunk_offset = vec3<f32>(f32(chunk_data.offset.x), f32(chunk_data.offset.y), f32(chunk_data.offset.z));
    let world_pos = vec4<f32>(vec3(lx, ly, lz) + chunk_offset, 1.0);

    out.clip_position = global.view_proj * world_pos;
    out.color = vec4(r, g, b, a);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
