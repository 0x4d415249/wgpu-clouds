// --- CONSTANTS ---
const CHUNK_SIZE: u32 = 64u;
const TOTAL_VOXELS: u32 = 262144u; // 64 * 64 * 64

// --- STRUCTS ---

struct VoxelStorage {
    data: array<u32>,
};

struct IndirectDrawArgs {
    vertex_count: atomic<u32>, // Index count
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
};

struct MeshMetadata {
    vertex_counter: atomic<u32>,
    index_counter: atomic<u32>,
};

struct Uniforms {
    view_proj: mat4x4<f32>,
    time: f32,
};

struct PackedVertex {
    position: vec4<f32>,
    color: vec4<f32>,
};

// --- BIND GROUPS ---

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var<storage, read_write> voxels: VoxelStorage;
@group(1) @binding(1) var<storage, read_write> draw_args: IndirectDrawArgs;
@group(1) @binding(2) var<storage, read_write> mesh_meta: MeshMetadata;

@group(2) @binding(0) var<storage, read_write> out_vertices: array<PackedVertex>;
@group(2) @binding(1) var<storage, read_write> out_indices: array<u32>;

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
    let n = p.x + p.y*57.0 + 113.0*p.z;
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

fn get_index(x: u32, y: u32, z: u32) -> u32 {
    return x + (z * CHUNK_SIZE) + (y * CHUNK_SIZE * CHUNK_SIZE);
}

@compute @workgroup_size(8, 8, 8)
fn generate(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;

    if (x >= CHUNK_SIZE || y >= CHUNK_SIZE || z >= CHUNK_SIZE) { return; }

    let index = get_index(x, y, z);
    var block_type = 0u;

    let pos = vec3<f32>(f32(x), f32(y), f32(z));
    let scale = 0.03;
    let height_val = fbm(vec3<f32>(f32(x)*scale, f32(z)*scale, uniforms.time * 0.01)) * f32(CHUNK_SIZE);
    let cave_val = fbm(pos * 0.05 + vec3(23.0));

    if (f32(y) < height_val) {
        if (cave_val > 0.45) {
             block_type = 0u;
        } else {
            if (f32(y) < 5.0) {
                block_type = 4u; // Bedrock
            } else if (f32(y) > height_val - 3.0) {
                if (f32(y) > height_val - 1.0) {
                     block_type = 1u; // Grass
                } else {
                     block_type = 2u; // Dirt
                }
            } else {
                block_type = 3u; // Stone
            }
        }
    }

    voxels.data[index] = block_type;
}

// --- MESHING ---

fn is_solid(x: u32, y: u32, z: u32) -> bool {
    if (x >= CHUNK_SIZE || y >= CHUNK_SIZE || z >= CHUNK_SIZE) { return false; }
    let idx = get_index(x, y, z);
    return voxels.data[idx] != 0u;
}

fn get_color(block: u32, normal_y: f32) -> vec4<f32> {
    if (block == 1u) { return vec4(0.2, 0.8, 0.2, 1.0); }
    if (block == 2u) { return vec4(0.5, 0.35, 0.1, 1.0); }
    if (block == 3u) { return vec4(0.6, 0.6, 0.6, 1.0); }
    if (block == 4u) { return vec4(0.1, 0.1, 0.1, 1.0); }
    return vec4(1.0, 0.0, 1.0, 1.0);
}

@compute @workgroup_size(8, 8, 8)
fn mesh(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;

    if (x >= CHUNK_SIZE || y >= CHUNK_SIZE || z >= CHUNK_SIZE) { return; }

    let my_idx = get_index(x, y, z);
    let block = voxels.data[my_idx];

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

        if (!is_solid(nx, ny, nz)) {
            let v_idx = atomicAdd(&mesh_meta.vertex_counter, 4u);
            let i_idx = atomicAdd(&draw_args.vertex_count, 6u);

            var dx = vec3<f32>(0.0); var dy = vec3<f32>(0.0);

            // FIX: Swapped logic for Y axis to fix inverted top/bottom faces
            if (abs(n.y) > 0) {
                dx = vec3(0.0, 0.0, 1.0); // Z
                dy = vec3(1.0, 0.0, 0.0); // X
            }
            else if (abs(n.x) > 0) { dx = vec3(0.0, 1.0, 0.0); dy = vec3(0.0, 0.0, 1.0); }
            else { dx = vec3(1.0, 0.0, 0.0); dy = vec3(0.0, 1.0, 0.0); }

            let center = vec3<f32>(f32(x), f32(y), f32(z)) + vec3<f32>(0.5) + vec3<f32>(n) * 0.5;
            let c = get_color(block, f32(n.y));
            let light = 0.5 + 0.5 * max(0.0, f32(n.y) * 0.6 + 0.4 + f32(n.x)*0.2);

            let v0_pos = center - dx * 0.5 - dy * 0.5;
            let v1_pos = center + dx * 0.5 - dy * 0.5;
            let v2_pos = center + dx * 0.5 + dy * 0.5;
            let v3_pos = center - dx * 0.5 + dy * 0.5;

            out_vertices[v_idx + 0u] = PackedVertex(vec4(v0_pos, light), c);
            out_vertices[v_idx + 1u] = PackedVertex(vec4(v1_pos, light), c);
            out_vertices[v_idx + 2u] = PackedVertex(vec4(v2_pos, light), c);
            out_vertices[v_idx + 3u] = PackedVertex(vec4(v3_pos, light), c);

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

// --- RENDER ---

struct VertexInput {
    @location(0) position: vec4<f32>,
    @location(1) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) ao: f32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = vec4<f32>(in.position.xyz, 1.0);
    out.clip_position = uniforms.view_proj * world_pos;
    out.color = in.color;
    out.ao = in.position.w;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let depth = in.clip_position.z;
    let fog = clamp(depth * 0.05, 0.0, 0.5);
    let base_color = in.color.rgb * in.ao;
    let final_color = mix(base_color, vec3(0.6, 0.8, 1.0), fog);
    return vec4<f32>(final_color, 1.0);
}
