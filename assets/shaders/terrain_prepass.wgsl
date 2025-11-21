// assets/shaders/terrain_prepass.wgsl
#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_clip}

// Re-declare the bindings because this is a separate shader entry point
@group(1) @binding(100) var terrain_atlas_texture: texture_2d<f32>;
@group(1) @binding(101) var terrain_atlas_sampler: sampler;

struct TerrainVertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(4) tangent: vec4<f32>, // UV Bounds
    @location(5) color: vec4<f32>,
};

struct PrepassVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) tangent: vec4<f32>,
    @location(2) color: vec4<f32>,
};

@vertex
fn vertex(vertex: TerrainVertex) -> PrepassVertexOutput {
    var out: PrepassVertexOutput;
    let model = get_world_from_local(vertex.instance_index);
    out.clip_position = mesh_position_local_to_clip(model, vec4<f32>(vertex.position, 1.0));
    out.uv = vertex.uv;
    out.tangent = vertex.tangent;
    out.color = vertex.color;
    return out;
}

@fragment
fn fragment(in: PrepassVertexOutput) {
    // --- COPY OF UV LOGIC ---
    let bounds = in.tangent;
    let min_uv = bounds.xy;
    let max_uv = bounds.zw;
    let tile_size = max_uv - min_uv;

    let tiled_uv = min_uv + fract(in.uv) * tile_size;

    // Sample texture
    let texture_color = textureSample(terrain_atlas_texture, terrain_atlas_sampler, tiled_uv);

    // Apply Vertex Color (important if you tint transparency)
    let alpha = texture_color.a * in.color.a;

    // --- DISCARD IN PREPASS ---
    // This ensures the depth buffer has "holes" matching the texture
    if (alpha < 0.5) {
        discard;
    }
}
