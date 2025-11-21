#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
    pbr_types::STANDARD_MATERIAL_FLAGS_UNLIT_BIT,
}

#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_clip, mesh_normal_local_to_world}

const COLOR_MULTIPLIER: vec4<f32> = vec4<f32>(1.0, 1.0, 1.0, 1.0);

// Custom bindings for TerrainExtension.
@group(#{MATERIAL_BIND_GROUP}) @binding(100) var terrain_atlas_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(101) var terrain_atlas_sampler: sampler;

struct TerrainVertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    // We use the tangent attribute to pass the UV bounds (min_u, min_v, max_u, max_v)
    @location(4) tangent: vec4<f32>,
    @location(5) color: vec4<f32>,
};

@vertex
fn vertex(vertex: TerrainVertex) -> VertexOutput {
    var out: VertexOutput;

    let model = get_world_from_local(vertex.instance_index);

    out.world_position = model * vec4<f32>(vertex.position, 1.0);
    out.position = mesh_position_local_to_clip(model, vec4<f32>(vertex.position, 1.0));
    out.world_normal = mesh_normal_local_to_world(vertex.normal, vertex.instance_index);
    out.uv = vertex.uv;

    // Pass the tangent (which holds our UV bounds) through to the fragment shader.
    // StandardMaterial expects actual tangents here for normal mapping, but since
    // we aren't using a normal map, we can safely repurpose this slot.
    out.world_tangent = vertex.tangent;

    out.color = vertex.color;

    return out;
}

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    // Initialize standard PBR input.
    // This sets up roughness, metallic, occlusion, etc. from the StandardMaterial config.
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    // --- Custom Atlas Tiling Logic ---
    // Retrieve the UV bounds we passed via the tangent attribute
    let bounds = in.world_tangent;
    let min_uv = bounds.xy;
    let max_uv = bounds.zw;
    let tile_size = max_uv - min_uv;

    // Calculate the tiled UVs.
    // fract(in.uv) ensures the texture repeats within the bounds defined for this specific face.
    let tiled_uv = min_uv + fract(in.uv) * tile_size;

    // Manual derivative calculation for smooth mipmapping across tile boundaries.
    let dx = dpdx(in.uv * tile_size);
    let dy = dpdy(in.uv * tile_size);

    // Sample our custom atlas texture
    let texture_color = textureSampleGrad(
        terrain_atlas_texture,
        terrain_atlas_sampler,
        tiled_uv,
        dx,
        dy
    );

    // Override the base color with our sampled texture color.
    // We multiply by the vertex color (used for biome tinting).
    pbr_input.material.base_color = texture_color * in.color * COLOR_MULTIPLIER;

    // Handle alpha cutout (transparency)
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

    // Apply standard PBR lighting using the modified input
    var out: FragmentOutput;
    if (pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_UNLIT_BIT) == 0u {
        out.color = apply_pbr_lighting(pbr_input);
    } else {
        out.color = pbr_input.material.base_color;
    }

    out.color = main_pass_post_lighting_processing(pbr_input, out.color);

    return out;
}