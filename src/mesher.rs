//! Optimized Greedy Mesher ported for WGPU.
//! Outputs vertices with Position, Normal, UV, Bounds (for tiling), and Color.

use crate::chunk::{CHUNK_HEIGHT, CHUNK_SIZE, Chunk};
use crate::data::{BlockGeometry, GameRegistry};
use crate::texture::TextureAtlas;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct VoxelVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub bounds: [f32; 4], // min_u, min_v, max_u, max_v
    pub color: [f32; 4],
}

const UV_EPSILON: f32 = 0.0001;

struct MeshBuffers {
    vertices: Vec<VoxelVertex>,
    indices: Vec<u32>,
}

pub fn generate_mesh(
    chunk: &Chunk,
    registry: &GameRegistry,
    atlas: &TextureAtlas,
) -> (Vec<VoxelVertex>, Vec<u32>) {
    let mut buffers = MeshBuffers {
        vertices: Vec::new(),
        indices: Vec::new(),
    };

    greedy_cube_mesh(chunk, registry, atlas, &mut buffers);
    cross_mesh(chunk, registry, atlas, &mut buffers);

    (buffers.vertices, buffers.indices)
}

const FACE_NORMALS: [[i32; 3]; 6] = [
    [1, 0, 0],  // Right (X+)
    [-1, 0, 0], // Left  (X-)
    [0, 1, 0],  // Top   (Y+)
    [0, -1, 0], // Bottom(Y-)
    [0, 0, 1],  // Front (Z+)
    [0, 0, -1], // Back  (Z-)
];

fn greedy_cube_mesh(
    chunk: &Chunk,
    registry: &GameRegistry,
    atlas: &TextureAtlas,
    buffers: &mut MeshBuffers,
) {
    let air = registry.get_block_def(0).unwrap();

    for face in 0..6 {
        let axis = face / 2;
        let dir = if face % 2 == 0 { 1 } else { -1 };
        let normal = FACE_NORMALS[face];
        let normal_f32 = [normal[0] as f32, normal[1] as f32, normal[2] as f32];

        let (u_axis, v_axis) = match axis {
            0 => (2, 1), // X face -> Z, Y axes
            1 => (0, 2), // Y face -> X, Z axes
            2 => (0, 1), // Z face -> X, Y axes
            _ => unreachable!(),
        };

        let i_limit = [CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE][axis];
        let j_limit = [CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE][u_axis];
        let k_limit = [CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE][v_axis];

        let mut mask = vec![None; (j_limit * k_limit) as usize];

        for i in 0..i_limit {
            // 1. Build Mask
            let mut n = 0;
            for k in 0..k_limit {
                for j in 0..j_limit {
                    let mut pos = [0, 0, 0];
                    pos[axis] = i;
                    pos[u_axis] = j;
                    pos[v_axis] = k;

                    let mut adj = pos;
                    adj[axis] += dir;

                    let blk_curr = chunk.get_block(pos[0], pos[1], pos[2]);
                    let blk_adj = if adj[0] >= 0
                        && adj[0] < CHUNK_SIZE
                        && adj[1] >= 0
                        && adj[1] < CHUNK_HEIGHT
                        && adj[2] >= 0
                        && adj[2] < CHUNK_SIZE
                    {
                        chunk.get_block(adj[0], adj[1], adj[2])
                    } else {
                        0
                    };

                    let def_curr = registry.get_block_def(blk_curr).unwrap_or(air);
                    let def_adj = registry.get_block_def(blk_adj).unwrap_or(air);

                    let visible = if def_curr.is_transparent {
                        def_curr.numeric_id != def_adj.numeric_id && def_adj.is_transparent
                    } else {
                        def_adj.is_transparent
                    };

                    if visible && def_curr.geometry == BlockGeometry::Cube {
                        mask[n] = Some(blk_curr);
                    } else {
                        mask[n] = None;
                    }
                    n += 1;
                }
            }

            // 2. Mesh Mask
            let mut n = 0;
            for k in 0..k_limit {
                for j in 0..j_limit {
                    if let Some(blk_id) = mask[n] {
                        let mut width = 1;
                        while j + width < j_limit && mask[n + width as usize] == Some(blk_id) {
                            width += 1;
                        }

                        let mut height = 1;
                        let mut done = false;
                        while k + height < k_limit {
                            for w in 0..width {
                                if mask[n + w as usize + (height * j_limit) as usize]
                                    != Some(blk_id)
                                {
                                    done = true;
                                    break;
                                }
                            }
                            if done {
                                break;
                            }
                            height += 1;
                        }

                        let def = registry.get_block_def(blk_id).unwrap();
                        if let Some(tex) = def.get_texture_for_face(axis, dir)
                            && let Some(uv_rect) = atlas.get_uv(tex)
                        {
                            let w_f = width as f32;
                            let h_f = height as f32;

                            let mut du = [0.0, 0.0, 0.0];
                            du[u_axis] = w_f;
                            let mut dv = [0.0, 0.0, 0.0];
                            dv[v_axis] = h_f;

                            let mut base = [0.0, 0.0, 0.0];
                            base[axis] = i as f32 + if dir == 1 { 1.0 } else { 0.0 };
                            base[u_axis] = j as f32;
                            base[v_axis] = k as f32;

                            let cx = chunk.position[0] as f32 * CHUNK_SIZE as f32;
                            let cz = chunk.position[1] as f32 * CHUNK_SIZE as f32;
                            base[0] += cx;
                            base[2] += cz;

                            let v0 = base;
                            let v1 = [base[0] + du[0], base[1] + du[1], base[2] + du[2]];
                            let v2 = [
                                base[0] + du[0] + dv[0],
                                base[1] + du[1] + dv[1],
                                base[2] + du[2] + dv[2],
                            ];
                            let v3 = [base[0] + dv[0], base[1] + dv[1], base[2] + dv[2]];

                            // 2. Assign vertices to Logical Quad Slots (BL, BR, TR, TL) for Texture Orientation.
                            // We choose these permutations to ensure the texture is upright and not mirrored.
                            let (p_bl, p_br, p_tr, p_tl) = match (axis, dir) {
                                (0, 1) => (v0, v1, v2, v3),  // Right (X+): Standard (Texture Correct)
                                (0, -1) => (v1, v0, v3, v2), // Left  (X-): Mirrored (Texture Correct)
                                (1, 1) => (v3, v2, v1, v0), // Top   (Y+): Rotated/Flipped (Texture Correct-ish)
                                (1, -1) => (v0, v1, v2, v3), // Bottom(Y-): Standard (Texture Correct)
                                (2, 1) => (v0, v1, v2, v3), // Front (Z+): Standard (Texture Correct)
                                (2, -1) => (v1, v0, v3, v2), // Back  (Z-): Mirrored (Texture Correct)
                                _ => unreachable!(),
                            };

                            let bounds = [
                                uv_rect.min[0] + UV_EPSILON,
                                uv_rect.min[1] + UV_EPSILON,
                                uv_rect.max[0] - UV_EPSILON,
                                uv_rect.max[1] - UV_EPSILON,
                            ];

                            let tint = if def.needs_biome_tint_for_face(axis, dir) {
                                [0.4, 0.8, 0.4, 1.0]
                            } else {
                                [1.0; 4]
                            };

                            let idx = buffers.vertices.len() as u32;

                            // 3. Push Vertices
                            buffers.vertices.push(VoxelVertex {
                                position: p_bl,
                                normal: normal_f32,
                                uv: [0.0, h_f],
                                bounds,
                                color: tint,
                            });
                            buffers.vertices.push(VoxelVertex {
                                position: p_br,
                                normal: normal_f32,
                                uv: [w_f, h_f],
                                bounds,
                                color: tint,
                            });
                            buffers.vertices.push(VoxelVertex {
                                position: p_tr,
                                normal: normal_f32,
                                uv: [w_f, 0.0],
                                bounds,
                                color: tint,
                            });
                            buffers.vertices.push(VoxelVertex {
                                position: p_tl,
                                normal: normal_f32,
                                uv: [0.0, 0.0],
                                bounds,
                                color: tint,
                            });

                            // 4. Push Indices
                            // The X-axis (Axis 0) naturally generates normals pointing INWARDS due to the coordinate winding (Z x Y = -X).
                            // We flip the indices for Axis 0 to point them OUTWARDS.
                            // Y and Z axes generate correct normals with the standard winding.
                            if axis == 0 {
                                // Flip Winding (0 -> 2 -> 1)
                                buffers.indices.extend_from_slice(&[
                                    idx,
                                    idx + 2,
                                    idx + 1,
                                    idx,
                                    idx + 3,
                                    idx + 2,
                                ]);
                            } else {
                                // Standard Winding (0 -> 1 -> 2)
                                buffers.indices.extend_from_slice(&[
                                    idx,
                                    idx + 1,
                                    idx + 2,
                                    idx,
                                    idx + 2,
                                    idx + 3,
                                ]);
                            }

                            // end
                        }

                        for h in 0..height {
                            for w in 0..width {
                                mask[n + w as usize + (h * j_limit) as usize] = None;
                            }
                        }
                    }
                    n += 1;
                }
            }
        }
    }
}

fn cross_mesh(
    chunk: &Chunk,
    registry: &GameRegistry,
    atlas: &TextureAtlas,
    buffers: &mut MeshBuffers,
) {
    for y in 0..CHUNK_HEIGHT {
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let id = chunk.get_block(x, y, z);
                if id == 0 {
                    continue;
                }
                let def = registry.get_block_def(id).unwrap();
                if def.geometry == BlockGeometry::Cross
                    && let Some(tex) = def.get_texture_for_face(0, 0)
                    && let Some(rect) = atlas.get_uv(tex)
                {
                    let cx = (chunk.position[0] * CHUNK_SIZE + x) as f32;
                    let cz = (chunk.position[1] * CHUNK_SIZE + z) as f32;
                    let cy = y as f32;

                    let tint = if def.needs_biome_tint {
                        [0.4, 0.8, 0.4, 1.0]
                    } else {
                        [1.0; 4]
                    };
                    let bounds = [rect.min[0], rect.min[1], rect.max[0], rect.max[1]];
                    let idx = buffers.vertices.len() as u32;

                    buffers.vertices.push(VoxelVertex {
                        position: [cx, cy, cz],
                        normal: [0.0, 1.0, 0.0],
                        uv: [0.0, 1.0],
                        bounds,
                        color: tint,
                    });
                    buffers.vertices.push(VoxelVertex {
                        position: [cx + 1.0, cy, cz + 1.0],
                        normal: [0.0, 1.0, 0.0],
                        uv: [1.0, 1.0],
                        bounds,
                        color: tint,
                    });
                    buffers.vertices.push(VoxelVertex {
                        position: [cx + 1.0, cy + 1.0, cz + 1.0],
                        normal: [0.0, 1.0, 0.0],
                        uv: [1.0, 0.0],
                        bounds,
                        color: tint,
                    });
                    buffers.vertices.push(VoxelVertex {
                        position: [cx, cy + 1.0, cz],
                        normal: [0.0, 1.0, 0.0],
                        uv: [0.0, 0.0],
                        bounds,
                        color: tint,
                    });

                    buffers.indices.extend_from_slice(&[
                        idx,
                        idx + 1,
                        idx + 2,
                        idx,
                        idx + 2,
                        idx + 3,
                    ]);
                    buffers.indices.extend_from_slice(&[
                        idx,
                        idx + 3,
                        idx + 2,
                        idx,
                        idx + 2,
                        idx + 1,
                    ]);
                }
            }
        }
    }
}
