use wgpu::util::DeviceExt;

pub const CHUNK_SIZE: usize = 32;

#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum BlockType {
    Air = 0,
    Grass = 1,
    Dirt = 2,
    Stone = 3,
    Sand = 4,
    Snow = 5,
}

impl BlockType {
    pub fn get_color(&self) -> [f32; 3] {
        match self {
            BlockType::Grass => [0.1, 0.6, 0.2],
            BlockType::Dirt => [0.4, 0.25, 0.15],
            BlockType::Stone => [0.5, 0.5, 0.5],
            BlockType::Sand => [0.8, 0.8, 0.5],
            BlockType::Snow => [0.9, 0.95, 1.0],
            _ => [1.0, 0.0, 1.0],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VoxelVertex {
    pub pos: [f32; 3],
    pub color: [f32; 3],
    pub normal: [f32; 3],
}

pub struct Chunk {
    pub position: [i32; 3],
    pub blocks: Box<[[[u8; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE]>,
    pub vertex_buffer: Option<wgpu::Buffer>,
    pub index_buffer: Option<wgpu::Buffer>,
    pub index_count: u32,
}

impl Chunk {
    pub fn new(position: [i32; 3]) -> Self {
        Self {
            position,
            blocks: Box::new([[[0; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE]),
            vertex_buffer: None,
            index_buffer: None,
            index_count: 0,
        }
    }

    pub fn generate_mesh(&mut self, device: &wgpu::Device) {
        let mut vertices: Vec<VoxelVertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut idx = 0;

        let wx = self.position[0] as f32 * CHUNK_SIZE as f32;
        let wy = self.position[1] as f32 * CHUNK_SIZE as f32;
        let wz = self.position[2] as f32 * CHUNK_SIZE as f32;

        for x in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for z in 0..CHUNK_SIZE {
                    let b = self.blocks[x][y][z];
                    if b == 0 {
                        continue;
                    }

                    let color = match b {
                        1 => BlockType::Grass.get_color(),
                        2 => BlockType::Dirt.get_color(),
                        3 => BlockType::Stone.get_color(),
                        4 => BlockType::Sand.get_color(),
                        5 => BlockType::Snow.get_color(),
                        _ => [1.0, 1.0, 1.0],
                    };

                    let fx = wx + x as f32;
                    let fy = wy + y as f32;
                    let fz = wz + z as f32;

                    // Macro to push quad (2 triangles)
                    let mut push_quad =
                        |p1: [f32; 3], p2: [f32; 3], p3: [f32; 3], p4: [f32; 3], norm: [f32; 3]| {
                            vertices.push(VoxelVertex {
                                pos: p1,
                                color,
                                normal: norm,
                            });
                            vertices.push(VoxelVertex {
                                pos: p2,
                                color,
                                normal: norm,
                            });
                            vertices.push(VoxelVertex {
                                pos: p3,
                                color,
                                normal: norm,
                            });
                            vertices.push(VoxelVertex {
                                pos: p4,
                                color,
                                normal: norm,
                            });

                            // 0, 1, 2, 2, 1, 3
                            indices.extend_from_slice(&[
                                idx,
                                idx + 1,
                                idx + 2,
                                idx + 2,
                                idx + 1,
                                idx + 3,
                            ]);
                            idx += 4;
                        };

                    // Top (Y+)
                    if y == CHUNK_SIZE - 1 || self.blocks[x][y + 1][z] == 0 {
                        push_quad(
                            [fx, fy + 1.0, fz],
                            [fx + 1.0, fy + 1.0, fz],
                            [fx, fy + 1.0, fz + 1.0],
                            [fx + 1.0, fy + 1.0, fz + 1.0],
                            [0.0, 1.0, 0.0],
                        );
                    }
                    // Bottom (Y-)
                    if y == 0 || self.blocks[x][y - 1][z] == 0 {
                        push_quad(
                            [fx, fy, fz + 1.0],
                            [fx + 1.0, fy, fz + 1.0],
                            [fx, fy, fz],
                            [fx + 1.0, fy, fz],
                            [0.0, -1.0, 0.0],
                        );
                    }
                    // Front (Z+)
                    if z == CHUNK_SIZE - 1 || self.blocks[x][y][z + 1] == 0 {
                        push_quad(
                            [fx, fy, fz + 1.0],
                            [fx + 1.0, fy, fz + 1.0],
                            [fx, fy + 1.0, fz + 1.0],
                            [fx + 1.0, fy + 1.0, fz + 1.0],
                            [0.0, 0.0, 1.0],
                        );
                    }
                    // Back (Z-)
                    if z == 0 || self.blocks[x][y][z - 1] == 0 {
                        push_quad(
                            [fx + 1.0, fy, fz],
                            [fx, fy, fz],
                            [fx + 1.0, fy + 1.0, fz],
                            [fx, fy + 1.0, fz],
                            [0.0, 0.0, -1.0],
                        );
                    }
                    // Right (X+)
                    if x == CHUNK_SIZE - 1 || self.blocks[x + 1][y][z] == 0 {
                        push_quad(
                            [fx + 1.0, fy, fz + 1.0],
                            [fx + 1.0, fy, fz],
                            [fx + 1.0, fy + 1.0, fz + 1.0],
                            [fx + 1.0, fy + 1.0, fz],
                            [1.0, 0.0, 0.0],
                        );
                    }
                    // Left (X-)
                    if x == 0 || self.blocks[x - 1][y][z] == 0 {
                        push_quad(
                            [fx, fy, fz],
                            [fx, fy, fz + 1.0],
                            [fx, fy + 1.0, fz],
                            [fx, fy + 1.0, fz + 1.0],
                            [-1.0, 0.0, 0.0],
                        );
                    }
                }
            }
        }

        if !vertices.is_empty() {
            self.vertex_buffer = Some(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("Chunk VB"),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                },
            ));
            self.index_buffer = Some(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("Chunk IB"),
                    contents: bytemuck::cast_slice(&indices),
                    usage: wgpu::BufferUsages::INDEX,
                },
            ));
            self.index_count = indices.len() as u32;
        }
    }
}
