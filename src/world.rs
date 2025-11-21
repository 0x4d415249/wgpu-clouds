use crate::chunk::{BlockType, CHUNK_SIZE, Chunk};
use std::collections::HashMap;
use wgpu::Device;

// Simple pseudo-random noise
fn noise2(x: f32, z: f32) -> f32 {
    let s = (x * 0.05).sin() + (z * 0.05).cos();
    let s2 = (x * 0.15 + 123.0).sin() * 0.5;
    s + s2
}

pub struct World {
    pub chunks: HashMap<[i32; 3], Chunk>,
}

impl World {
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
        }
    }

    pub fn generate_area(&mut self, device: &Device, radius: i32) {
        // Generate a area of chunks for testing
        for x in -radius..radius {
            for z in -radius..radius {
                for y in 0..2 {
                    // Vertical chunks
                    let pos = [x, y, z];
                    let mut chunk = Chunk::new(pos);
                    self.populate_chunk(&mut chunk);
                    chunk.generate_mesh(device);
                    self.chunks.insert(pos, chunk);
                }
            }
        }
    }

    fn populate_chunk(&self, chunk: &mut Chunk) {
        let wx = chunk.position[0] * CHUNK_SIZE as i32;
        let wy = chunk.position[1] * CHUNK_SIZE as i32;
        let wz = chunk.position[2] * CHUNK_SIZE as i32;

        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                let world_x = wx + x as i32;
                let world_z = wz + z as i32;

                // Terrain Height
                let n = noise2(world_x as f32, world_z as f32);
                let height = (n * 10.0 + 20.0) as usize;

                for y in 0..CHUNK_SIZE {
                    // Convert local y to world y for height check
                    let world_y_val = (wy as usize) + y;

                    if world_y_val < height {
                        if world_y_val < height.saturating_sub(3) {
                            chunk.blocks[x][y][z] = BlockType::Stone as u8;
                        } else if world_y_val < height.saturating_sub(1) {
                            chunk.blocks[x][y][z] = BlockType::Dirt as u8;
                        } else {
                            chunk.blocks[x][y][z] = BlockType::Grass as u8;
                        }
                    }
                }
            }
        }
    }
}
