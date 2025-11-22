pub const CHUNK_SIZE: usize = 32;
pub const CHUNK_VOL: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

#[derive(Clone)]
pub struct Chunk {
    pub position: [i32; 3],
    pub blocks: Vec<u16>,
}

impl Chunk {
    pub fn new(cx: i32, cy: i32, cz: i32) -> Self {
        Self {
            position: [cx, cy, cz],
            blocks: vec![0; CHUNK_VOL],
        }
    }

    #[inline(always)]
    pub fn get_block(&self, x: u32, y: u32, z: u32) -> u16 {
        // x + z*32 + y*32*32
        // x | (z << 5) | (y << 10)
        let idx = (x as usize) | ((z as usize) << 5) | ((y as usize) << 10);
        if idx < CHUNK_VOL {
            unsafe { *self.blocks.get_unchecked(idx) }
        } else {
            0
        }
    }

    #[inline(always)]
    pub fn set_block(&mut self, x: u32, y: u32, z: u32, id: u16) {
        let idx = (x as usize) | ((z as usize) << 5) | ((y as usize) << 10);
        if idx < CHUNK_VOL {
            self.blocks[idx] = id;
        }
    }
}
