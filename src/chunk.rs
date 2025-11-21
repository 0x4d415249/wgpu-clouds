use crate::data::BlockId;

pub const CHUNK_SIZE: i32 = 64;
pub const CHUNK_HEIGHT: i32 = 255;
pub const CHUNK_VOLUME: usize = (CHUNK_SIZE * CHUNK_HEIGHT * CHUNK_SIZE) as usize;

pub struct Chunk {
    pub position: [i32; 2], // Chunk coordinates (X, Z)
    pub blocks: Vec<BlockId>,
}

impl Chunk {
    pub fn new(cx: i32, cz: i32) -> Self {
        Self {
            position: [cx, cz],
            blocks: vec![0; CHUNK_VOLUME],
        }
    }

    #[inline]
    pub fn get_index(x: i32, y: i32, z: i32) -> Option<usize> {
        if (0..CHUNK_SIZE).contains(&x)
            && (0..CHUNK_HEIGHT).contains(&y)
            && (0..CHUNK_SIZE).contains(&z)
        {
            Some(
                (y as usize * (CHUNK_SIZE * CHUNK_SIZE) as usize)
                    + (z as usize * CHUNK_SIZE as usize)
                    + x as usize,
            )
        } else {
            None
        }
    }

    #[inline]
    pub fn get_block(&self, x: i32, y: i32, z: i32) -> BlockId {
        Self::get_index(x, y, z).map_or(0, |i| self.blocks[i])
    }

    #[inline]
    pub fn set_block(&mut self, x: i32, y: i32, z: i32, id: BlockId) {
        if let Some(i) = Self::get_index(x, y, z) {
            self.blocks[i] = id;
        }
    }
}
