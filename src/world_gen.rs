use crate::chunk::{CHUNK_HEIGHT, CHUNK_SIZE, Chunk};
use crate::data::{BiomeDefinition, GameRegistry};
use crate::noise::NoiseMaps;

const SEA_LEVEL: i32 = 62;

pub struct WorldGenerator {
    noise: NoiseMaps,
    registry: GameRegistry,
}

impl WorldGenerator {
    pub fn new(seed: u32, registry: GameRegistry) -> Self {
        Self {
            noise: NoiseMaps::new(seed),
            registry,
        }
    }

    fn choose_biome(&self, x: f64, z: f64) -> (&str, &BiomeDefinition) {
        let temp = self.noise.get_temp(x, z);
        let humidity = self.noise.get_humidity(x, z);
        let weirdness = self.noise.get_weirdness(x, z);

        for (id, def) in &self.registry.biomes {
            let s = &def.selector;
            let check = |val: f64, range: (Option<f64>, Option<f64>)| {
                (range.0.is_none() || val >= range.0.unwrap())
                    && (range.1.is_none() || val < range.1.unwrap())
            };

            if check(weirdness, s.weirdness)
                && check(temp, s.temperature)
                && check(humidity, s.humidity)
            {
                return (id, def);
            }
        }
        ("plains", &self.registry.biomes["plains"])
    }

    pub fn generate_chunk(&self, cx: i32, cz: i32) -> Chunk {
        let mut chunk = Chunk::new(cx, cz);

        let bedrock = self.registry.get_block_id("maricraft:bedrock").unwrap_or(0);
        let water = self.registry.get_block_id("maricraft:water").unwrap_or(0);
        let stone = self.registry.get_block_id("maricraft:stone").unwrap_or(0);
        let dirt = self.registry.get_block_id("maricraft:dirt").unwrap_or(0);

        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                let wx = (cx * CHUNK_SIZE + x) as f64;
                let wz = (cz * CHUNK_SIZE + z) as f64;

                let (biome_name, biome) = self.choose_biome(wx, wz);

                let noise_val = self.noise.get_terrain(wx, wz);
                let height = (noise_val * biome.terrain.amplitude as f64).round() as i32
                    + biome.terrain.base_height;

                let top_blk = self
                    .registry
                    .get_block_id(&biome.terrain.top_block)
                    .unwrap_or(dirt);
                let mid_blk = self
                    .registry
                    .get_block_id(&biome.terrain.middle_block)
                    .unwrap_or(dirt);

                for y in 0..CHUNK_HEIGHT {
                    if y == 0 {
                        chunk.set_block(x, y, z, bedrock);
                    } else if y < height {
                        if y == height - 1 {
                            chunk.set_block(x, y, z, top_blk);
                        } else if y > height - 5 {
                            chunk.set_block(x, y, z, mid_blk);
                        } else {
                            chunk.set_block(x, y, z, stone);
                        }
                    } else if y <= SEA_LEVEL && biome_name != "desert" {
                        chunk.set_block(x, y, z, water);
                    }
                }
            }
        }
        chunk
    }
}
