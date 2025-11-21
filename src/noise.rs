//! Procedural noise generation utilities.

use noise::{NoiseFn, Simplex};

pub struct Mulberry32 {
    state: u32,
}

impl Mulberry32 {
    pub fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    pub fn next(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x6d2b79f5);
        let mut t = self.state;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        t = t ^ (t >> 14);
        (t as f64) / 4294967296.0
    }
}

pub struct NoiseMaps {
    pub temp_noise: Simplex,
    pub humidity_noise: Simplex,
    pub weirdness_noise: Simplex,
    pub terrain_noise: Simplex,
    pub cave_noise: Simplex,
    pub ore_noise: Simplex,
    pub river_noise: Simplex,
    pub feature_seed: u32,
}

impl NoiseMaps {
    pub fn new(seed: u32) -> Self {
        Self {
            temp_noise: Simplex::new(seed),
            humidity_noise: Simplex::new(seed.wrapping_add(1)),
            weirdness_noise: Simplex::new(seed.wrapping_add(2)),
            terrain_noise: Simplex::new(seed.wrapping_add(3)),
            cave_noise: Simplex::new(seed.wrapping_add(4)),
            ore_noise: Simplex::new(seed.wrapping_add(5)),
            river_noise: Simplex::new(seed.wrapping_add(6)),
            feature_seed: seed.wrapping_add(7),
        }
    }

    pub fn get_temp(&self, x: f64, z: f64) -> f64 {
        self.temp_noise.get([x / 1024.0, z / 1024.0])
    }

    pub fn get_humidity(&self, x: f64, z: f64) -> f64 {
        self.humidity_noise.get([x / 1024.0, z / 1024.0])
    }

    pub fn get_weirdness(&self, x: f64, z: f64) -> f64 {
        self.weirdness_noise.get([x / 512.0, z / 512.0])
    }

    pub fn get_terrain(&self, x: f64, z: f64) -> f64 {
        self.terrain_noise.get([x / 128.0, z / 128.0])
    }

    pub fn get_cave(&self, x: f64, y: f64, z: f64) -> f64 {
        self.cave_noise.get([x / 50.0, y / 50.0, z / 50.0])
    }

    pub fn get_ore(&self, x: f64, y: f64, z: f64) -> f64 {
        self.ore_noise.get([x / 20.0, y / 20.0, z / 20.0])
    }

    pub fn get_river(&self, x: f64, z: f64) -> f64 {
        self.river_noise.get([x / 400.0, z / 400.0])
    }
}
