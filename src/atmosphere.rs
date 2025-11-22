use std::time::{SystemTime, UNIX_EPOCH};

pub struct AtmosphereState {
    pub sim_time: f32,
    pub day_time: f32,
    pub weather_val: f32,
    pub target_weather: f32,
    pub rain_intensity: f32,
    pub wind_vec: [f32; 2],
    pub lightning_timer: f32,
    pub lightning_active: f32,
    pub lightning_pos: [f32; 3],
    pub rng_seed: u32,
}

impl Default for AtmosphereState {
    fn default() -> Self {
        Self::new()
    }
}

impl AtmosphereState {
    pub fn new() -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        Self {
            sim_time: 0.0,
            day_time: 0.3,
            weather_val: 0.0,
            target_weather: 0.0,
            rain_intensity: 0.0,
            wind_vec: [0.1, 0.0],
            lightning_timer: 5.0,
            lightning_active: 0.0,
            lightning_pos: [0.0, 150.0, 0.0],
            rng_seed: seed,
        }
    }

    pub fn update(&mut self, dt: f32, camera_pos: [f32; 3]) {
        self.sim_time += dt;
        self.day_time = (self.day_time + dt * 0.005) % 1.0;

        let drift_speed = 0.1;
        if self.weather_val < self.target_weather {
            self.weather_val += drift_speed * dt;
        } else if self.weather_val > self.target_weather {
            self.weather_val -= drift_speed * dt;
        }
        self.weather_val = self.weather_val.clamp(0.0, 1.0);

        let rain_threshold = 0.6;
        let target_rain = if self.weather_val > rain_threshold {
            (self.weather_val - rain_threshold) / (1.0 - rain_threshold)
        } else {
            0.0
        };
        self.rain_intensity = self.rain_intensity + (target_rain - self.rain_intensity) * dt;

        self.wind_vec[0] = (self.sim_time * 0.1).sin() * 0.5 + (self.weather_val * 2.0);
        self.wind_vec[1] = (self.sim_time * 0.07).cos() * 0.5;

        if self.weather_val > 0.85 {
            self.lightning_timer -= dt;
            if self.lightning_timer <= 0.0 {
                self.lightning_active = 1.0;
                self.lightning_timer = self.rand_float(0.5, 4.0);
                let lx = camera_pos[0] + self.rand_float(-100.0, 100.0);
                let lz = camera_pos[2] + self.rand_float(-100.0, 100.0);
                self.lightning_pos = [lx, 150.0, lz];
            }
        }
        self.lightning_active = (self.lightning_active - dt * 4.0).max(0.0);
    }

    fn rand_float(&mut self, min: f32, max: f32) -> f32 {
        self.rng_seed = self.rng_seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let val = (self.rng_seed as f32) / (u32::MAX as f32);
        min + val * (max - min)
    }
}
