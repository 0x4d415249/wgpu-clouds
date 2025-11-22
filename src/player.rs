use crate::physics::{self};
use crate::world::WorldManager;
use cgmath::{InnerSpace, Point3, Vector3};
use winit::event::ElementState;
use winit::keyboard::KeyCode;

#[derive(PartialEq, Copy, Clone, Debug)]
pub enum GameMode {
    Survival,
    Creative,
}

pub struct Player {
    pub position: Vector3<f32>,
    pub velocity: Vector3<f32>,
    pub yaw: f32,
    pub pitch: f32,
    pub mode: GameMode,

    // State
    pub is_flying: bool,
    pub on_ground: bool,

    // Dimensions
    pub width: f32,
    pub height: f32,

    // Inputs
    move_forward: bool,
    move_backward: bool,
    move_left: bool,
    move_right: bool,
    move_up: bool,   // Space
    move_down: bool, // Shift/C
    sprint: bool,

    // Settings
    walk_speed: f32,
    fly_speed: f32,
    mouse_sensitivity: f32,

    // Double Jump Logic
    jump_timer: f32,
}

impl Player {
    pub fn new(position: [f32; 3]) -> Self {
        Self {
            position: Vector3::from(position),
            velocity: Vector3::new(0.0, 0.0, 0.0),
            yaw: -90.0,
            pitch: 0.0,
            mode: GameMode::Creative, // Start in Creative for debugging
            is_flying: true,          // Start flying
            on_ground: false,
            width: 0.6,
            height: 1.8,
            move_forward: false,
            move_backward: false,
            move_left: false,
            move_right: false,
            move_up: false,
            move_down: false,
            sprint: false,
            walk_speed: 8.0,
            fly_speed: 20.0,
            mouse_sensitivity: 0.1,
            jump_timer: 0.0,
        }
    }

    pub fn process_keyboard(&mut self, key: KeyCode, state: ElementState) {
        let pressed = state == ElementState::Pressed;
        match key {
            KeyCode::KeyW => self.move_forward = pressed,
            KeyCode::KeyS => self.move_backward = pressed,
            KeyCode::KeyA => self.move_left = pressed,
            KeyCode::KeyD => self.move_right = pressed,

            // Jump / Fly Up / Double Tap Logic
            KeyCode::Space => {
                if pressed {
                    // Check double tap
                    if self.mode == GameMode::Creative && self.jump_timer > 0.0 {
                        self.is_flying = !self.is_flying;
                        println!("Flight: {}", self.is_flying);
                        self.jump_timer = 0.0; // Reset
                    } else {
                        self.jump_timer = 0.25; // Start window
                    }
                }
                self.move_up = pressed;
            }

            // Down / Sneak
            KeyCode::ShiftLeft | KeyCode::ShiftRight | KeyCode::KeyC => self.move_down = pressed,

            // Sprint
            KeyCode::ControlLeft | KeyCode::ControlRight => self.sprint = pressed,

            // Toggle Game Mode (F5)
            KeyCode::F5 if pressed => {
                self.mode = match self.mode {
                    GameMode::Creative => GameMode::Survival,
                    GameMode::Survival => GameMode::Creative,
                };
                // Auto-disable flight if switching to survival
                if self.mode == GameMode::Survival {
                    self.is_flying = false;
                }
                println!("Game Mode: {:?}", self.mode);
            }
            _ => {}
        }
    }

    pub fn process_mouse_button(&mut self, button: u32, state: ElementState) {
        let pressed = state == ElementState::Pressed;
        // Button 4 is usually "Forward" on mice
        if button == 4 {
            self.sprint = pressed;
        }
    }

    pub fn process_mouse(&mut self, dx: f64, dy: f64) {
        self.yaw += dx as f32 * self.mouse_sensitivity;
        self.pitch -= dy as f32 * self.mouse_sensitivity;
        self.pitch = self.pitch.clamp(-89.0, 89.0);
    }

    pub fn update(&mut self, dt: f32, world: &WorldManager) {
        // Update Timers
        if self.jump_timer > 0.0 {
            self.jump_timer -= dt;
        }

        let (sin_yaw, cos_yaw) = self.yaw.to_radians().sin_cos();

        // Calculate Forward/Right vectors (XZ plane)
        let forward = Vector3::new(cos_yaw, 0.0, sin_yaw).normalize();
        let right = Vector3::new(-sin_yaw, 0.0, cos_yaw).normalize();

        let mut wish_dir = Vector3::new(0.0, 0.0, 0.0);
        if self.move_forward {
            wish_dir += forward;
        }
        if self.move_backward {
            wish_dir -= forward;
        }
        if self.move_right {
            wish_dir += right;
        }
        if self.move_left {
            wish_dir -= right;
        }

        if wish_dir.magnitude2() > 0.0 {
            wish_dir = wish_dir.normalize();
        }

        // Calculate Speed
        let mut speed = if self.is_flying {
            self.fly_speed
        } else {
            self.walk_speed
        };
        if self.sprint {
            speed *= if self.is_flying { 50.0 } else { 1.5 }; // Super fast flight for debugging
        }

        if self.is_flying {
            // --- FLYING PHYSICS ---
            self.velocity = wish_dir * speed;

            // Vertical Flight
            if self.move_up {
                self.velocity.y = speed;
            } else if self.move_down {
                self.velocity.y = -speed;
            } else {
                self.velocity.y = 0.0;
            }

            // Smooth acceleration/friction for flight could be added,
            // but direct control is better for debug/creative.
            self.position += self.velocity * dt;
            self.on_ground = false;
        } else {
            // --- WALKING/SURVIVAL PHYSICS ---

            // Horizontal movement
            let target_vel_x = wish_dir.x * speed;
            let target_vel_z = wish_dir.z * speed;

            // Accelerate/Friction
            let accel = if self.on_ground { 15.0 } else { 2.0 }; // Air control is lower
            self.velocity.x += (target_vel_x - self.velocity.x) * accel * dt;
            self.velocity.z += (target_vel_z - self.velocity.z) * accel * dt;

            // Gravity
            self.velocity.y -= 30.0 * dt;

            // Jump
            if self.move_up && self.on_ground {
                self.velocity.y = 9.0;
                self.on_ground = false;
            }

            // Collision
            let (new_pos, new_vel, on_ground) = physics::resolve_collision(
                world,
                self.position,
                self.velocity * dt,
                Vector3::new(self.width, self.height, self.width),
            );

            self.position = new_pos;
            self.velocity = new_vel / dt; // Convert displacement back to velocity
            self.on_ground = on_ground;
        }
    }

    pub fn get_view_pos(&self) -> Point3<f32> {
        let eye_height = if self.is_flying && self.move_down {
            1.0
        } else {
            1.62
        };
        Point3::new(
            self.position.x,
            self.position.y + eye_height,
            self.position.z,
        )
    }
}
