use crate::physics::{self};
use crate::world::WorldManager;
use cgmath::{InnerSpace, Point3, Vector3};
use winit::event::ElementState;
use winit::keyboard::KeyCode;

#[derive(PartialEq, Copy, Clone, Debug)]
pub enum GameMode {
    Survival,
    Creative,
    Spectator,
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
                    if !self.move_up { // Only trigger on fresh press
                        // Check double tap
                        if (self.mode == GameMode::Creative || self.mode == GameMode::Spectator) && self.jump_timer > 0.0 {
                            self.is_flying = !self.is_flying;
                            println!("Flight: {}", self.is_flying);
                            self.jump_timer = 0.0; // Reset
                        } else {
                            self.jump_timer = 0.25; // Start window
                        }
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
                    GameMode::Survival => GameMode::Spectator,
                    GameMode::Spectator => GameMode::Creative,
                };
                // Auto-disable flight if switching to survival
                if self.mode == GameMode::Survival {
                    self.is_flying = false;
                } else {
                    self.is_flying = true;
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
            speed *= if self.is_flying { 2.0 } else { 1.5 }; // Reduced from 50.0 to 2.0 for sanity
        }

        if self.is_flying || self.mode == GameMode::Spectator {
            // --- FLYING PHYSICS ---
            // Apply inertia/drag
            let friction = if self.mode == GameMode::Spectator { 5.0 } else { 5.0 };
            let drag_factor = (1.0 - friction * dt).max(0.0);
            self.velocity *= drag_factor;
            
            // Acceleration
            let accel = 50.0;
            self.velocity += wish_dir * accel * dt;
            
            // Vertical Flight
            if self.move_up {
                self.velocity.y += accel * dt;
            } else if self.move_down {
                self.velocity.y -= accel * dt;
            }
            
            // Cap speed
            if self.velocity.magnitude() > speed {
                self.velocity = self.velocity.normalize() * speed;
            }

            self.position += self.velocity * dt;
            self.on_ground = false;
        } else {
            // --- WALKING/SURVIVAL PHYSICS ---

            // Physics Constants (Tuned for Minecraft-like feel)
            let gravity = 32.0;
            let jump_force = 9.0; // ~1.25m jump height
            
            let (accel, drag) = if self.on_ground {
                if self.sprint { (60.0, 10.0) } else { (45.0, 10.0) }
            } else {
                (8.0, 0.5) // Air control
            };

            // Apply Drag (Friction)
            // Damping factor: v_new = v_old * (1 - drag * dt)
            // We use a slightly more stable integration for drag
            let drag_factor = (1.0 - drag * dt).max(0.0);
            self.velocity.x *= drag_factor;
            self.velocity.z *= drag_factor;

            // Apply Input Acceleration
            if wish_dir.magnitude2() > 0.0 {
                self.velocity.x += wish_dir.x * accel * dt;
                self.velocity.z += wish_dir.z * accel * dt;
            }

            // Gravity
            self.velocity.y -= gravity * dt;

            // Terminal velocity check (optional, but good for stability)
            self.velocity.y = self.velocity.y.max(-60.0);

            // Jump
            if self.move_up && self.on_ground {
                self.velocity.y = jump_force;
                self.on_ground = false;
            }

            // Collision
            // Spectator mode disables collision
            if self.mode != GameMode::Spectator {
                let (new_pos, new_vel, on_ground) = physics::resolve_collision(
                    world,
                    self.position,
                    self.velocity * dt,
                    Vector3::new(self.width, self.height, self.width),
                );

                self.position = new_pos;
                self.velocity = new_vel / dt; 
                self.on_ground = on_ground;
                
                // Fix for skyrocketing: If we are on ground, ensure Y velocity is not positive (unless jumping next frame)
                // Actually, resolve_collision should zero out Y velocity if we hit ground.
                // But if we are bunny hopping, we might be adding jump force while still "on ground" from previous frame?
                // No, jump logic sets on_ground = false.
                
                // The issue might be `new_vel / dt`. If `dt` is very small, `new_vel` (displacement) / `dt` can be huge.
                // `resolve_collision` returns modified displacement.
                // If we hit the ground, `new_vel.y` becomes 0.0.
                // So `self.velocity.y` becomes 0.0.
                
                // However, if we hit a wall or step up, `new_vel` might have small adjustments that get amplified by `1/dt`.
                // It is safer to reconstruct velocity from the *actual* movement if we want, OR just trust `resolve_collision` zeroing.
                
                // Let's cap the velocity to avoid explosions.
                self.velocity.x = self.velocity.x.clamp(-100.0, 100.0);
                self.velocity.y = self.velocity.y.clamp(-100.0, 100.0);
                self.velocity.z = self.velocity.z.clamp(-100.0, 100.0);
                
            } else {
                self.position += self.velocity * dt;
            }
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
