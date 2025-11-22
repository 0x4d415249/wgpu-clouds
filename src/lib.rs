pub mod atmosphere;
pub mod chunk;
pub mod data;
pub mod mesher;
pub mod physics;
pub mod player;
pub mod renderer;
pub mod shader;
pub mod shader_gen;
pub mod texture;
pub mod world;

use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::atmosphere::AtmosphereState;
use crate::data::GameRegistry;
use crate::player::Player;
use crate::renderer::Renderer;
use crate::texture::TextureAtlas;
use crate::world::WorldManager;

const BLOCKS_JSON: &str = include_str!("../../maricraft/assets/definitions/blocks.json");
const BIOMES_JSON: &str = include_str!("../../maricraft/assets/definitions/biomes.json");

struct Game {
    renderer: Renderer,
    world: WorldManager,
    player: Player,
    atmosphere: AtmosphereState,
    mouse_captured: bool,
    show_wireframe: bool,
    last_frame: Instant,
    window: Arc<Window>,
    frame_count: u64,
    last_log: Instant,
}

struct App {
    game: Option<Game>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        println!("[DEBUG] Resumed - Initializing...");
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("WGPU Voxel Engine"))
                .unwrap(),
        );

        let registry = GameRegistry::new_from_json(BLOCKS_JSON, BIOMES_JSON).unwrap();
        let atlas = TextureAtlas::load_from_folder("assets/textures/block").unwrap();

        println!("[DEBUG] Creating Renderer...");
        let renderer = pollster::block_on(Renderer::new(window.clone(), &registry, &atlas));

        println!("[DEBUG] Creating WorldManager...");
        let mut world = WorldManager::new(
            &renderer.device,
            &renderer.queue,
            &renderer.shader_module,
            renderer.bind_layouts.gen_layout.clone(),
            registry,
            atlas,
            6,
        );

        let player = Player::new([0.0, 150.0, 0.0]);
        let atmosphere = AtmosphereState::new();

        println!("[DEBUG] Initial Chunk Update...");
        world.update_chunks(player.position.into(), &renderer.device, &renderer.queue);

        self.game = Some(Game {
            renderer,
            world,
            player,
            atmosphere,
            mouse_captured: false,
            show_wireframe: false,
            last_frame: Instant::now(),
            window,
            frame_count: 0,
            last_log: Instant::now(),
        });
        println!("[DEBUG] Initialization Complete.");
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(game) = &mut self.game else { return };
        if window_id != game.window.id() {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                game.renderer.resize(size);
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state,
                        ..
                    },
                ..
            } => {
                game.player.process_keyboard(key, state);
                if state == ElementState::Pressed {
                    match key {
                        KeyCode::Escape => {
                            game.mouse_captured = false;
                            let _ = game
                                .window
                                .set_cursor_grab(winit::window::CursorGrabMode::None);
                            game.window.set_cursor_visible(true);
                        }
                        KeyCode::KeyM => {
                            game.show_wireframe = !game.show_wireframe;
                            game.renderer.set_wireframe(game.show_wireframe);
                        }
                        KeyCode::F1 => game.renderer.cycle_render_scale(),
                        // Weather Controls
                        KeyCode::Digit1 => {
                            game.atmosphere.target_weather = 0.0;
                            println!("Weather set to CLEAR");
                        }
                        KeyCode::Digit2 => {
                            game.atmosphere.target_weather = 0.7;
                            println!("Weather set to RAIN");
                        }
                        KeyCode::Digit3 => {
                            game.atmosphere.target_weather = 1.0;
                            println!("Weather set to STORM");
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if state == ElementState::Pressed || state == ElementState::Released {
                    match button {
                        winit::event::MouseButton::Forward => {
                            game.player.process_mouse_button(4, state)
                        }
                        winit::event::MouseButton::Back => {
                            game.player.process_mouse_button(3, state)
                        }
                        _ => {}
                    }
                }
                if state == ElementState::Pressed
                    && button == winit::event::MouseButton::Left
                    && !game.mouse_captured
                {
                    game.mouse_captured = true;
                    let _ = game
                        .window
                        .set_cursor_grab(winit::window::CursorGrabMode::Confined)
                        .or_else(|_| {
                            game.window
                                .set_cursor_grab(winit::window::CursorGrabMode::Locked)
                        });
                    game.window.set_cursor_visible(false);
                }
            }
            WindowEvent::RedrawRequested => {
                let size = game.window.inner_size();
                if size.width == 0 || size.height == 0 {
                    return;
                }

                game.frame_count += 1;
                let now = Instant::now();
                let dt = (now - game.last_frame).as_secs_f32();
                game.last_frame = now;

                if now.duration_since(game.last_log).as_secs_f32() >= 1.0 {
                    println!(
                        "[DEBUG] FPS: {} | Chunks: {}",
                        game.frame_count,
                        game.world.chunks.len()
                    );
                    game.frame_count = 0;
                    game.last_log = now;
                }

                // Update Logic
                if game.mouse_captured {
                    game.player.update(dt, &game.world);
                }

                // Update Atmosphere
                let view_pos = game.player.get_view_pos();
                let cam_pos = [view_pos.x, view_pos.y, view_pos.z];
                game.atmosphere.update(dt, cam_pos);

                game.world.update_chunks(
                    game.player.position.into(),
                    &game.renderer.device,
                    &game.renderer.queue,
                );

                // Render
                match game
                    .renderer
                    .render(&game.world, &game.player, &game.atmosphere)
                {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost) | Err(wgpu::SurfaceError::Outdated) => {
                        game.renderer.resize(game.window.inner_size());
                    }
                    Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                    Err(wgpu::SurfaceError::Timeout) => {}
                    Err(e) => eprintln!("[ERROR] Render failed: {:?}", e),
                }

                game.window.request_redraw();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(game) = &self.game {
            game.window.request_redraw();
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if let Some(game) = &mut self.game
            && let DeviceEvent::MouseMotion { delta } = event
            && game.mouse_captured
        {
            game.player.process_mouse(delta.0, delta.1);
        }
    }
}

pub async fn run() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App { game: None };
    let _ = event_loop.run_app(&mut app);
}
