mod chunk;
mod data;
mod mesher;
mod noise;
mod texture;
mod world_gen;

use cgmath::prelude::*;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

use crate::data::GameRegistry;
use crate::mesher::VoxelVertex;
use crate::texture::TextureAtlas;
use crate::world_gen::WorldGenerator;

const BLOCKS_JSON: &str = include_str!("../../maricraft/assets/definitions/blocks.json");
const BIOMES_JSON: &str = include_str!("../../maricraft/assets/definitions/biomes.json");

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    time: f32,
    day_progress: f32,
    weather_offset: f32,
    cloud_type: f32,
    lightning_intensity: f32,
    lightning_pos: [f32; 3],
    _pad1: f32,
    lightning_color: [f32; 3],
    _pad2: f32,
    wind: [f32; 2],
    rain: f32,
    _pad3: f32,
}

struct AtmosphereState {
    sim_time: f32,
    day_time: f32,
    weather_val: f32,
    target_weather: f32,
    rain_intensity: f32,
    wind_vec: [f32; 2],
    lightning_timer: f32,
    lightning_active: f32,
    lightning_pos: [f32; 3],
    rng_seed: u32,
}

impl AtmosphereState {
    fn new() -> Self {
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
            lightning_pos: [0.0, 100.0, 0.0],
            rng_seed: seed,
        }
    }

    fn update(&mut self, dt: f32, camera_pos: [f32; 3]) {
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

struct CameraController {
    speed: f32,
    sensitivity: f32,
    yaw: f32,
    pitch: f32,
    position: cgmath::Point3<f32>,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
    is_up_pressed: bool,
    is_down_pressed: bool,
    mouse_captured: bool,
    show_wireframe: bool,
}

impl CameraController {
    fn new(position: [f32; 3], speed: f32, sensitivity: f32) -> Self {
        Self {
            speed,
            sensitivity,
            yaw: -90.0,
            pitch: 0.0,
            position: position.into(),
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
            is_up_pressed: false,
            is_down_pressed: false,
            mouse_captured: false,
            show_wireframe: false,
        }
    }

    fn process_keyboard(
        &mut self,
        key: KeyCode,
        state: ElementState,
        atmos: &mut AtmosphereState,
        window: &Window,
    ) {
        let amount = state == ElementState::Pressed;
        match key {
            KeyCode::KeyW => self.is_forward_pressed = amount,
            KeyCode::KeyS => self.is_backward_pressed = amount,
            KeyCode::KeyA => self.is_left_pressed = amount,
            KeyCode::KeyD => self.is_right_pressed = amount,
            KeyCode::Space => self.is_up_pressed = amount,
            KeyCode::ShiftLeft => self.is_down_pressed = amount,
            KeyCode::KeyM if amount => self.show_wireframe = !self.show_wireframe,
            KeyCode::Escape if amount => {
                self.mouse_captured = false;
                let _ = window.set_cursor_grab(CursorGrabMode::None);
                window.set_cursor_visible(true);
            }
            KeyCode::Digit1 if amount => {
                atmos.target_weather = 0.0;
                println!("Weather: Clear");
            }
            KeyCode::Digit2 if amount => {
                atmos.target_weather = 0.7;
                println!("Weather: Rain");
            }
            KeyCode::Digit3 if amount => {
                atmos.target_weather = 1.0;
                println!("Weather: Storm");
            }
            _ => {}
        }
    }

    fn process_mouse(&mut self, dx: f64, dy: f64) {
        if self.mouse_captured {
            self.yaw += dx as f32 * self.sensitivity;
            self.pitch -= dy as f32 * self.sensitivity;
            self.pitch = self.pitch.clamp(-89.0, 89.0);
        }
    }

    fn update(&mut self, dt: f32) {
        if !self.mouse_captured {
            return;
        }
        let (sin_yaw, cos_yaw) = self.yaw.to_radians().sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.to_radians().sin_cos();
        let forward =
            cgmath::Vector3::new(cos_yaw * cos_pitch, sin_pitch, sin_yaw * cos_pitch).normalize();
        let right = forward.cross(cgmath::Vector3::unit_y()).normalize();
        let up = cgmath::Vector3::unit_y();

        let mut move_dir = cgmath::Vector3::zero();
        if self.is_forward_pressed {
            move_dir += forward;
        }
        if self.is_backward_pressed {
            move_dir -= forward;
        }
        if self.is_right_pressed {
            move_dir += right;
        }
        if self.is_left_pressed {
            move_dir -= right;
        }
        if self.is_up_pressed {
            move_dir += up;
        }
        if self.is_down_pressed {
            move_dir -= up;
        }

        if move_dir.magnitude2() > 0.0 {
            self.position += move_dir.normalize() * self.speed * dt;
        }
    }

    fn get_matrices(&self, aspect: f32) -> ([[f32; 4]; 4], [[f32; 4]; 4]) {
        let proj = cgmath::perspective(cgmath::Deg(70.0), aspect, 0.1, 2000.0);
        let (sin_yaw, cos_yaw) = self.yaw.to_radians().sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.to_radians().sin_cos();
        let forward =
            cgmath::Vector3::new(cos_yaw * cos_pitch, sin_pitch, sin_yaw * cos_pitch).normalize();
        let view = cgmath::Matrix4::look_to_rh(self.position, forward, cgmath::Vector3::unit_y());

        #[rustfmt::skip]
        let correction = cgmath::Matrix4::new(
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0,
        );
        let view_proj = correction * proj * view;
        let inv_view_proj = view_proj.invert().unwrap_or(cgmath::Matrix4::identity());
        (view_proj.into(), inv_view_proj.into())
    }
}

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    wireframe_pipeline: wgpu::RenderPipeline,
    skybox_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    texture_bind_group: wgpu::BindGroup,
    camera_bind_group: wgpu::BindGroup,
    camera_buffer: wgpu::Buffer,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    window: Arc<Window>,
    camera_controller: CameraController,
    atmosphere: AtmosphereState,
    last_frame: Instant,
}

struct App {
    state: Option<State>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("WGPU Maricraft"))
                .unwrap(),
        );
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).unwrap();

        let (adapter, device, queue) = pollster::block_on(async {
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: Some(&surface),
                    force_fallback_adapter: false,
                })
                .await
                .unwrap();

            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: None,
                    // REQUIRE POLYGON_MODE_LINE for wireframe
                    required_features: wgpu::Features::POLYGON_MODE_LINE,
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                    ..Default::default()
                })
                .await
                .unwrap();
            (adapter, device, queue)
        });

        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);
        let config = surface.get_default_config(&adapter, width, height).unwrap();
        surface.configure(&device, &config);

        let registry = GameRegistry::new_from_json(BLOCKS_JSON, BIOMES_JSON).unwrap();
        let atlas = TextureAtlas::load_from_folder("assets/textures/block").unwrap();

        let diffuse_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: atlas.image.width(),
                height: atlas.image.height(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            label: Some("diffuse_texture"),
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &diffuse_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &atlas.image,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * atlas.image.width()),
                rows_per_image: Some(atlas.image.height()),
            },
            wgpu::Extent3d {
                width: atlas.image.width(),
                height: atlas.image.height(),
                depth_or_array_layers: 1,
            },
        );
        let diffuse_view = diffuse_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let diffuse_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let generator = WorldGenerator::new(12345, registry.clone());
        let chunk = generator.generate_chunk(0, 0);
        let (vertices, indices) = mesher::generate_mesh(&chunk, &registry, &atlas);

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let num_indices = indices.len() as u32;

        let camera_controller = CameraController::new([32.0, 80.0, 32.0], 20.0, 0.1);
        let atmosphere = AtmosphereState::new();
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Buffer"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            label: Some("texture_layout"),
        });
        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_sampler),
                },
            ],
            label: Some("texture_group"),
        });

        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("camera_layout"),
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_group"),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&camera_layout, &texture_layout],
            push_constant_ranges: &[],
        });

        // SOLID PIPELINE (Standard Culling)
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<VoxelVertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset: 12,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset: 24,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            offset: 32,
                            shader_location: 3,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                        wgpu::VertexAttribute {
                            offset: 48,
                            shader_location: 4,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            // FIXED: Revert to Back Culling now that vertices are CCW
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // WIREFRAME PIPELINE (No Culling, Line Mode)
        let wireframe_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Wireframe Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<VoxelVertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset: 12,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset: 24,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            offset: 32,
                            shader_location: 3,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                        wgpu::VertexAttribute {
                            offset: 48,
                            shader_location: 4,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            // Wireframe settings
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                polygon_mode: wgpu::PolygonMode::Line,
                cull_mode: None, // See through wireframe
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let skybox_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Skybox Layout"),
            bind_group_layouts: &[&camera_layout, &texture_layout],
            push_constant_ranges: &[],
        });
        let skybox_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Skybox Pipeline"),
            layout: Some(&skybox_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_skybox"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_skybox"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            label: Some("depth"),
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.state = Some(State {
            surface,
            device,
            queue,
            config,
            render_pipeline,
            wireframe_pipeline,
            skybox_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices: indices.len() as u32,
            texture_bind_group,
            camera_bind_group,
            camera_buffer,
            depth_texture,
            depth_view,
            window,
            camera_controller,
            atmosphere,
            last_frame: Instant::now(),
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = &mut self.state else { return };
        if window_id != state.window.id() {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(physical_size) => {
                if physical_size.width > 0 && physical_size.height > 0 {
                    state.config.width = physical_size.width;
                    state.config.height = physical_size.height;
                    state.surface.configure(&state.device, &state.config);
                    state.depth_texture = state.device.create_texture(&wgpu::TextureDescriptor {
                        size: wgpu::Extent3d {
                            width: physical_size.width,
                            height: physical_size.height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Depth32Float,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                        label: Some("depth"),
                        view_formats: &[],
                    });
                    state.depth_view = state
                        .depth_texture
                        .create_view(&wgpu::TextureViewDescriptor::default());
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: k_state,
                        ..
                    },
                ..
            } => {
                state.camera_controller.process_keyboard(
                    code,
                    k_state,
                    &mut state.atmosphere,
                    &state.window,
                );
            }
            WindowEvent::MouseInput {
                state: m_state,
                button,
                ..
            } => {
                if m_state == ElementState::Pressed && button == winit::event::MouseButton::Left {
                    state.camera_controller.mouse_captured = true;
                    let _ = state
                        .window
                        .set_cursor_grab(CursorGrabMode::Confined)
                        .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Locked));
                    state.window.set_cursor_visible(false);
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - state.last_frame).as_secs_f32();
                state.last_frame = now;
                state.camera_controller.update(dt);
                let cam_pos: [f32; 3] = state.camera_controller.position.into();
                state.atmosphere.update(dt, cam_pos);

                let aspect = state.config.width as f32 / state.config.height as f32;
                let (view_proj, inv_view_proj) = state.camera_controller.get_matrices(aspect);
                let uniform = CameraUniform {
                    view_proj,
                    inv_view_proj,
                    camera_pos: cam_pos,
                    time: state.atmosphere.sim_time,
                    day_progress: state.atmosphere.day_time,
                    weather_offset: state.atmosphere.weather_val,
                    cloud_type: state.atmosphere.weather_val,
                    lightning_intensity: state.atmosphere.lightning_active,
                    lightning_pos: state.atmosphere.lightning_pos,
                    lightning_color: [0.9, 0.9, 1.0],
                    wind: state.atmosphere.wind_vec,
                    rain: state.atmosphere.rain_intensity,
                    _pad1: 0.0,
                    _pad2: 0.0,
                    _pad3: 0.0,
                };
                state
                    .queue
                    .write_buffer(&state.camera_buffer, 0, bytemuck::cast_slice(&[uniform]));

                let Ok(frame) = state.surface.get_current_texture() else {
                    return;
                };
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder = state
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &state.depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });

                    // A. Skybox
                    rpass.set_pipeline(&state.skybox_pipeline);
                    rpass.set_bind_group(0, &state.camera_bind_group, &[]);
                    rpass.set_bind_group(1, &state.texture_bind_group, &[]);
                    rpass.draw(0..3, 0..1);

                    // B. Voxels (Filled or Wireframe)
                    if state.num_indices > 0 {
                        let pipeline = if state.camera_controller.show_wireframe {
                            &state.wireframe_pipeline
                        } else {
                            &state.render_pipeline
                        };
                        rpass.set_pipeline(pipeline);
                        rpass.set_bind_group(0, &state.camera_bind_group, &[]);
                        rpass.set_bind_group(1, &state.texture_bind_group, &[]);
                        rpass.set_vertex_buffer(0, state.vertex_buffer.slice(..));
                        rpass.set_index_buffer(
                            state.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        rpass.draw_indexed(0..state.num_indices, 0, 0..1);
                    }
                }
                state.queue.submit(Some(encoder.finish()));
                frame.present();
                state.window.request_redraw();
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if let Some(state) = &mut self.state
            && let DeviceEvent::MouseMotion { delta } = event
        {
            state.camera_controller.process_mouse(delta.0, delta.1);
        }
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App { state: None };
    let _ = event_loop.run_app(&mut app);
}
