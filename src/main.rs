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

// --- CONSTANTS ---
const BLOCKS_JSON: &str = include_str!("../../maricraft/assets/definitions/blocks.json");
const BIOMES_JSON: &str = include_str!("../../maricraft/assets/definitions/biomes.json");
const RAIN_PARTICLES: u32 = 20000;

// --- HELPERS ---
fn random_f32(seed: &mut u32) -> f32 {
    *seed = (*seed).wrapping_mul(1664525).wrapping_add(1013904223);
    (*seed as f32) / (u32::MAX as f32)
}

fn lerp(start: f32, end: f32, speed: f32, dt: f32) -> f32 {
    start + (end - start) * (speed * dt).clamp(0.0, 1.0)
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// --- UNIFORMS & STRUCTS ---

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

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ParticleInit {
    pos: [f32; 4], // xyz, scale
    vel: [f32; 4], // xyz, padding
}

// --- ATMOSPHERE STATE ---

struct AtmosphereState {
    // Uniform Data
    day_time: f32,
    weather: f32,
    cloud_type: f32,
    lightning_intensity: f32,
    lightning_pos: [f32; 3],
    lightning_color: [f32; 3],
    rain: f32,
    wind: [f32; 2],

    // Sim Internals
    sim_time: f32,
    weather_offset: f32,
    target_weather_offset: f32,
    lightning_timer: f32,
    rng_seed: u32,
}

impl AtmosphereState {
    fn new() -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        Self {
            day_time: 0.3,
            weather: 0.0,
            cloud_type: 0.0,
            lightning_intensity: 0.0,
            lightning_pos: [0.0, 150.0, 0.0],
            lightning_color: [1.0, 1.0, 1.0],
            rain: 0.0,
            wind: [0.1, 0.0],
            sim_time: 0.0,
            weather_offset: 0.0,
            target_weather_offset: 0.0,
            lightning_timer: 5.0,
            rng_seed: seed,
        }
    }

    fn update(&mut self, dt: f32, camera_pos: [f32; 3]) {
        // Day Cycle
        self.day_time += dt / 60.0; // 60s day
        if self.day_time > 1.0 {
            self.day_time -= 1.0;
        }

        // Weather Sim
        self.sim_time += dt * 0.1;
        let w_noise = self.sim_time.sin()
            + (self.sim_time * 2.3).cos() * 0.5
            + (self.sim_time * 0.7).sin() * 0.2;
        let auto_weather = (w_noise * 0.5 + 0.5).clamp(0.0, 1.0);

        self.weather_offset = lerp(self.weather_offset, self.target_weather_offset, 1.0, dt);
        let target_weather = (auto_weather + self.weather_offset).clamp(0.0, 1.0);
        self.weather = lerp(self.weather, target_weather, 0.5, dt);

        // Clouds & Rain
        let t_noise = ((self.sim_time * 1.5).cos() * 0.5 + 0.5).clamp(0.0, 1.0);
        self.cloud_type = lerp(self.cloud_type, t_noise, 0.5, dt);

        let target_rain = smoothstep(0.6, 0.8, self.weather);
        self.rain = lerp(self.rain, target_rain, 0.5, dt);

        // Wind
        let storm_boost = 1.0 + self.weather * 3.0;
        self.wind[0] = lerp(
            self.wind[0],
            (self.sim_time * 3.0).sin() * storm_boost,
            0.1,
            dt,
        );
        self.wind[1] = lerp(
            self.wind[1],
            (self.sim_time * 2.5).cos() * storm_boost,
            0.1,
            dt,
        );

        // Lightning
        if self.weather > 0.85 {
            self.lightning_timer -= dt;
            if self.lightning_timer <= 0.0 {
                self.lightning_intensity = 1.0;
                let rx = (random_f32(&mut self.rng_seed) - 0.5) * 400.0;
                let rz = (random_f32(&mut self.rng_seed) - 0.5) * 400.0;
                self.lightning_pos = [
                    camera_pos[0] + rx,
                    140.0 + random_f32(&mut self.rng_seed) * 30.0,
                    camera_pos[2] + rz,
                ];
                self.lightning_color = [
                    0.6 + random_f32(&mut self.rng_seed) * 0.4,
                    0.6 + random_f32(&mut self.rng_seed) * 0.4,
                    1.0,
                ];
                self.lightning_timer = 0.2 + random_f32(&mut self.rng_seed) * 3.0;
            }
        } else {
            self.lightning_intensity = 0.0;
        }
        self.lightning_intensity = lerp(self.lightning_intensity, 0.0, 8.0, dt);
    }
}

// --- CAMERA CONTROLLER ---

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
            KeyCode::ArrowUp if amount => {
                atmos.target_weather_offset += 0.1;
                println!("Weather: {:.1}", atmos.target_weather_offset);
            }
            KeyCode::ArrowDown if amount => {
                atmos.target_weather_offset -= 0.1;
                println!("Weather: {:.1}", atmos.target_weather_offset);
            }
            KeyCode::KeyE if amount => atmos.target_weather_offset = -1.0,
            KeyCode::KeyR if amount => atmos.target_weather_offset = 0.0,
            KeyCode::KeyT if amount => atmos.target_weather_offset = 1.0,
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

        let mut dir = cgmath::Vector3::zero();
        if self.is_forward_pressed {
            dir += forward;
        }
        if self.is_backward_pressed {
            dir -= forward;
        }
        if self.is_right_pressed {
            dir += right;
        }
        if self.is_left_pressed {
            dir -= right;
        }
        if self.is_up_pressed {
            dir += cgmath::Vector3::unit_y();
        }
        if self.is_down_pressed {
            dir -= cgmath::Vector3::unit_y();
        }

        if dir.magnitude2() > 0.0 {
            self.position += dir.normalize() * self.speed * dt;
        }
    }

    fn get_matrices(&self, aspect: f32) -> ([[f32; 4]; 4], [[f32; 4]; 4]) {
        let proj = cgmath::perspective(cgmath::Deg(70.0), aspect, 0.1, 2000.0);
        let (sin_yaw, cos_yaw) = self.yaw.to_radians().sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.to_radians().sin_cos();
        let forward =
            cgmath::Vector3::new(cos_yaw * cos_pitch, sin_pitch, sin_yaw * cos_pitch).normalize();
        let view = cgmath::Matrix4::look_to_rh(self.position, forward, cgmath::Vector3::unit_y());
        let correction = cgmath::Matrix4::new(
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0,
        );
        let vp = correction * proj * view;
        (
            vp.into(),
            vp.invert().unwrap_or(cgmath::Matrix4::identity()).into(),
        )
    }
}

// --- STATE ---

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,

    // Pipelines
    render_pipeline: wgpu::RenderPipeline,
    wireframe_pipeline: wgpu::RenderPipeline,
    skybox_pipeline: wgpu::RenderPipeline,
    compute_pipeline: wgpu::ComputePipeline,
    rain_pipeline: wgpu::RenderPipeline,

    // Resources
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,

    camera_buffer: wgpu::Buffer,
    particle_buffer: wgpu::Buffer,

    texture_bind_group: wgpu::BindGroup,
    camera_bind_group: wgpu::BindGroup,
    particle_bind_group: wgpu::BindGroup,

    depth_view: wgpu::TextureView,
    depth_texture: wgpu::Texture,

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
                    // Note: VERTEX_WRITABLE_STORAGE required for compute particles
                    required_features: wgpu::Features::POLYGON_MODE_LINE
                        | wgpu::Features::VERTEX_WRITABLE_STORAGE,
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                    ..Default::default()
                })
                .await
                .unwrap();
            (adapter, device, queue)
        });

        let size = window.inner_size();
        let config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .unwrap();
        surface.configure(&device, &config);

        // Load Game Data
        let registry = GameRegistry::new_from_json(BLOCKS_JSON, BIOMES_JSON).unwrap();
        let atlas = TextureAtlas::load_from_folder("assets/textures/block").unwrap();
        let generator = WorldGenerator::new(12345, registry.clone());
        let chunk = generator.generate_chunk(0, 0);
        let (vertices, indices) = mesher::generate_mesh(&chunk, &registry, &atlas);

        // --- TEXTURES ---
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

        // --- BUFFERS ---
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

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Buffer"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut rng_seed = 12345;
        let particles: Vec<ParticleInit> = (0..RAIN_PARTICLES)
            .map(|_| {
                let x = (random_f32(&mut rng_seed) - 0.5) * 80.0;
                let y = (random_f32(&mut rng_seed) - 0.5) * 60.0;
                let z = (random_f32(&mut rng_seed) - 0.5) * 80.0;
                let scale = 0.5 + random_f32(&mut rng_seed) * 0.5;
                let speed = -40.0 - random_f32(&mut rng_seed) * 20.0;
                ParticleInit {
                    pos: [x, y, z, scale],
                    vel: [0.0, speed, 0.0, 0.0],
                }
            })
            .collect();

        let particle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle Buffer"),
            contents: bytemuck::cast_slice(&particles),
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
        });

        // --- BIND GROUPS ---
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Camera Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX
                    | wgpu::ShaderStages::FRAGMENT
                    | wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: None,
        });

        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Texture Layout"),
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
            label: None,
        });

        let particle_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Particle Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let particle_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &particle_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: particle_buffer.as_entire_binding(),
            }],
            label: None,
        });

        // --- PIPELINES ---
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        // --- PIPELINES ---
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Main Pipeline Layout"),
            bind_group_layouts: &[&camera_layout, &texture_layout],
            push_constant_ranges: &[],
        });

        // Define common Vertex Layout to reuse
        let vertex_layout = wgpu::VertexBufferLayout {
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
        };

        // Define common Color Target to reuse
        let color_targets = [Some(wgpu::ColorTargetState {
            format: config.format,
            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        })];

        // Define common Depth State to reuse
        let depth_state = wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };

        // 1. Voxel Render Pipeline
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Voxel Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout.clone()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &color_targets,
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(depth_state.clone()),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // 2. Wireframe Pipeline (Reuses layout/targets, changes Primitive state)
        let wireframe_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Wireframe Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout.clone()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &color_targets,
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                polygon_mode: wgpu::PolygonMode::Line,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(depth_state.clone()),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // 3. Skybox Pipeline (No vertex buffers, different VS/FS)
        let skybox_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Skybox Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_skybox"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_skybox"),
                targets: &color_targets,
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

        // Rain Pipelines
        let rain_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Rain Layout"),
            bind_group_layouts: &[&camera_layout, &texture_layout, &particle_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Rain Compute"),
            layout: Some(&rain_layout),
            module: &shader,
            entry_point: Some("cs_rain"),
            compilation_options: Default::default(),
            cache: None,
        });

        let rain_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Rain Render"),
            layout: Some(&rain_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_rain"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_rain"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Less,
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
            compute_pipeline,
            rain_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            camera_buffer,
            particle_buffer,
            texture_bind_group,
            camera_bind_group,
            particle_bind_group,
            depth_view,
            depth_texture,
            window,
            camera_controller: CameraController::new([32.0, 80.0, 32.0], 20.0, 0.1),
            atmosphere: AtmosphereState::new(),
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
            WindowEvent::Resized(ps) => {
                if ps.width > 0 && ps.height > 0 {
                    state.config.width = ps.width;
                    state.config.height = ps.height;
                    state.surface.configure(&state.device, &state.config);
                    state.depth_texture = state.device.create_texture(&wgpu::TextureDescriptor {
                        size: wgpu::Extent3d {
                            width: ps.width,
                            height: ps.height,
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
            } => state.camera_controller.process_keyboard(
                code,
                k_state,
                &mut state.atmosphere,
                &state.window,
            ),
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
                state
                    .atmosphere
                    .update(dt, state.camera_controller.position.into());

                let (vp, inv_vp) = state
                    .camera_controller
                    .get_matrices(state.config.width as f32 / state.config.height as f32);
                let uniform = CameraUniform {
                    view_proj: vp,
                    inv_view_proj: inv_vp,
                    camera_pos: state.camera_controller.position.into(),
                    time: state.atmosphere.sim_time,
                    day_progress: state.atmosphere.day_time,
                    weather_offset: state.atmosphere.weather,
                    cloud_type: state.atmosphere.cloud_type,
                    lightning_intensity: state.atmosphere.lightning_intensity,
                    lightning_pos: state.atmosphere.lightning_pos,
                    lightning_color: state.atmosphere.lightning_color,
                    wind: state.atmosphere.wind,
                    rain: state.atmosphere.rain,
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

                // 1. Rain Physics
                if state.atmosphere.rain > 0.01 {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Rain Compute"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&state.compute_pipeline);
                    cpass.set_bind_group(0, &state.camera_bind_group, &[]);
                    cpass.set_bind_group(1, &state.texture_bind_group, &[]); // Satisfy layout
                    cpass.set_bind_group(2, &state.particle_bind_group, &[]);
                    let workgroups = RAIN_PARTICLES.div_ceil(64);
                    cpass.dispatch_workgroups(workgroups, 1, 1);
                }

                // 2. Main Render
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

                    // Skybox
                    rpass.set_pipeline(&state.skybox_pipeline);
                    rpass.set_bind_group(0, &state.camera_bind_group, &[]);
                    rpass.set_bind_group(1, &state.texture_bind_group, &[]);
                    rpass.draw(0..3, 0..1);

                    // Voxels
                    if state.num_indices > 0 {
                        rpass.set_pipeline(if state.camera_controller.show_wireframe {
                            &state.wireframe_pipeline
                        } else {
                            &state.render_pipeline
                        });
                        rpass.set_bind_group(0, &state.camera_bind_group, &[]);
                        rpass.set_bind_group(1, &state.texture_bind_group, &[]);
                        rpass.set_vertex_buffer(0, state.vertex_buffer.slice(..));
                        rpass.set_index_buffer(
                            state.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        rpass.draw_indexed(0..state.num_indices, 0, 0..1);
                    }

                    // Rain
                    if state.atmosphere.rain > 0.01 {
                        rpass.set_pipeline(&state.rain_pipeline);
                        rpass.set_bind_group(0, &state.camera_bind_group, &[]);
                        rpass.set_bind_group(1, &state.texture_bind_group, &[]);
                        rpass.set_bind_group(2, &state.particle_bind_group, &[]);
                        rpass.draw(0..4, 0..RAIN_PARTICLES);
                    }
                }
                state.queue.submit(Some(encoder.finish()));
                frame.present();
                state.window.request_redraw();
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: winit::event::DeviceId, event: DeviceEvent) {
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
