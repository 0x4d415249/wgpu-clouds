mod chunk;
mod data;
mod mesher;
mod physics;
mod player;
mod texture;
mod world;

use cgmath::Vector3;
use cgmath::prelude::*;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

use crate::data::GameRegistry;
use crate::mesher::VoxelVertex;
use crate::player::Player;
use crate::texture::TextureAtlas;
use crate::world::WorldManager;

#[allow(dead_code)]
const BLOCKS_JSON: &str = include_str!("../../maricraft/assets/definitions/blocks.json");
#[allow(dead_code)]
const BIOMES_JSON: &str = include_str!("../../maricraft/assets/definitions/biomes.json");
const RAIN_PARTICLES: u32 = 20000;

fn random_f32(seed: &mut u32) -> f32 {
    *seed = (*seed).wrapping_mul(1664525).wrapping_add(1013904223);
    (*seed as f32) / (u32::MAX as f32)
}

// --- UNIFORMS ---
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
    pos: [f32; 4],
    vel: [f32; 4],
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

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,

    render_pipeline: wgpu::RenderPipeline,
    wireframe_pipeline: wgpu::RenderPipeline,
    skybox_pipeline: wgpu::RenderPipeline,
    rain_render_pipeline: wgpu::RenderPipeline,
    rain_compute_pipeline: wgpu::ComputePipeline,

    texture_bind_group: wgpu::BindGroup,
    camera_bind_group: wgpu::BindGroup,
    particle_bind_group: wgpu::BindGroup,
    depth_bind_group: wgpu::BindGroup,

    camera_buffer: wgpu::Buffer,

    depth_read_layout: wgpu::BindGroupLayout,
    camera_layout: wgpu::BindGroupLayout,

    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,

    window: Arc<Window>,

    world: WorldManager,
    player: Player,
    atmosphere: AtmosphereState,

    last_frame: Instant,
    mouse_captured: bool,
    show_wireframe: bool,
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
                .create_window(Window::default_attributes().with_title("WGPU Maricraft Async"))
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

        let mut world = WorldManager::new(&device, &shader, registry, atlas, 8);
        let player = Player::new([0.0, 150.0, 0.0]);
        let atmosphere = AtmosphereState::new();

        world.update_chunks(player.position.into(), &device, &queue);

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            label: Some("depth"),
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

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
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let depth_read_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
            label: Some("depth_read_layout"),
        });
        let depth_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &depth_read_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&depth_view),
            }],
            label: Some("depth_read_group"),
        });

        let particle_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            label: Some("particle_layout"),
        });
        let particle_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &particle_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: particle_buffer.as_entire_binding(),
            }],
            label: Some("particle_group"),
        });

        let voxel_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Voxel Pipeline Layout"),
                bind_group_layouts: &[&camera_layout, &texture_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&voxel_pipeline_layout),
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

        let wireframe_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Wireframe Pipeline"),
            layout: Some(&voxel_pipeline_layout),
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
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                polygon_mode: wgpu::PolygonMode::Line,
                cull_mode: None,
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
            bind_group_layouts: &[&camera_layout, &texture_layout, &depth_read_layout],
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
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let rain_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Rain Layout"),
            bind_group_layouts: &[&camera_layout, &texture_layout, &particle_layout],
            push_constant_ranges: &[],
        });

        let rain_compute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Rain Compute"),
                layout: Some(&rain_pipeline_layout),
                module: &shader,
                entry_point: Some("cs_rain"),
                compilation_options: Default::default(),
                cache: None,
            });

        let rain_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Rain Render"),
            layout: Some(&rain_pipeline_layout),
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

        self.state = Some(State {
            surface,
            device,
            queue,
            config,
            render_pipeline,
            wireframe_pipeline,
            skybox_pipeline,
            rain_render_pipeline,
            rain_compute_pipeline,
            texture_bind_group,
            camera_bind_group,
            particle_bind_group,
            depth_bind_group,
            camera_buffer,
            camera_layout,
            depth_read_layout,
            depth_texture,
            depth_view,
            window,
            world,
            player,
            atmosphere,
            last_frame: Instant::now(),
            mouse_captured: false,
            show_wireframe: false,
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
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::TEXTURE_BINDING,
                        label: Some("depth"),
                        view_formats: &[],
                    });
                    state.depth_view = state
                        .depth_texture
                        .create_view(&wgpu::TextureViewDescriptor::default());

                    state.depth_bind_group =
                        state.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            layout: &state.depth_read_layout,
                            entries: &[wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&state.depth_view),
                            }],
                            label: Some("depth_read_group"),
                        });
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
                state.player.process_keyboard(code, k_state);
                let amount = k_state == ElementState::Pressed;
                match code {
                    KeyCode::KeyM if amount => state.show_wireframe = !state.show_wireframe,
                    KeyCode::Escape if amount => {
                        state.mouse_captured = false;
                        let _ = state.window.set_cursor_grab(CursorGrabMode::None);
                        state.window.set_cursor_visible(true);
                    }
                    KeyCode::Digit1 if amount => {
                        state.atmosphere.target_weather = 0.0;
                        println!("Weather: Clear");
                    }
                    KeyCode::Digit2 if amount => {
                        state.atmosphere.target_weather = 0.7;
                        println!("Weather: Rain");
                    }
                    KeyCode::Digit3 if amount => {
                        state.atmosphere.target_weather = 1.0;
                        println!("Weather: Storm");
                    }
                    _ => {}
                }
            }
            WindowEvent::MouseInput {
                state: m_state,
                button,
                ..
            } => {
                if m_state == ElementState::Pressed || m_state == ElementState::Released {
                    match button {
                        MouseButton::Forward => state.player.process_mouse_button(4, m_state),
                        MouseButton::Back => state.player.process_mouse_button(3, m_state),
                        _ => {}
                    }
                }
                if m_state == ElementState::Pressed {
                    if button == MouseButton::Left && !state.mouse_captured {
                        state.mouse_captured = true;
                        let _ = state
                            .window
                            .set_cursor_grab(CursorGrabMode::Confined)
                            .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Locked));
                        state.window.set_cursor_visible(false);
                    } else if state.mouse_captured {
                        if button == MouseButton::Left {
                            let view_pos = state.player.get_view_pos();
                            let (sin_yaw, cos_yaw) = state.player.yaw.to_radians().sin_cos();
                            let (sin_pitch, cos_pitch) = state.player.pitch.to_radians().sin_cos();
                            let dir =
                                Vector3::new(cos_yaw * cos_pitch, sin_pitch, sin_yaw * cos_pitch)
                                    .normalize();

                            if let Some(hit) = physics::raycast(
                                &state.world,
                                Vector3::new(view_pos.x, view_pos.y, view_pos.z),
                                dir,
                                8.0,
                            ) {
                                state.world.set_block(
                                    hit.position[0],
                                    hit.position[1],
                                    hit.position[2],
                                    0,
                                );
                            }
                        } else if button == MouseButton::Right {
                            let view_pos = state.player.get_view_pos();
                            let (sin_yaw, cos_yaw) = state.player.yaw.to_radians().sin_cos();
                            let (sin_pitch, cos_pitch) = state.player.pitch.to_radians().sin_cos();
                            let dir =
                                Vector3::new(cos_yaw * cos_pitch, sin_pitch, sin_yaw * cos_pitch)
                                    .normalize();

                            if let Some(hit) = physics::raycast(
                                &state.world,
                                Vector3::new(view_pos.x, view_pos.y, view_pos.z),
                                dir,
                                8.0,
                            ) {
                                let px = hit.position[0] + hit.normal[0];
                                let py = hit.position[1] + hit.normal[1];
                                let pz = hit.position[2] + hit.normal[2];
                                let player_box = physics::AABB::new(
                                    state.player.position,
                                    Vector3::new(0.3, 0.9, 0.3),
                                );
                                let block_box = physics::AABB::new(
                                    Vector3::new(px as f32 + 0.5, py as f32 + 0.5, pz as f32 + 0.5),
                                    Vector3::new(0.5, 0.5, 0.5),
                                );
                                let overlap = (player_box.min.x < block_box.max.x
                                    && player_box.max.x > block_box.min.x)
                                    && (player_box.min.y < block_box.max.y
                                        && player_box.max.y > block_box.min.y)
                                    && (player_box.min.z < block_box.max.z
                                        && player_box.max.z > block_box.min.z);
                                if !overlap {
                                    state.world.set_block(px, py, pz, 1);
                                }
                            }
                        }
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - state.last_frame).as_secs_f32();
                state.last_frame = now;

                let view_pos = state.player.get_view_pos();
                let cam_pos = [view_pos.x, view_pos.y, view_pos.z];

                state
                    .world
                    .update_chunks(cam_pos, &state.device, &state.queue);

                let cx = (cam_pos[0] / 32.0).floor() as i32;
                let cy = (cam_pos[1] / 32.0).floor() as i32;
                let cz = (cam_pos[2] / 32.0).floor() as i32;

                if state.player.is_flying || state.world.chunks.contains_key(&[cx, cy, cz]) {
                    if state.mouse_captured {
                        state.player.update(dt, &state.world);
                    }
                } else {
                    state.player.velocity = Vector3::new(0.0, 0.0, 0.0);
                }

                let view_pos = state.player.get_view_pos();
                let cam_pos = [view_pos.x, view_pos.y, view_pos.z];
                state.atmosphere.update(dt, cam_pos);

                let aspect = state.config.width as f32 / state.config.height as f32;
                let proj = cgmath::perspective(cgmath::Deg(70.0), aspect, 0.1, 4000.0);
                let (sin_yaw, cos_yaw) = state.player.yaw.to_radians().sin_cos();
                let (sin_pitch, cos_pitch) = state.player.pitch.to_radians().sin_cos();
                let forward =
                    Vector3::new(cos_yaw * cos_pitch, sin_pitch, sin_yaw * cos_pitch).normalize();
                let view = cgmath::Matrix4::look_to_rh(view_pos, forward, Vector3::unit_y());
                #[rustfmt::skip]
                let correction = cgmath::Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0);
                let view_proj = correction * proj * view;
                let inv_view_proj = view_proj.invert().unwrap_or(cgmath::Matrix4::identity());

                let uniform = CameraUniform {
                    view_proj: view_proj.into(),
                    inv_view_proj: inv_view_proj.into(),
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

                if state.atmosphere.rain_intensity > 0.01 {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Rain Compute"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&state.rain_compute_pipeline);
                    cpass.set_bind_group(0, &state.camera_bind_group, &[]);
                    cpass.set_bind_group(2, &state.particle_bind_group, &[]);
                    let workgroups = RAIN_PARTICLES.div_ceil(64);
                    cpass.dispatch_workgroups(workgroups, 1, 1);
                }

                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Voxel Pass"),
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
                    let pipeline = if state.show_wireframe {
                        &state.wireframe_pipeline
                    } else {
                        &state.render_pipeline
                    };
                    rpass.set_pipeline(pipeline);
                    rpass.set_bind_group(0, &state.camera_bind_group, &[]);
                    rpass.set_bind_group(1, &state.texture_bind_group, &[]);
                    for mesh in state.world.meshes.values() {
                        rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                        rpass.set_index_buffer(
                            mesh.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        rpass.draw_indexed(0..mesh.index_count, 0, 0..1);
                    }
                }

                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Skybox Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    rpass.set_pipeline(&state.skybox_pipeline);
                    rpass.set_bind_group(0, &state.camera_bind_group, &[]);
                    rpass.set_bind_group(1, &state.texture_bind_group, &[]);
                    rpass.set_bind_group(2, &state.depth_bind_group, &[]);
                    rpass.draw(0..3, 0..1);
                }

                if state.atmosphere.rain_intensity > 0.01 {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Rain Render Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &state.depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    rpass.set_pipeline(&state.rain_render_pipeline);
                    rpass.set_bind_group(0, &state.camera_bind_group, &[]);
                    rpass.set_bind_group(1, &state.texture_bind_group, &[]);
                    rpass.set_bind_group(2, &state.particle_bind_group, &[]);
                    rpass.draw(0..4, 0..RAIN_PARTICLES);
                }

                state.queue.submit(Some(encoder.finish()));
                frame.present();

                let _ = state.device.poll(wgpu::PollType::Poll);
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
            && state.mouse_captured
            && let DeviceEvent::MouseMotion { delta } = event
        {
            state.player.process_mouse(delta.0, delta.1);
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
