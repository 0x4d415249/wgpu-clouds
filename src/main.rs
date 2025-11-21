use cgmath::prelude::*;
use std::borrow::Cow;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::event::*;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

const RENDER_SCALE: f32 = 0.5;
const RAIN_PARTICLES: u32 = 20000;

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

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct RainInstance {
    offset: [f32; 3],
}

struct Camera {
    eye: cgmath::Point3<f32>,
    yaw: f32,
    pitch: f32,
}

impl Camera {
    fn new(pos: [f32; 3]) -> Self {
        Self {
            eye: cgmath::Point3::new(pos[0], pos[1], pos[2]),
            yaw: -90.0,
            pitch: 0.0,
        }
    }

    fn build_view_projection_matrix(
        &self,
        aspect: f32,
    ) -> (cgmath::Matrix4<f32>, cgmath::Matrix4<f32>) {
        let proj = cgmath::perspective(cgmath::Deg(75.0), aspect, 0.1, 4000.0);
        let (sin_pitch, cos_pitch) = cgmath::Rad::from(cgmath::Deg(self.pitch)).0.sin_cos();
        let (sin_yaw, cos_yaw) = cgmath::Rad::from(cgmath::Deg(self.yaw)).0.sin_cos();
        let forward =
            cgmath::Vector3::new(cos_pitch * cos_yaw, sin_pitch, cos_pitch * sin_yaw).normalize();
        let view = cgmath::Matrix4::look_to_rh(self.eye, forward, cgmath::Vector3::unit_y());
        let view_proj = proj * view;
        let inv_view_proj = view_proj.invert().unwrap_or(cgmath::Matrix4::identity());
        (view_proj, inv_view_proj)
    }
}

struct AtmosphereState {
    time: f32,
    weather: f32,
    cloud_type: f32,
    lightning_intensity: f32,
    lightning_pos: [f32; 3],
    lightning_color: [f32; 3],
    rain: f32,
    wind: [f32; 2],

    // Simulation vars
    target_time: f32, // Still used for time interpolation
    weather_offset: f32,
    target_weather_offset: f32,
    sim_time: f32,

    rng_seed: u32,
    lightning_timer: f32,
}

impl AtmosphereState {
    fn new() -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        Self {
            time: 0.25,
            weather: 0.0,
            cloud_type: 0.0,
            lightning_intensity: 0.0,
            lightning_pos: [0.0, 150.0, 0.0],
            lightning_color: [1.0, 1.0, 1.0],
            rain: 0.0,
            wind: [0.2, 0.1],
            target_time: 0.25,
            weather_offset: 0.0,
            target_weather_offset: 0.0,
            sim_time: 0.0,
            rng_seed: seed,
            lightning_timer: 2.0,
        }
    }

    fn update(&mut self, dt: f32, camera_pos: [f32; 3]) {
        // Day Cycle
        let day_duration = 60.0;
        self.time += dt / day_duration;
        if self.time > 1.0 {
            self.time -= 1.0;
        }

        // Sync target time to avoid jump if we switch modes manually
        self.target_time = self.time;

        // Weather Sim
        self.sim_time += dt * 0.1;

        let w_noise = self.sim_time.sin()
            + (self.sim_time * 2.3).cos() * 0.5
            + (self.sim_time * 0.7).sin() * 0.2;
        let auto_weather = (w_noise * 0.5 + 0.5).clamp(0.0, 1.0);

        self.weather_offset = lerp(self.weather_offset, self.target_weather_offset, 1.0, dt);

        let target_weather = (auto_weather + self.weather_offset).clamp(0.0, 1.0);
        self.weather = lerp(self.weather, target_weather, 0.5, dt);

        let t_noise = ((self.sim_time * 1.5).cos() * 0.5 + 0.5).clamp(0.0, 1.0);
        self.cloud_type = lerp(self.cloud_type, t_noise, 0.5, dt);

        let target_rain = smoothstep(0.6, 0.8, self.weather);
        self.rain = lerp(self.rain, target_rain, 0.5, dt);

        // Wind
        let target_wind_x = (self.sim_time * 3.0).sin();
        let target_wind_z = (self.sim_time * 2.5).cos();
        let storm_boost = 1.0 + self.weather * 3.0;
        self.wind[0] = lerp(self.wind[0], target_wind_x * storm_boost, 0.1, dt);
        self.wind[1] = lerp(self.wind[1], target_wind_z * storm_boost, 0.1, dt);

        // Lightning
        if self.weather > 0.85 {
            self.lightning_timer -= dt;
            if self.lightning_timer <= 0.0 {
                self.lightning_intensity = 1.0;
                let rx = (random_f32(&mut self.rng_seed) - 0.5) * 4000.0;
                let rz = (random_f32(&mut self.rng_seed) - 0.5) * 4000.0;
                let ry = 140.0 + random_f32(&mut self.rng_seed) * 30.0;
                self.lightning_pos = [camera_pos[0] + rx, ry, camera_pos[2] + rz];

                let r = 0.6 + random_f32(&mut self.rng_seed) * 0.4;
                let g = 0.6 + random_f32(&mut self.rng_seed) * 0.4;
                let b = 1.0;
                self.lightning_color = [r, g, b];

                self.lightning_timer = 0.2 + random_f32(&mut self.rng_seed) * 3.0;
            }
        } else {
            self.lightning_intensity = 0.0;
        }
        self.lightning_intensity = lerp(self.lightning_intensity, 0.0, 8.0, dt);
    }
}

struct CameraController {
    speed: f32,
    sensitivity: f32,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
    is_up_pressed: bool,
    is_down_pressed: bool,
    mouse_captured: bool,
}

impl CameraController {
    fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            speed,
            sensitivity,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
            is_up_pressed: false,
            is_down_pressed: false,
            mouse_captured: false,
        }
    }

    fn process_keyboard(&mut self, key: KeyCode, state: ElementState, atmos: &mut AtmosphereState) {
        let amount = state == ElementState::Pressed;
        match key {
            KeyCode::KeyW => {
                self.is_forward_pressed = amount;
            }
            KeyCode::KeyS => {
                self.is_backward_pressed = amount;
            }
            KeyCode::KeyA => {
                self.is_left_pressed = amount;
            }
            KeyCode::KeyD => {
                self.is_right_pressed = amount;
            }
            KeyCode::Space => {
                self.is_up_pressed = amount;
            }
            KeyCode::ShiftLeft => {
                self.is_down_pressed = amount;
            }

            KeyCode::ArrowUp if amount => {
                atmos.target_weather_offset += 0.1;
                println!("Stormier");
            }
            KeyCode::ArrowDown if amount => {
                atmos.target_weather_offset -= 0.1;
                println!("Clearer");
            }

            KeyCode::KeyE if amount => {
                atmos.target_weather_offset = -1.0;
                println!("Force Clear");
            }
            KeyCode::KeyR if amount => {
                atmos.target_weather_offset = 0.0;
                println!("Neutral");
            }
            KeyCode::KeyT if amount => {
                atmos.target_weather_offset = 1.0;
                println!("Force Storm");
            }

            _ => {}
        }
    }

    fn process_mouse(&mut self, mouse_dx: f64, mouse_dy: f64, camera: &mut Camera) {
        if self.mouse_captured {
            camera.yaw += mouse_dx as f32 * self.sensitivity;
            camera.pitch -= mouse_dy as f32 * self.sensitivity;
            camera.pitch = camera.pitch.clamp(-89.0, 89.0);
        }
    }

    fn update_camera(&mut self, camera: &mut Camera, dt: f32) {
        let (sin_yaw, cos_yaw) = cgmath::Rad::from(cgmath::Deg(camera.yaw)).0.sin_cos();
        let forward = cgmath::Vector3::new(cos_yaw, 0.0, sin_yaw).normalize();
        let right = cgmath::Vector3::new(-sin_yaw, 0.0, cos_yaw).normalize();
        let up = cgmath::Vector3::unit_y();
        let mut velocity = cgmath::Vector3::zero();
        if self.is_forward_pressed {
            velocity += forward;
        }
        if self.is_backward_pressed {
            velocity -= forward;
        }
        if self.is_right_pressed {
            velocity += right;
        }
        if self.is_left_pressed {
            velocity -= right;
        }
        if self.is_up_pressed {
            velocity += up;
        }
        if self.is_down_pressed {
            velocity -= up;
        }
        if velocity.magnitude2() > 0.0 {
            velocity = velocity.normalize();
        }
        camera.eye += velocity * self.speed * dt;
    }
}

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
    _padding1: f32,
    lightning_color: [f32; 3],
    _padding2: f32,
    wind: [f32; 2],
    rain: f32,
    _pad_end: f32,
}

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    cloud_pipeline: wgpu::RenderPipeline,
    blit_pipeline: wgpu::RenderPipeline,
    rain_pipeline: wgpu::RenderPipeline,
    compute_pipeline: wgpu::ComputePipeline,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    _camera_bind_group_layout: wgpu::BindGroupLayout,
    _rain_buffer: wgpu::Buffer,
    particle_bind_group: wgpu::BindGroup,
    offscreen_texture: wgpu::Texture,
    offscreen_view: wgpu::TextureView,
    offscreen_bind_group_layout: wgpu::BindGroupLayout,
    offscreen_bind_group: wgpu::BindGroup,
    offscreen_sampler: wgpu::Sampler,
    empty_bind_group: wgpu::BindGroup, // NEW: Placeholder for Compute Slot 1
    start_time: std::time::Instant,
    last_frame_time: std::time::Instant,
    camera: Camera,
    camera_controller: CameraController,
    atmosphere: AtmosphereState,
    window: Arc<Window>,
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = instance.create_surface(window.clone()).unwrap();
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
                required_features: wgpu::Features::empty()
                    | wgpu::Features::VERTEX_WRITABLE_STORAGE,
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let present_mode = [
            wgpu::PresentMode::Immediate,
            wgpu::PresentMode::Mailbox,
            wgpu::PresentMode::Fifo,
        ]
        .into_iter()
        .find(|&mode| surface_caps.present_modes.contains(&mode))
        .unwrap_or(surface_caps.present_modes[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let camera = Camera::new([0.0, 30.0, 0.0]);
        let camera_controller = CameraController::new(50.0, 0.2);
        let atmosphere = AtmosphereState::new();

        let camera_uniform = CameraUniform {
            view_proj: cgmath::Matrix4::identity().into(),
            inv_view_proj: cgmath::Matrix4::identity().into(),
            camera_pos: [0.0, 0.0, 0.0],
            time: 0.0,
            day_progress: 0.0,
            weather_offset: 0.0,
            cloud_type: 0.0,
            lightning_intensity: 0.0,
            lightning_pos: [0.0, 0.0, 0.0],
            _padding1: 0.0,
            lightning_color: [1.0, 1.0, 1.0],
            _padding2: 0.0,
            wind: [0.0, 0.0],
            rain: 0.0,
            _pad_end: 0.0,
        };
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_group"),
        });

        let mut rng_seed = 12345;
        #[repr(C)]
        #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        struct ParticleInit {
            pos: [f32; 4],
            vel: [f32; 4],
        }
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
        let rain_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Rain Buffer"),
            contents: bytemuck::cast_slice(&particles),
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
        });

        let particle_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            label: Some("Particle Group"),
            layout: &particle_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: rain_buffer.as_entire_binding(),
            }],
        });

        // NEW: Empty Bind Group for Compute Slot 1 (to skip texture)
        let empty_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Empty Layout"),
                entries: &[],
            });
        let empty_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Empty Group"),
            layout: &empty_bind_group_layout,
            entries: &[],
        });

        let offscreen_width = (size.width as f32 * RENDER_SCALE) as u32;
        let offscreen_height = (size.height as f32 * RENDER_SCALE) as u32;
        let texture_desc = wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: offscreen_width,
                height: offscreen_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: config.format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            label: Some("offscreen_texture"),
            view_formats: &[],
        };
        let offscreen_texture = device.create_texture(&texture_desc);
        let offscreen_view = offscreen_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let offscreen_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let offscreen_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
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
                label: Some("offscreen_layout"),
            });
        let offscreen_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &offscreen_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&offscreen_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&offscreen_sampler),
                },
            ],
            label: Some("offscreen_group"),
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        // Compute Pipeline: Uses Empty Layout for Group 1
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Layout"),
                bind_group_layouts: &[
                    &camera_bind_group_layout,
                    &empty_bind_group_layout,
                    &particle_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: Some("cs_rain"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let cloud_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Cloud Layout"),
                bind_group_layouts: &[&camera_bind_group_layout],
                push_constant_ranges: &[],
            });
        let cloud_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Cloud Pipeline"),
            layout: Some(&cloud_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_clouds"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Blit Layout"),
            bind_group_layouts: &[&camera_bind_group_layout, &offscreen_bind_group_layout],
            push_constant_ranges: &[],
        });
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Blit Pipeline"),
            layout: Some(&blit_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_blit"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Rain Pipeline: Uses Camera, Offscreen (unused in vs_rain but kept for index consistency if reused?), Particles
        // vs_rain does NOT access texture.
        // However, we must match shader @group definitions.
        // Shader: Group 0 Camera, Group 1 Texture, Group 2 Particles.
        // vs_rain uses Camera(0) and Particles(2). It ignores Texture(1).
        // So we can bind the texture OR the empty group here too?
        // Actually, let's bind the REAL texture layout here because fs_rain might want to sample depth or something later.
        // But wait, `fs_rain` outputs color.
        // In WGSL for `vs_rain`, `t_diffuse` is defined in scope.
        // So we MUST provide a layout for Group 1.
        // Since we are rendering INTO `offscreen_view` in the same pass, we CANNOT bind `offscreen_view` as Group 1 (Read-Write conflict).
        // SOLUTION: Use `empty_bind_group_layout` for Rain Pipeline as well, effectively unbinding the texture from the rain pass.
        // This requires `shader.wgsl` to NOT define Group 1 as texture if it's not used in rain?
        // No, WGSL is one file.
        // If `vs_rain` and `fs_rain` do NOT reference `t_diffuse`, WGPU reflection *might* allow it to be missing.
        // But to be safe, let's use `empty_bind_group` and in WGSL we won't touch `t_diffuse` in rain functions.

        // Wait, in WGSL `@group(1) ... t_diffuse`. If we use empty layout, WGPU validation will fail if shader declares it.
        // Actually, RenderPipeline only cares about resources USED by the entry points.
        // `vs_rain` and `fs_rain` DO NOT use `t_diffuse`.
        // So we can use `&[&camera_layout, &empty_layout, &particle_layout]`.
        let rain_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Rain Layout"),
            bind_group_layouts: &[
                &camera_bind_group_layout,
                &empty_bind_group_layout,
                &particle_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        let rain_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Rain Pipeline"),
            layout: Some(&rain_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_rain"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_rain"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let mut s = Self {
            surface,
            device,
            queue,
            config,
            size,
            cloud_pipeline,
            blit_pipeline,
            rain_pipeline,
            compute_pipeline,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            _camera_bind_group_layout: camera_bind_group_layout,
            _rain_buffer: rain_buffer,
            particle_bind_group,
            offscreen_texture,
            offscreen_view,
            offscreen_bind_group_layout,
            offscreen_bind_group,
            offscreen_sampler,
            empty_bind_group, // Stored
            start_time: std::time::Instant::now(),
            last_frame_time: std::time::Instant::now(),
            camera,
            camera_controller,
            atmosphere,
            window,
        };
        s.set_capture(true);
        s
    }

    fn set_capture(&mut self, captured: bool) {
        self.camera_controller.mouse_captured = captured;
        self.window.set_cursor_visible(!captured);
        if captured {
            let _ = self
                .window
                .set_cursor_grab(CursorGrabMode::Confined)
                .or_else(|_| self.window.set_cursor_grab(CursorGrabMode::Locked));
        } else {
            let _ = self.window.set_cursor_grab(CursorGrabMode::None);
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            let off_w = (new_size.width as f32 * RENDER_SCALE) as u32;
            let off_h = (new_size.height as f32 * RENDER_SCALE) as u32;
            let tex_desc = wgpu::TextureDescriptor {
                size: wgpu::Extent3d {
                    width: off_w,
                    height: off_h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.config.format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                label: Some("offscreen_texture"),
                view_formats: &[],
            };
            self.offscreen_texture = self.device.create_texture(&tex_desc);
            self.offscreen_view = self
                .offscreen_texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            self.offscreen_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.offscreen_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.offscreen_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.offscreen_sampler),
                    },
                ],
                label: Some("offscreen_group"),
            });
        }
    }

    fn update(&mut self, dt: f32) {
        self.camera_controller.update_camera(&mut self.camera, dt);
        let cam_pos_arr: [f32; 3] = self.camera.eye.into();
        self.atmosphere.update(dt, cam_pos_arr);
        let aspect = self.config.width as f32 / self.config.height as f32;
        let (view_proj, inv_view_proj) = self.camera.build_view_projection_matrix(aspect);
        self.camera_uniform.view_proj = view_proj.into();
        self.camera_uniform.inv_view_proj = inv_view_proj.into();
        self.camera_uniform.camera_pos = self.camera.eye.into();
        self.camera_uniform.time = self.start_time.elapsed().as_secs_f32();
        self.camera_uniform.day_progress = self.atmosphere.time;
        self.camera_uniform.weather_offset = self.atmosphere.weather_offset;
        self.camera_uniform.cloud_type = self.atmosphere.cloud_type;
        self.camera_uniform.lightning_intensity = self.atmosphere.lightning_intensity;
        self.camera_uniform.lightning_pos = self.atmosphere.lightning_pos;
        self.camera_uniform.lightning_color = self.atmosphere.lightning_color;
        self.camera_uniform.wind = self.atmosphere.wind;
        self.camera_uniform.rain = self.atmosphere.rain;
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // 1. Compute
        if self.atmosphere.rain > 0.01 {
            let mut c_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Rain Compute"),
                timestamp_writes: None,
            });
            c_pass.set_pipeline(&self.compute_pipeline);
            c_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            // BIND EMPTY GROUP TO SLOT 1
            c_pass.set_bind_group(1, &self.empty_bind_group, &[]);
            c_pass.set_bind_group(2, &self.particle_bind_group, &[]);
            let workgroups = RAIN_PARTICLES.div_ceil(64);
            c_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // 2. Cloud + Rain (Offscreen)
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Cloud Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.offscreen_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Clouds
            render_pass.set_pipeline(&self.cloud_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.draw(0..3, 0..1);

            // Rain
            if self.atmosphere.rain > 0.01 {
                render_pass.set_pipeline(&self.rain_pipeline);
                render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                // Bind EMPTY group 1 (Texture unused in VS/FS rain)
                render_pass.set_bind_group(1, &self.empty_bind_group, &[]);
                render_pass.set_bind_group(2, &self.particle_bind_group, &[]);
                render_pass.draw(0..4, 0..RAIN_PARTICLES);
            }
        }

        // 3. Blit
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Blit Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            render_pass.set_pipeline(&self.blit_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(1, &self.offscreen_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }
}

// ... App & Main ...
struct App {
    state: Option<State>,
    frame_count: u32,
    last_fps_time: std::time::Instant,
}
impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("WGPU Clouds - Loading..."))
                .unwrap(),
        );
        self.state = Some(pollster::block_on(State::new(window)));
    }
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if let Some(state) = &mut self.state {
            match event {
                WindowEvent::CloseRequested => event_loop.exit(),
                WindowEvent::Resized(size) => state.resize(size),
                WindowEvent::RedrawRequested => {
                    let now = std::time::Instant::now();
                    let dt = (now - state.last_frame_time).as_secs_f32();
                    state.last_frame_time = now;
                    state.update(dt);
                    match state.render() {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                        Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                        Err(_) => {}
                    }
                    self.frame_count += 1;
                    if now.duration_since(self.last_fps_time).as_secs() >= 1 {
                        state
                            .window
                            .set_title(&format!("Minecraft Shader - FPS: {}", self.frame_count));
                        self.frame_count = 0;
                        self.last_fps_time = now;
                    }
                    state.window.request_redraw();
                }
                WindowEvent::KeyboardInput { event: k, .. } => {
                    if k.state == ElementState::Pressed {
                        if let PhysicalKey::Code(KeyCode::Escape) = k.physical_key {
                            state.set_capture(false);
                        } else if let PhysicalKey::Code(code) = k.physical_key {
                            state.camera_controller.process_keyboard(
                                code,
                                k.state,
                                &mut state.atmosphere,
                            );
                        }
                    }
                }
                WindowEvent::MouseInput {
                    state: m_state,
                    button,
                    ..
                } => {
                    if m_state == ElementState::Pressed && button == MouseButton::Left {
                        state.set_capture(true);
                    }
                }
                _ => {}
            }
        }
    }
    fn device_event(
        &mut self,
        _el: &ActiveEventLoop,
        _id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if let Some(state) = &mut self.state
            && let DeviceEvent::MouseMotion { delta } = event
        {
            state
                .camera_controller
                .process_mouse(delta.0, delta.1, &mut state.camera);
        }
    }
}
fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        state: None,
        frame_count: 0,
        last_fps_time: std::time::Instant::now(),
    };
    let _ = event_loop.run_app(&mut app);
}
