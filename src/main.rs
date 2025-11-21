use cgmath::prelude::*;
use std::borrow::Cow;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

// RENDER_SCALE: 0.5 = 50% resolution (Massive FPS boost)
const RENDER_SCALE: f32 = 0.5;

fn lerp(start: f32, end: f32, speed: f32, dt: f32) -> f32 {
    start + (end - start) * (speed * dt).clamp(0.0, 1.0)
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
    current_time: f32,
    current_weather: f32,
    current_cloud_type: f32,
    target_time: f32,
    target_weather: f32,
    target_cloud_type: f32,
}

impl AtmosphereState {
    fn new() -> Self {
        Self {
            current_time: 0.1,
            current_weather: 0.0,
            current_cloud_type: 0.0,
            target_time: 0.1,
            target_weather: 0.0,
            target_cloud_type: 0.0,
        }
    }

    fn update(&mut self, dt: f32) {
        self.current_time = lerp(self.current_time, self.target_time, 1.0, dt);
        self.current_weather = lerp(self.current_weather, self.target_weather, 1.0, dt);
        self.current_cloud_type = lerp(self.current_cloud_type, self.target_cloud_type, 1.0, dt);
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

            KeyCode::KeyE if amount => {
                atmos.target_time = 0.15;
                atmos.target_weather = 0.0;
                atmos.target_cloud_type = 0.0;
                println!("Sunny");
            }
            KeyCode::KeyR if amount => {
                atmos.target_time = 0.15;
                atmos.target_weather = 0.6;
                atmos.target_cloud_type = 1.0;
                println!("Overcast");
            }
            KeyCode::KeyT if amount => {
                atmos.target_time = 0.15;
                atmos.target_weather = 1.0;
                atmos.target_cloud_type = 0.5;
                println!("Storm");
            }
            KeyCode::KeyY if amount => {
                atmos.target_time = 0.23;
                atmos.target_weather = 0.3;
                atmos.target_cloud_type = 0.0;
                println!("Sunset");
            }
            KeyCode::KeyU if amount => {
                atmos.target_time = 0.5;
                println!("Night");
            }
            KeyCode::KeyG if amount => {
                atmos.target_cloud_type = 0.0;
                println!("Cumulus");
            }
            KeyCode::KeyH if amount => {
                atmos.target_cloud_type = 1.0;
                println!("Stratus");
            }
            _ => {}
        }
    }

    fn process_mouse(&mut self, mouse_dx: f64, mouse_dy: f64, camera: &mut Camera) {
        camera.yaw += mouse_dx as f32 * self.sensitivity;
        camera.pitch -= mouse_dy as f32 * self.sensitivity;
        camera.pitch = camera.pitch.clamp(-89.0, 89.0);
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
    weather: f32,
    cloud_type: f32,
    padding: f32,
}

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,

    cloud_pipeline: wgpu::RenderPipeline,
    blit_pipeline: wgpu::RenderPipeline,

    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    camera_bind_group_layout: wgpu::BindGroupLayout,

    offscreen_texture: wgpu::Texture,
    offscreen_view: wgpu::TextureView,
    offscreen_bind_group_layout: wgpu::BindGroupLayout,
    offscreen_bind_group: wgpu::BindGroup,
    offscreen_sampler: wgpu::Sampler,

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
                required_features: wgpu::Features::empty(),
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

        // --- VSYNC LOGIC ---
        // Try to find Immediate (VSync Off), fallback to Mailbox (Uncapped), fallback to Fifo (VSync On)
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
            present_mode, // Use the selected mode
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // --- Camera ---
        let camera = Camera::new([0.0, 30.0, 0.0]);
        let camera_controller = CameraController::new(50.0, 0.2);
        let atmosphere = AtmosphereState::new();
        let camera_uniform = CameraUniform {
            view_proj: cgmath::Matrix4::identity().into(),
            inv_view_proj: cgmath::Matrix4::identity().into(),
            camera_pos: [0.0, 0.0, 0.0],
            time: 0.0,
            day_progress: 0.0,
            weather: 0.0,
            cloud_type: 0.0,
            padding: 0.0,
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
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_group"),
        });

        // --- Offscreen ---
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

        // --- Pipelines ---
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        // Cloud Pipeline
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

        // Blit Pipeline
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

        Self {
            surface,
            device,
            queue,
            config,
            size,
            cloud_pipeline,
            blit_pipeline,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            camera_bind_group_layout,
            offscreen_texture,
            offscreen_view,
            offscreen_bind_group_layout,
            offscreen_bind_group,
            offscreen_sampler,
            start_time: std::time::Instant::now(),
            last_frame_time: std::time::Instant::now(),
            camera,
            camera_controller,
            atmosphere,
            window,
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
        self.atmosphere.update(dt);

        let aspect = self.config.width as f32 / self.config.height as f32;
        let (view_proj, inv_view_proj) = self.camera.build_view_projection_matrix(aspect);

        self.camera_uniform.view_proj = view_proj.into();
        self.camera_uniform.inv_view_proj = inv_view_proj.into();
        self.camera_uniform.camera_pos = self.camera.eye.into();
        self.camera_uniform.time = self.start_time.elapsed().as_secs_f32();
        self.camera_uniform.day_progress = self.atmosphere.current_time;
        self.camera_uniform.weather = self.atmosphere.current_weather;
        self.camera_uniform.cloud_type = self.atmosphere.current_cloud_type;

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

        // 1. CLOUDS (Offscreen)
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
            render_pass.set_pipeline(&self.cloud_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }

        // 2. BLIT (Screen)
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

struct App {
    state: Option<State>,
    // FPS Tracking
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

                    // FPS Calculation
                    self.frame_count += 1;
                    if now.duration_since(self.last_fps_time).as_secs() >= 1 {
                        let fps = self.frame_count;
                        let title = format!("Minecraft Shader - FPS: {}", fps);
                        state.window.set_title(&title);
                        self.frame_count = 0;
                        self.last_fps_time = now;
                    }

                    state.window.request_redraw();
                }
                WindowEvent::KeyboardInput { event: k, .. } => {
                    if let PhysicalKey::Code(KeyCode::Escape) = k.physical_key {
                        event_loop.exit();
                    }
                    if let PhysicalKey::Code(code) = k.physical_key {
                        state.camera_controller.process_keyboard(
                            code,
                            k.state,
                            &mut state.atmosphere,
                        );
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
        if let Some(state) = &mut self.state {
            if let DeviceEvent::MouseMotion { delta } = event {
                state
                    .camera_controller
                    .process_mouse(delta.0, delta.1, &mut state.camera);
            }
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
