use cgmath::prelude::*;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::event::*;
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, WindowBuilder};

// --- Camera Controller Logic ---

struct Camera {
    eye: cgmath::Point3<f32>,
    yaw: f32,
    pitch: f32,
}

impl Camera {
    fn new(pos: [f32; 3]) -> Self {
        Self {
            eye: cgmath::Point3::new(pos[0], pos[1], pos[2]),
            yaw: -90.0, // -90 is looking along -Z (Forward in OpenGL/WGPU conventions)
            pitch: 0.0, // Level horizon
        }
    }

    fn build_view_projection_matrix(
        &self,
        aspect: f32,
    ) -> (cgmath::Matrix4<f32>, cgmath::Matrix4<f32>) {
        // Minecraft uses a high FOV usually, around 70-90
        let proj = cgmath::perspective(cgmath::Deg(85.0), aspect, 0.1, 2000.0);

        let (sin_pitch, cos_pitch) = cgmath::Rad::from(cgmath::Deg(self.pitch)).0.sin_cos();
        let (sin_yaw, cos_yaw) = cgmath::Rad::from(cgmath::Deg(self.yaw)).0.sin_cos();

        // Calculate Forward Vector
        let forward =
            cgmath::Vector3::new(cos_pitch * cos_yaw, sin_pitch, cos_pitch * sin_yaw).normalize();

        let view = cgmath::Matrix4::look_to_rh(self.eye, forward, cgmath::Vector3::unit_y());

        let view_proj = proj * view;
        let inv_view_proj = view_proj.invert().unwrap_or(cgmath::Matrix4::identity());

        (view_proj, inv_view_proj)
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

    fn process_keyboard(&mut self, key: KeyCode, state: ElementState) -> bool {
        let amount = state == ElementState::Pressed;
        match key {
            KeyCode::KeyW => {
                self.is_forward_pressed = amount;
                true
            }
            KeyCode::KeyS => {
                self.is_backward_pressed = amount;
                true
            }
            KeyCode::KeyA => {
                self.is_left_pressed = amount;
                true
            }
            KeyCode::KeyD => {
                self.is_right_pressed = amount;
                true
            }
            KeyCode::Space => {
                self.is_up_pressed = amount;
                true
            }
            KeyCode::ShiftLeft => {
                self.is_down_pressed = amount;
                true
            }
            _ => false,
        }
    }

    fn process_mouse(&mut self, mouse_dx: f64, mouse_dy: f64, camera: &mut Camera) {
        camera.yaw += mouse_dx as f32 * self.sensitivity;
        camera.pitch -= mouse_dy as f32 * self.sensitivity;
        camera.pitch = camera.pitch.clamp(-89.0, 89.0);
    }

    fn update_camera(&self, camera: &mut Camera, dt: f32) {
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

// --- WGPU Setup ---

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    time: f32,
}

struct State<'window> {
    surface: wgpu::Surface<'window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    start_time: std::time::Instant,
    last_frame_time: std::time::Instant,
    camera: Camera,
    camera_controller: CameraController,
}

impl<'window> State<'window> {
    async fn new(window: &'window winit::window::Window) -> Self {
        let size = window.inner_size();

        // Lock cursor for FPS control
        let _ = window
            .set_cursor_grab(CursorGrabMode::Confined)
            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Locked));
        window.set_cursor_visible(false);

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let camera = Camera::new([0.0, 1.5, 0.0]); // Start at eye level (1.5 blocks up)
        let camera_controller = CameraController::new(20.0, 0.2); // Fast speed, standard sensitivity

        let camera_uniform = CameraUniform {
            view_proj: cgmath::Matrix4::identity().into(),
            inv_view_proj: cgmath::Matrix4::identity().into(),
            camera_pos: [0.0, 0.0, 0.0],
            time: 0.0,
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
                label: Some("camera_bind_group_layout"),
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            start_time: std::time::Instant::now(),
            last_frame_time: std::time::Instant::now(),
            camera,
            camera_controller,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state,
                        ..
                    },
                ..
            } => self.camera_controller.process_keyboard(*key, *state),
            _ => false,
        }
    }

    fn update(&mut self) {
        let now = std::time::Instant::now();
        let dt = (now - self.last_frame_time).as_secs_f32();
        self.last_frame_time = now;

        self.camera_controller.update_camera(&mut self.camera, dt);

        let aspect = self.config.width as f32 / self.config.height as f32;
        let (view_proj, inv_view_proj) = self.camera.build_view_projection_matrix(aspect);

        self.camera_uniform.view_proj = view_proj.into();
        self.camera_uniform.inv_view_proj = inv_view_proj.into();
        self.camera_uniform.camera_pos = self.camera.eye.into();
        self.camera_uniform.time = self.start_time.elapsed().as_secs_f32();

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

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();

    let window = Arc::new(
        WindowBuilder::new()
            .with_title("Minecraft Shader Renderer")
            .build(&event_loop)
            .unwrap(),
    );

    let mut state = pollster::block_on(State::new(&window));
    event_loop.set_control_flow(ControlFlow::Poll);
    let window_clone = window.clone();

    let _ = event_loop.run(move |event, target| match event {
        Event::DeviceEvent {
            event: DeviceEvent::MouseMotion { delta, .. },
            ..
        } => {
            state
                .camera_controller
                .process_mouse(delta.0, delta.1, &mut state.camera);
        }
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window_clone.id() => {
            if !state.input(event) {
                match event {
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        event:
                            KeyEvent {
                                state: ElementState::Pressed,
                                physical_key: PhysicalKey::Code(KeyCode::Escape),
                                ..
                            },
                        ..
                    } => target.exit(),
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::RedrawRequested => {
                        state.update();
                        match state.render() {
                            Ok(_) => {}
                            Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                            Err(wgpu::SurfaceError::OutOfMemory) => target.exit(),
                            Err(e) => eprintln!("{:?}", e),
                        }
                    }
                    _ => {}
                }
            }
        }
        Event::AboutToWait => {
            window_clone.request_redraw();
        }
        _ => {}
    });
}
