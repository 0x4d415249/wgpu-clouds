use std::borrow::Cow;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

// Chunk Constants
const CHUNK_SIZE: u32 = 64;
const TOTAL_VOXELS: usize = (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as usize;
const MAX_FACES: usize = TOTAL_VOXELS * 3;
const MAX_VERTICES: usize = MAX_FACES * 4;
const MAX_INDICES: usize = MAX_FACES * 6;

// Shader
const SHADER_SOURCE: &str = include_str!("shader.wgsl");

struct Camera {
    pos: glam::Vec3,
    yaw: f32,   // Radians
    pitch: f32, // Radians
}

impl Camera {
    fn new(pos: [f32; 3], yaw_deg: f32, pitch_deg: f32) -> Self {
        Self {
            pos: glam::Vec3::from(pos),
            yaw: yaw_deg.to_radians(),
            pitch: pitch_deg.to_radians(),
        }
    }

    fn build_view_proj_matrix(&self, aspect: f32) -> [[f32; 4]; 4] {
        let direction = glam::Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalize();

        let view = glam::Mat4::look_to_rh(self.pos, direction, glam::Vec3::Y);
        let proj = glam::Mat4::perspective_rh(45.0f32.to_radians(), aspect, 0.1, 1000.0);
        (proj * view).to_cols_array_2d()
    }

    // Returns a flattened forward vector (y=0) for movement
    fn get_planar_forward(&self) -> glam::Vec3 {
        glam::Vec3::new(self.yaw.cos(), 0.0, self.yaw.sin()).normalize()
    }

    fn get_right_vector(&self) -> glam::Vec3 {
        let forward = self.get_planar_forward();
        forward.cross(glam::Vec3::Y).normalize()
    }
}

struct App {
    window: Option<Arc<Window>>,
    instance: wgpu::Instance,
    surface: Option<wgpu::Surface<'static>>,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    config: Option<wgpu::SurfaceConfiguration>,

    // Pipelines
    gen_pipeline: Option<wgpu::ComputePipeline>,
    mesh_pipeline: Option<wgpu::ComputePipeline>,
    render_pipeline: Option<wgpu::RenderPipeline>,

    // Bind Groups
    bind_group_0: Option<wgpu::BindGroup>, // Uniforms
    bind_group_1: Option<wgpu::BindGroup>, // Compute Storage 1
    bind_group_2: Option<wgpu::BindGroup>, // Compute Storage 2

    // Buffers
    uniform_buffer: Option<wgpu::Buffer>,
    indirect_buffer: Option<wgpu::Buffer>,
    mesh_meta_buffer: Option<wgpu::Buffer>, // STORED TO RESET
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    depth_texture: Option<wgpu::Texture>,
    depth_view: Option<wgpu::TextureView>,

    // Logic
    camera: Camera,
    keys: std::collections::HashSet<KeyCode>,
    mouse_buttons: std::collections::HashSet<MouseButton>,
    start_time: std::time::Instant,
    mouse_captured: bool,
}

impl App {
    fn new() -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        Self {
            window: None,
            instance,
            surface: None,
            device: None,
            queue: None,
            config: None,
            gen_pipeline: None,
            mesh_pipeline: None,
            render_pipeline: None,
            bind_group_0: None,
            bind_group_1: None,
            bind_group_2: None,
            uniform_buffer: None,
            indirect_buffer: None,
            mesh_meta_buffer: None,
            vertex_buffer: None,
            index_buffer: None,
            depth_texture: None,
            depth_view: None,
            camera: Camera::new([-10.0, 40.0, -10.0], 45.0, -30.0),
            keys: std::collections::HashSet::new(),
            mouse_buttons: std::collections::HashSet::new(),
            start_time: std::time::Instant::now(),
            mouse_captured: false,
        }
    }

    async fn init_gpu(&mut self, window: Arc<Window>) {
        let size = window.inner_size();
        let surface = self.instance.create_surface(window.clone()).unwrap();

        let adapter = self
            .instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find an appropriate adapter");

        let mut required_limits = wgpu::Limits::default();
        required_limits.max_storage_buffer_binding_size = 256 * 1024 * 1024;
        required_limits.max_compute_invocations_per_workgroup = 512;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::INDIRECT_FIRST_INSTANCE
                    | wgpu::Features::VERTEX_WRITABLE_STORAGE,
                required_limits,
                memory_hints: Default::default(),
                ..Default::default()
            })
            .await
            .expect("Failed to create device");

        let config = surface
            .get_default_config(&adapter, size.width, size.height)
            .unwrap();
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Voxel Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER_SOURCE)),
        });

        // --- BUFFERS ---

        let voxel_size = TOTAL_VOXELS * 4;
        let voxel_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Voxel Buffer"),
            size: voxel_size as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let indirect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Indirect Draw Buffer"),
            size: 20,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mesh_meta_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Mesh Metadata"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let vertex_buffer_size = (MAX_VERTICES * 32) as u64;
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Buffer"),
            size: vertex_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });

        let index_buffer_size = (MAX_INDICES * 4) as u64;
        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Index Buffer"),
            size: index_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDEX,
            mapped_at_creation: false,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: 80,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- BIND GROUPS ---

        let bind_group_layout_0 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("BG Layout 0 (Uniforms)"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let bind_group_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group 0"),
            layout: &bind_group_layout_0,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let bind_group_layout_1 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("BG Layout 1 (Compute Storage)"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let bind_group_1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group 1"),
            layout: &bind_group_layout_1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: voxel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: indirect_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: mesh_meta_buffer.as_entire_binding(),
                },
            ],
        });

        let bind_group_layout_2 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("BG Layout 2 (Geometry)"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let bind_group_2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group 2"),
            layout: &bind_group_layout_2,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: index_buffer.as_entire_binding(),
                },
            ],
        });

        // --- PIPELINES ---

        let gen_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Gen Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout_0, &bind_group_layout_1],
            push_constant_ranges: &[],
        });

        let gen_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Generation Pipeline"),
            layout: Some(&gen_pipeline_layout),
            module: &shader,
            entry_point: Some("generate"),
            compilation_options: Default::default(),
            cache: None,
        });

        let mesh_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Mesh Pipeline Layout"),
            bind_group_layouts: &[
                &bind_group_layout_0,
                &bind_group_layout_1,
                &bind_group_layout_2,
            ],
            push_constant_ranges: &[],
        });

        let mesh_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Mesh Pipeline"),
            layout: Some(&mesh_pipeline_layout),
            module: &shader,
            entry_point: Some("mesh"),
            compilation_options: Default::default(),
            cache: None,
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout_0],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 32,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 16,
                            shader_location: 1,
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
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
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
            label: Some("Depth Texture"),
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.surface = Some(surface);
        self.device = Some(device);
        self.queue = Some(queue);
        self.config = Some(config);
        self.gen_pipeline = Some(gen_pipeline);
        self.mesh_pipeline = Some(mesh_pipeline);
        self.render_pipeline = Some(render_pipeline);
        self.bind_group_0 = Some(bind_group_0);
        self.bind_group_1 = Some(bind_group_1);
        self.bind_group_2 = Some(bind_group_2);
        self.uniform_buffer = Some(uniform_buffer);
        self.indirect_buffer = Some(indirect_buffer);
        self.mesh_meta_buffer = Some(mesh_meta_buffer);
        self.vertex_buffer = Some(vertex_buffer);
        self.index_buffer = Some(index_buffer);
        self.depth_texture = Some(depth_texture);
        self.depth_view = Some(depth_view);

        self.window = Some(window);
    }

    fn toggle_mouse_capture(&mut self, captured: bool) {
        if let Some(window) = &self.window {
            if captured {
                let _ = window
                    .set_cursor_grab(CursorGrabMode::Locked)
                    .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
                window.set_cursor_visible(false);
            } else {
                let _ = window.set_cursor_grab(CursorGrabMode::None);
                window.set_cursor_visible(true);
            }
            self.mouse_captured = captured;
        }
    }

    fn update(&mut self) {
        if let Some(queue) = &self.queue {
            // Movement Logic
            if self.mouse_captured {
                let mut base_speed = 0.5; // voxels per frame

                // Speed boost
                if self.keys.contains(&KeyCode::ControlLeft)
                    || self.keys.contains(&KeyCode::ControlRight)
                    || self.mouse_buttons.contains(&MouseButton::Forward)
                {
                    base_speed *= 2.0;
                }

                // Planar Forward (Moves flat on XZ plane)
                let forward = self.camera.get_planar_forward();
                let right = self.camera.get_right_vector();
                let up = glam::Vec3::Y;

                if self.keys.contains(&KeyCode::KeyW) {
                    self.camera.pos += forward * base_speed;
                }
                if self.keys.contains(&KeyCode::KeyS) {
                    self.camera.pos -= forward * base_speed;
                }
                if self.keys.contains(&KeyCode::KeyA) {
                    self.camera.pos -= right * base_speed;
                }
                if self.keys.contains(&KeyCode::KeyD) {
                    self.camera.pos += right * base_speed;
                }

                // Vertical Movement
                if self.keys.contains(&KeyCode::Space) {
                    self.camera.pos += up * base_speed;
                }
                if self.keys.contains(&KeyCode::ShiftLeft)
                    || self.keys.contains(&KeyCode::ShiftRight)
                {
                    self.camera.pos -= up * base_speed;
                }
            }

            // Update Window Title (Debug Info)
            if let Some(window) = &self.window {
                let title = format!(
                    "GPU Voxel Engine | Pos: {:.1}, {:.1}, {:.1} | Look: {:.1}, {:.1}",
                    self.camera.pos.x,
                    self.camera.pos.y,
                    self.camera.pos.z,
                    self.camera.yaw.to_degrees(),
                    self.camera.pitch.to_degrees()
                );
                window.set_title(&title);
            }

            let aspect = self.config.as_ref().unwrap().width as f32
                / self.config.as_ref().unwrap().height as f32;
            let vp = self.camera.build_view_proj_matrix(aspect);
            let time = self.start_time.elapsed().as_secs_f32();

            let mut uniform_data = Vec::new();
            for col in vp {
                uniform_data.extend_from_slice(bytemuck::cast_slice(&col));
            }
            uniform_data.extend_from_slice(bytemuck::bytes_of(&time));
            uniform_data.extend_from_slice(&[0u8; 12]);

            queue.write_buffer(self.uniform_buffer.as_ref().unwrap(), 0, &uniform_data);
        }
    }

    fn render(&mut self) {
        if self.device.is_none() {
            return;
        }
        let device = self.device.as_ref().unwrap();
        let queue = self.queue.as_ref().unwrap();
        let surface = self.surface.as_ref().unwrap();

        let output = match surface.get_current_texture() {
            Ok(output) => output,
            Err(wgpu::SurfaceError::Lost) => {
                // Resize will be handled by next event, but we can try to reconfigure here or just return
                // self.resize(...)
                return;
            }
            Err(_) => return,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // --- RESET COUNTERS (CPU SIDE) ---
        // This is crucial. We must reset the atomic counters before the compute pass starts.
        // Doing it in the shader causes race conditions (threads starting before thread 0 resets).

        // IndirectDrawArgs: [vertex_count=0, instance_count=1, first_index=0, base_vertex=0, first_instance=0]
        let indirect_reset_data = [0u32, 1u32, 0u32, 0u32, 0u32];
        queue.write_buffer(
            self.indirect_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&indirect_reset_data),
        );

        // MeshMetadata: [vertex_counter=0, index_counter=0]
        let meta_reset_data = [0u32, 0u32];
        queue.write_buffer(
            self.mesh_meta_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&meta_reset_data),
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Encoder"),
        });

        // Pass 1a: Generation (Separate Pass)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Gen Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(self.gen_pipeline.as_ref().unwrap());
            cpass.set_bind_group(0, self.bind_group_0.as_ref().unwrap(), &[]);
            cpass.set_bind_group(1, self.bind_group_1.as_ref().unwrap(), &[]);
            let workgroups = CHUNK_SIZE / 8;
            cpass.dispatch_workgroups(workgroups, workgroups, workgroups);
        } // End of pass forces synchronization

        // Pass 1b: Meshing (Separate Pass)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Mesh Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(self.mesh_pipeline.as_ref().unwrap());
            cpass.set_bind_group(0, self.bind_group_0.as_ref().unwrap(), &[]);
            cpass.set_bind_group(1, self.bind_group_1.as_ref().unwrap(), &[]);
            cpass.set_bind_group(2, self.bind_group_2.as_ref().unwrap(), &[]);
            let workgroups = CHUNK_SIZE / 8;
            cpass.dispatch_workgroups(workgroups, workgroups, workgroups);
        }

        // Pass 2: Rendering
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: self.depth_view.as_ref().unwrap(),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rpass.set_pipeline(self.render_pipeline.as_ref().unwrap());
            rpass.set_bind_group(0, self.bind_group_0.as_ref().unwrap(), &[]);
            rpass.set_vertex_buffer(0, self.vertex_buffer.as_ref().unwrap().slice(..));
            rpass.set_index_buffer(
                self.index_buffer.as_ref().unwrap().slice(..),
                wgpu::IndexFormat::Uint32,
            );
            rpass.draw_indexed_indirect(self.indirect_buffer.as_ref().unwrap(), 0);
        }

        queue.submit(Some(encoder.finish()));
        output.present();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let win_attr = Window::default_attributes().with_title("GPU Voxel Engine");
            let window = Arc::new(event_loop.create_window(win_attr).unwrap());

            let mut app_ptr = unsafe { std::ptr::NonNull::new_unchecked(self as *mut Self) };
            let win_clone = window.clone();

            pollster::block_on(unsafe { app_ptr.as_mut().init_gpu(win_clone) });
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if self.mouse_captured {
            if let DeviceEvent::MouseMotion { delta } = event {
                let sensitivity = 0.003;
                // Inverted Pitch: Subtract delta.1 instead of adding
                self.camera.yaw += delta.0 as f32 * sensitivity;
                self.camera.pitch -= delta.1 as f32 * sensitivity;
                self.camera.pitch = self.camera.pitch.clamp(-1.55, 1.55); // Nearly 90 deg
            }
        }
    }

    // Critical for smooth updates on some platforms in Poll mode
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(new_size) => {
                if let (Some(device), Some(surface), Some(config)) =
                    (&self.device, &self.surface, &mut self.config)
                {
                    if new_size.width > 0 && new_size.height > 0 {
                        config.width = new_size.width;
                        config.height = new_size.height;
                        surface.configure(device, config);

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
                            label: Some("Depth Texture"),
                            view_formats: &[],
                        });
                        self.depth_view = Some(
                            depth_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                        );
                        self.depth_texture = Some(depth_texture);
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                self.update();
                self.render();
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if state == ElementState::Pressed && button == MouseButton::Left {
                    self.toggle_mouse_capture(true);
                }
                match state {
                    ElementState::Pressed => {
                        self.mouse_buttons.insert(button);
                    }
                    ElementState::Released => {
                        self.mouse_buttons.remove(&button);
                    }
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state,
                        ..
                    },
                ..
            } => {
                if code == KeyCode::Escape && state == ElementState::Pressed {
                    self.toggle_mouse_capture(false);
                }
                match state {
                    ElementState::Pressed => {
                        self.keys.insert(code);
                    }
                    ElementState::Released => {
                        self.keys.remove(&code);
                    }
                }
            }
            _ => (),
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
