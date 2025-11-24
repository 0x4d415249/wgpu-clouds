use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, mpsc};
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

// Chunk Constants
const CHUNK_SIZE: u32 = 64;
const TOTAL_VOXELS: usize = (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as usize;
const VOXEL_INTS: usize = TOTAL_VOXELS / 4; // Packed 4 voxels per u32

// Memory Optimization
const MAX_FACES_PER_CHUNK: usize = 30_000;
const MAX_VERTICES: usize = MAX_FACES_PER_CHUNK * 4;
const MAX_INDICES: usize = MAX_FACES_PER_CHUNK * 6;

// Rendering Settings
const RENDER_DISTANCE: i32 = 16;
const UNLOAD_DISTANCE: i32 = 20;
const CHUNKS_PER_FRAME: usize = 128;
const COMPUTE_PER_FRAME: usize = 128;

// Shader
const SHADER_SOURCE: &str = include_str!("shader.wgsl");

struct Camera {
    pos: glam::Vec3,
    yaw: f32,
    pitch: f32,
}

impl Camera {
    fn new(pos: [f32; 3], yaw_deg: f32, pitch_deg: f32) -> Self {
        Self {
            pos: glam::Vec3::from(pos),
            yaw: yaw_deg.to_radians(),
            pitch: pitch_deg.to_radians(),
        }
    }

    fn build_view_proj_matrix(&self, aspect: f32) -> (glam::Mat4, glam::Mat4) {
        let direction = self.get_forward_vector();
        let view = glam::Mat4::look_to_rh(self.pos, direction, glam::Vec3::Y);
        let proj = glam::Mat4::perspective_rh(45.0f32.to_radians(), aspect, 0.1, 5000.0);

        let correction = glam::Mat4::from_cols(
            glam::Vec4::new(1.0, 0.0, 0.0, 0.0),
            glam::Vec4::new(0.0, 1.0, 0.0, 0.0),
            glam::Vec4::new(0.0, 0.0, 0.5, 0.0),
            glam::Vec4::new(0.0, 0.0, 0.5, 1.0),
        );

        let vp = correction * proj * view;
        (vp, vp.inverse())
    }

    fn get_forward_vector(&self) -> glam::Vec3 {
        glam::Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalize()
    }

    fn get_planar_forward(&self) -> glam::Vec3 {
        glam::Vec3::new(self.yaw.cos(), 0.0, self.yaw.sin()).normalize()
    }

    fn get_right_vector(&self) -> glam::Vec3 {
        let forward = self.get_planar_forward();
        forward.cross(glam::Vec3::Y).normalize()
    }

    fn get_chunk_coords(&self) -> (i32, i32) {
        (
            (self.pos.x / CHUNK_SIZE as f32).floor() as i32,
            (self.pos.z / CHUNK_SIZE as f32).floor() as i32,
        )
    }
}

struct Chunk {
    x: i32,
    z: i32,
    generated: bool,
    meshed: bool,
    dist_sq: i32,
    synced_to_cpu: bool,

    bind_group_chunk_data: wgpu::BindGroup,
    bind_group_storage: wgpu::BindGroup,
    bind_group_geometry: wgpu::BindGroup,

    indirect_buffer: wgpu::Buffer,
    mesh_meta_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    voxel_buffer: wgpu::Buffer,
    _uniform_buffer: wgpu::Buffer,
}

struct ReadbackRequest {
    coords: (i32, i32),
    buffer: wgpu::Buffer,
}

struct App {
    window: Option<Arc<Window>>,
    instance: wgpu::Instance,
    surface: Option<wgpu::Surface<'static>>,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    config: Option<wgpu::SurfaceConfiguration>,

    // Pipelines
    clear_pipeline: Option<wgpu::ComputePipeline>,
    gen_pipeline: Option<wgpu::ComputePipeline>,
    mesh_pipeline: Option<wgpu::ComputePipeline>,
    render_pipeline: Option<wgpu::RenderPipeline>,
    sky_pipeline: Option<wgpu::RenderPipeline>,

    layout_chunk: Option<wgpu::BindGroupLayout>,
    layout_storage: Option<wgpu::BindGroupLayout>,
    layout_geometry: Option<wgpu::BindGroupLayout>,

    bind_group_global: Option<wgpu::BindGroup>,
    global_uniform_buffer: Option<wgpu::Buffer>,
    depth_view: Option<wgpu::TextureView>,

    chunks: HashMap<(i32, i32), Chunk>,
    chunk_pool: Vec<Chunk>,
    voxel_cache: HashMap<(i32, i32), Arc<Vec<u8>>>,

    active_readback: Option<ReadbackRequest>,
    readback_tx: mpsc::Sender<()>,
    readback_rx: mpsc::Receiver<()>,

    camera: Camera,
    keys: HashSet<KeyCode>,
    mouse_buttons: HashSet<MouseButton>,
    start_time: std::time::Instant,
    mouse_captured: bool,
}

impl App {
    fn new() -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let (readback_tx, readback_rx) = mpsc::channel();

        Self {
            window: None,
            instance,
            surface: None,
            device: None,
            queue: None,
            config: None,
            clear_pipeline: None,
            gen_pipeline: None,
            mesh_pipeline: None,
            render_pipeline: None,
            sky_pipeline: None,
            layout_chunk: None,
            layout_storage: None,
            layout_geometry: None,
            bind_group_global: None,
            global_uniform_buffer: None,
            depth_view: None,
            chunks: HashMap::new(),
            chunk_pool: Vec::new(),
            voxel_cache: HashMap::new(),
            active_readback: None,
            readback_tx,
            readback_rx,
            camera: Camera::new([0.0, 150.0, 0.0], 45.0, -30.0),
            keys: HashSet::new(),
            mouse_buttons: HashSet::new(),
            start_time: std::time::Instant::now(),
            mouse_captured: false,
        }
    }

    fn configure_surface(&mut self) {
        if let (Some(window), Some(device), Some(surface), Some(config)) =
            (&self.window, &self.device, &self.surface, &mut self.config)
        {
            let size = window.inner_size();
            if size.width > 0 && size.height > 0 {
                config.width = size.width;
                config.height = size.height;
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
                self.depth_view =
                    Some(depth_texture.create_view(&wgpu::TextureViewDescriptor::default()));
            }
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
            .expect("Failed to find adapter");

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

        let width = size.width.max(1);
        let height = size.height.max(1);
        let config = surface.get_default_config(&adapter, width, height).unwrap();

        self.device = Some(device);
        self.queue = Some(queue);
        self.surface = Some(surface);
        self.config = Some(config);
        self.window = Some(window);

        self.configure_surface();

        let device = self.device.as_ref().unwrap();

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER_SOURCE)),
        });

        // --- LAYOUTS ---
        let layout_global = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Global Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE
                    | wgpu::ShaderStages::VERTEX
                    | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let layout_chunk = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Chunk Data Layout"),
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
        let layout_storage = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Storage Layout"),
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
        let layout_geometry = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Geometry Output Layout"),
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

        // --- PIPELINES ---
        let gen_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Gen Layout"),
            bind_group_layouts: &[&layout_global, &layout_chunk, &layout_storage],
            push_constant_ranges: &[],
        });

        // 0. Clear Pipeline (Fast zeroing of voxels)
        let clear_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Clear Pipe"),
            layout: Some(&gen_layout), // Re-uses gen layout (needs storage 0)
            module: &shader,
            entry_point: Some("clear_voxels"),
            compilation_options: Default::default(),
            cache: None,
        });

        // 1. Gen Pipeline
        let gen_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Gen Pipe"),
            layout: Some(&gen_layout),
            module: &shader,
            entry_point: Some("generate"),
            compilation_options: Default::default(),
            cache: None,
        });

        let mesh_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Mesh Layout"),
            bind_group_layouts: &[
                &layout_global,
                &layout_chunk,
                &layout_storage,
                &layout_geometry,
            ],
            push_constant_ranges: &[],
        });
        let mesh_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Mesh Pipe"),
            layout: Some(&mesh_layout),
            module: &shader,
            entry_point: Some("mesh"),
            compilation_options: Default::default(),
            cache: None,
        });

        let render_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Layout"),
            bind_group_layouts: &[&layout_global, &layout_chunk],
            push_constant_ranges: &[],
        });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipe"),
            layout: Some(&render_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 8,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Uint32,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Uint32,
                            offset: 4,
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
                    format: self.config.as_ref().unwrap().format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
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

        let sky_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sky Layout"),
            bind_group_layouts: &[&layout_global],
            push_constant_ranges: &[],
        });
        let sky_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sky Pipeline"),
            layout: Some(&sky_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_sky"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_sky"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.config.as_ref().unwrap().format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let global_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Global Uniforms"),
            size: 256,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group_global = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Global BG"),
            layout: &layout_global,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: global_uniform_buffer.as_entire_binding(),
            }],
        });

        self.clear_pipeline = Some(clear_pipeline);
        self.gen_pipeline = Some(gen_pipeline);
        self.mesh_pipeline = Some(mesh_pipeline);
        self.render_pipeline = Some(render_pipeline);
        self.sky_pipeline = Some(sky_pipeline);
        self.layout_chunk = Some(layout_chunk);
        self.layout_storage = Some(layout_storage);
        self.layout_geometry = Some(layout_geometry);
        self.bind_group_global = Some(bind_group_global);
        self.global_uniform_buffer = Some(global_uniform_buffer);

        self.update();
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn process_readback(&mut self) {
        if self.active_readback.is_some() {
            let device = self.device.as_ref().unwrap();
            let _ = device.poll(wgpu::PollType::Poll);

            if let Ok(_) = self.readback_rx.try_recv() {
                if let Some(req) = self.active_readback.take() {
                    let slice = req.buffer.slice(..);
                    {
                        let view = slice.get_mapped_range();
                        let data = view.to_vec();
                        self.voxel_cache.insert(req.coords, Arc::new(data));

                        if let Some(chunk) = self.chunks.get_mut(&req.coords) {
                            chunk.synced_to_cpu = true;
                        }
                    }
                    req.buffer.unmap();
                }
            }
        }

        if self.active_readback.is_none() {
            if let Some((&coords, chunk)) = self
                .chunks
                .iter()
                .find(|(_, c)| c.generated && !c.synced_to_cpu)
            {
                let device = self.device.as_ref().unwrap();
                let size = (TOTAL_VOXELS * 4) as u64;

                let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Readback Staging"),
                    size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });

                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Readback"),
                });
                encoder.copy_buffer_to_buffer(&chunk.voxel_buffer, 0, &staging_buffer, 0, size);
                self.queue.as_ref().unwrap().submit(Some(encoder.finish()));

                let slice = staging_buffer.slice(..);
                let tx = self.readback_tx.clone();
                slice.map_async(wgpu::MapMode::Read, move |_| {
                    let _ = tx.send(());
                });

                self.active_readback = Some(ReadbackRequest {
                    coords,
                    buffer: staging_buffer,
                });
            }
        }
    }

    fn update_chunks(&mut self) {
        if self.device.is_none() {
            return;
        }

        let (cx, cz) = self.camera.get_chunk_coords();

        // Unload & Recycle
        let to_remove: Vec<(i32, i32)> = self
            .chunks
            .iter()
            .filter(|(coords, _)| {
                let dx = coords.0 - cx;
                let dz = coords.1 - cz;
                (dx * dx + dz * dz) >= UNLOAD_DISTANCE * UNLOAD_DISTANCE
            })
            .map(|(&c, _)| c)
            .collect();

        for coords in to_remove {
            if let Some(mut chunk) = self.chunks.remove(&coords) {
                chunk.generated = false;
                chunk.meshed = false;
                chunk.synced_to_cpu = false;
                self.chunk_pool.push(chunk);
            }
        }

        // Find missing
        let mut missing_chunks = Vec::new();
        for x in (cx - RENDER_DISTANCE)..=(cx + RENDER_DISTANCE) {
            for z in (cz - RENDER_DISTANCE)..=(cz + RENDER_DISTANCE) {
                if (x - cx).pow(2) + (z - cz).pow(2) > RENDER_DISTANCE.pow(2) {
                    continue;
                }
                if !self.chunks.contains_key(&(x, z)) {
                    let dist_sq = (x - cx).pow(2) + (z - cz).pow(2);
                    missing_chunks.push((dist_sq, x, z));
                }
            }
        }

        missing_chunks.sort_by_key(|k| k.0);

        // Instantiate new chunks
        let device = self.device.as_ref().unwrap();
        let queue = self.queue.as_ref().unwrap();

        for (_, x, z) in missing_chunks.iter().take(CHUNKS_PER_FRAME) {
            let mut chunk = if let Some(mut c) = self.chunk_pool.pop() {
                c.x = *x;
                c.z = *z;
                let offset_data = [*x * 64, 0, *z * 64, 0];
                queue.write_buffer(&c._uniform_buffer, 0, bytemuck::cast_slice(&offset_data));
                c
            } else {
                self.create_chunk(device, *x, *z)
            };

            chunk.dist_sq = (x - cx).pow(2) + (z - cz).pow(2);

            // If cached, we load from CPU and skip generation
            if let Some(data) = self.voxel_cache.get(&(*x, *z)) {
                queue.write_buffer(&chunk.voxel_buffer, 0, data);
                chunk.generated = true;
                chunk.synced_to_cpu = true;
            }
            // If NOT cached, we do NOT clear with write_buffer anymore.
            // We will use clear_pipeline in render()

            self.chunks.insert((*x, *z), chunk);
        }

        for ((x, z), chunk) in self.chunks.iter_mut() {
            chunk.dist_sq = (x - cx).pow(2) + (z - cz).pow(2);
        }
    }

    fn create_chunk(&self, device: &wgpu::Device, x: i32, z: i32) -> Chunk {
        let voxel_size = (TOTAL_VOXELS * 4) as u64;
        let voxel_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Voxel"),
            size: voxel_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let indirect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Indirect"),
            size: 20,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mesh_meta = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Meta"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex"),
            size: (MAX_VERTICES * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });
        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Index"),
            size: (MAX_INDICES * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDEX,
            mapped_at_creation: false,
        });

        let chunk_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Chunk Uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let queue = self.queue.as_ref().unwrap();
        let offset_data = [x * 64, 0, z * 64, 0];
        queue.write_buffer(&chunk_uniform, 0, bytemuck::cast_slice(&offset_data));

        let bg_chunk = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Chunk Data BG"),
            layout: self.layout_chunk.as_ref().unwrap(),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: chunk_uniform.as_entire_binding(),
            }],
        });

        let bg_storage = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Chunk Storage BG"),
            layout: self.layout_storage.as_ref().unwrap(),
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
                    resource: mesh_meta.as_entire_binding(),
                },
            ],
        });

        let bg_geom = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Chunk Geometry BG"),
            layout: self.layout_geometry.as_ref().unwrap(),
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

        Chunk {
            x,
            z,
            generated: false,
            meshed: false,
            synced_to_cpu: false,
            dist_sq: 0,
            bind_group_chunk_data: bg_chunk,
            bind_group_storage: bg_storage,
            bind_group_geometry: bg_geom,
            indirect_buffer,
            mesh_meta_buffer: mesh_meta,
            vertex_buffer,
            index_buffer,
            voxel_buffer,
            _uniform_buffer: chunk_uniform,
        }
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
        if self.mouse_captured {
            let mut base_speed = 5.0;
            if self.keys.contains(&KeyCode::ControlLeft)
                || self.mouse_buttons.contains(&MouseButton::Forward)
            {
                base_speed = 200.0;
            }
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
            if self.keys.contains(&KeyCode::Space) {
                self.camera.pos += up * base_speed;
            }
            if self.keys.contains(&KeyCode::ShiftLeft) {
                self.camera.pos -= up * base_speed;
            }
        }

        self.update_chunks();
        self.process_readback();

        if let Some(queue) = &self.queue {
            if let Some(window) = &self.window {
                window.set_title(&format!(
                    "GPU Voxel Engine | Pos: {:.1}, {:.1}, {:.1} | Chunks: {} | Pool: {} | Cache: {}",
                    self.camera.pos.x,
                    self.camera.pos.y,
                    self.camera.pos.z,
                    self.chunks.len(),
                    self.chunk_pool.len(),
                    self.voxel_cache.len()
                ));
            }

            let aspect = self.config.as_ref().unwrap().width as f32
                / self.config.as_ref().unwrap().height as f32;
            let (vp, inv_vp) = self.camera.build_view_proj_matrix(aspect);
            let time = self.start_time.elapsed().as_secs_f32();

            let mut uniform_data = Vec::new();
            uniform_data.extend_from_slice(bytemuck::cast_slice(&vp.to_cols_array_2d()));
            uniform_data.extend_from_slice(bytemuck::cast_slice(&inv_vp.to_cols_array_2d()));
            uniform_data.extend_from_slice(bytemuck::cast_slice(&self.camera.pos.to_array()));
            uniform_data.extend_from_slice(&[0u8; 4]);
            uniform_data.extend_from_slice(bytemuck::bytes_of(&time));
            uniform_data.extend_from_slice(&[0u8; 12]);

            queue.write_buffer(
                self.global_uniform_buffer.as_ref().unwrap(),
                0,
                &uniform_data,
            );
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        if self.device.is_none() {
            return Ok(());
        }
        let device = self.device.as_ref().unwrap();
        let queue = self.queue.as_ref().unwrap();
        let surface = self.surface.as_ref().unwrap();

        let output = match surface.get_current_texture() {
            Ok(frame) => frame,
            Err(wgpu::SurfaceError::Outdated) | Err(wgpu::SurfaceError::Lost) => {
                return Err(wgpu::SurfaceError::Outdated);
            }
            Err(e) => return Err(e),
        };

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Sort by distance for prioritization
        let mut chunks_to_gen: Vec<(&(i32, i32), &Chunk)> =
            self.chunks.iter().filter(|(_, c)| !c.generated).collect();
        chunks_to_gen.sort_by_key(|(_, c)| c.dist_sq);
        let gen_target: Vec<(i32, i32)> = chunks_to_gen
            .iter()
            .take(COMPUTE_PER_FRAME)
            .map(|(k, _)| **k)
            .collect();

        let mut chunks_to_mesh: Vec<(&(i32, i32), &Chunk)> = self
            .chunks
            .iter()
            .filter(|(_, c)| c.generated && !c.meshed)
            .collect();
        chunks_to_mesh.sort_by_key(|(_, c)| c.dist_sq);
        let mesh_target: Vec<(i32, i32)> = chunks_to_mesh
            .iter()
            .take(COMPUTE_PER_FRAME)
            .map(|(k, _)| **k)
            .collect();

        // Reset buffers
        for (x, z) in &mesh_target {
            if let Some(chunk) = self.chunks.get(&(*x, *z)) {
                let _indirect_reset = [0u32, 1u32, 0u32, 0u32, 0u32];
                let _meta_reset = [0u32, 0u32];
                queue.write_buffer(
                    &chunk.indirect_buffer,
                    0,
                    bytemuck::cast_slice(&_indirect_reset),
                );
                queue.write_buffer(
                    &chunk.mesh_meta_buffer,
                    0,
                    bytemuck::cast_slice(&_meta_reset),
                );
            }
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Encoder"),
        });

        // Pass 0: Clear (Fast GPU Zeroing)
        if !gen_target.is_empty() {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Clear"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(self.clear_pipeline.as_ref().unwrap());
            cpass.set_bind_group(0, self.bind_group_global.as_ref().unwrap(), &[]);

            for (x, z) in &gen_target {
                if let Some(chunk) = self.chunks.get(&(*x, *z)) {
                    cpass.set_bind_group(1, &chunk.bind_group_chunk_data, &[]);
                    cpass.set_bind_group(2, &chunk.bind_group_storage, &[]);
                    // Dispatch: 256 items per group. 65536 total items (VOXEL_INTS).
                    // 65536 / 256 = 256 groups.
                    cpass.dispatch_workgroups(256, 1, 1);
                }
            }
        }

        // Pass 1: Gen
        if !gen_target.is_empty() {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Gen"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(self.gen_pipeline.as_ref().unwrap());
            cpass.set_bind_group(0, self.bind_group_global.as_ref().unwrap(), &[]);

            for (x, z) in &gen_target {
                if let Some(chunk) = self.chunks.get(&(*x, *z)) {
                    cpass.set_bind_group(1, &chunk.bind_group_chunk_data, &[]);
                    cpass.set_bind_group(2, &chunk.bind_group_storage, &[]);
                    cpass.dispatch_workgroups(CHUNK_SIZE / 8, CHUNK_SIZE / 8, CHUNK_SIZE / 8);
                }
            }
        }

        // Pass 2: Mesh
        if !mesh_target.is_empty() {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Mesh"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(self.mesh_pipeline.as_ref().unwrap());
            cpass.set_bind_group(0, self.bind_group_global.as_ref().unwrap(), &[]);
            for (x, z) in &mesh_target {
                if let Some(chunk) = self.chunks.get(&(*x, *z)) {
                    cpass.set_bind_group(1, &chunk.bind_group_chunk_data, &[]);
                    cpass.set_bind_group(2, &chunk.bind_group_storage, &[]);
                    cpass.set_bind_group(3, &chunk.bind_group_geometry, &[]);
                    cpass.dispatch_workgroups(CHUNK_SIZE / 8, CHUNK_SIZE / 8, CHUNK_SIZE / 8);
                }
            }
        }

        // Mark flags
        for (x, z) in gen_target {
            if let Some(chunk) = self.chunks.get_mut(&(x, z)) {
                chunk.generated = true;
            }
        }
        for (x, z) in mesh_target {
            if let Some(chunk) = self.chunks.get_mut(&(x, z)) {
                chunk.meshed = true;
            }
        }

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render"),
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

            rpass.set_pipeline(self.sky_pipeline.as_ref().unwrap());
            rpass.set_bind_group(0, self.bind_group_global.as_ref().unwrap(), &[]);
            rpass.draw(0..3, 0..1);

            rpass.set_pipeline(self.render_pipeline.as_ref().unwrap());
            rpass.set_bind_group(0, self.bind_group_global.as_ref().unwrap(), &[]);

            let mut render_list: Vec<&Chunk> = self.chunks.values().filter(|c| c.meshed).collect();
            render_list.sort_by_key(|c| c.dist_sq);

            for chunk in render_list {
                rpass.set_bind_group(1, &chunk.bind_group_chunk_data, &[]);
                rpass.set_vertex_buffer(0, chunk.vertex_buffer.slice(..));
                rpass.set_index_buffer(chunk.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                rpass.draw_indexed_indirect(&chunk.indirect_buffer, 0);
            }
        }

        queue.submit(Some(encoder.finish()));
        output.present();

        Ok(())
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
                self.camera.yaw += delta.0 as f32 * sensitivity;
                self.camera.pitch -= delta.1 as f32 * sensitivity;
                self.camera.pitch = self.camera.pitch.clamp(-1.55, 1.55);
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(_) => {
                self.configure_surface();
            }
            WindowEvent::RedrawRequested => {
                self.update();
                match self.render() {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost) | Err(wgpu::SurfaceError::Outdated) => {
                        self.configure_surface()
                    }
                    Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                    Err(e) => eprintln!("{:?}", e),
                }
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
