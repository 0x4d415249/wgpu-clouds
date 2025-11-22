use crate::atmosphere::AtmosphereState;
use crate::data::GameRegistry;
use crate::player::Player;
use crate::texture::TextureAtlas;
use crate::world::WorldManager;
use crate::{shader, shader_gen};
use cgmath::{InnerSpace, SquareMatrix};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;
use winit::window::Window;

pub struct BindLayouts {
    pub camera: wgpu::BindGroupLayout,
    pub texture: wgpu::BindGroupLayout,
    pub depth: wgpu::BindGroupLayout,
    pub gen_layout: wgpu::BindGroupLayout,
    pub particle: wgpu::BindGroupLayout,
}

pub struct Renderer {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,

    pub shader_module: wgpu::ShaderModule,
    pub bind_layouts: BindLayouts,

    render_pipeline: wgpu::RenderPipeline,
    sky_pipeline: wgpu::RenderPipeline,
    upscale_pipeline: wgpu::RenderPipeline,
    rain_render: wgpu::RenderPipeline,
    rain_compute: wgpu::ComputePipeline,

    depth_tex: wgpu::Texture,
    depth_view: wgpu::TextureView,

    scene_tex: wgpu::Texture,
    scene_view: wgpu::TextureView,
    scene_bind_group: wgpu::BindGroup,

    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    texture_bind_group: wgpu::BindGroup,
    depth_bind_group: wgpu::BindGroup,
    particle_bind_group: wgpu::BindGroup,

    pub render_scale: f32,
}

impl Renderer {
    pub async fn new(window: Arc<Window>, registry: &GameRegistry, atlas: &TextureAtlas) -> Self {
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::POLYGON_MODE_LINE
                    | wgpu::Features::VERTEX_WRITABLE_STORAGE,
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await
            .unwrap();

        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let present_mode = caps
            .present_modes
            .iter()
            .cloned()
            .find(|&m| m == wgpu::PresentMode::Mailbox)
            .unwrap_or(wgpu::PresentMode::Fifo);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: caps.formats[0],
            width: size.width,
            height: size.height,
            present_mode,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let wgsl = shader_gen::generate_wgsl(registry);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });

        // --- LAYOUTS ---
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::all(),
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: None,
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
            label: None,
        });
        let depth_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            label: None,
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
            label: None,
        });
        let gen_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
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
            label: None,
        });

        // --- RESOURCES ---
        let atlas_tex = crate::texture::create_atlas_texture(&device, &queue, atlas);
        let atlas_view = atlas_tex.create_view(&Default::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera"),
            contents: &[0u8; 256],
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let mut rng_seed = 12345;
        fn random_f32(seed: &mut u32) -> f32 {
            *seed = (*seed).wrapping_mul(1664525).wrapping_add(1013904223);
            (*seed as f32) / (u32::MAX as f32)
        }
        #[repr(C)]
        #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        struct ParticleInit {
            pos: [f32; 4],
            vel: [f32; 4],
        }
        let particles: Vec<ParticleInit> = (0..20000)
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
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: None,
        });
        let texture_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: None,
        });
        let particle_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &particle_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: particle_buffer.as_entire_binding(),
            }],
            label: None,
        });

        // --- PIPELINES ---
        let voxel_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&camera_layout, &texture_layout],
            push_constant_ranges: &[],
            label: None,
        });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Voxel"),
            layout: Some(&voxel_pll),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[crate::mesher::VoxelVertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });

        let sky_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&camera_layout, &texture_layout, &depth_layout],
            push_constant_ranges: &[],
            label: None,
        });
        let sky_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sky"),
            layout: Some(&sky_pll),
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
                    format: config.format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            depth_stencil: None,
            multisample: Default::default(),
            primitive: Default::default(),
            multiview: None,
            cache: None,
        });

        let rain_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&camera_layout, &texture_layout, &particle_layout],
            push_constant_ranges: &[],
            label: None,
        });
        let rain_compute = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Rain Sim"),
            layout: Some(&rain_pll),
            module: &shader,
            entry_point: Some("cs_rain"),
            compilation_options: Default::default(),
            cache: None,
        });
        let rain_render = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Rain Render"),
            layout: Some(&rain_pll),
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
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });

        let upscale_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(shader::POST.into()),
        });
        let upscale_l = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            label: None,
        });
        let upscale_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&upscale_l],
            push_constant_ranges: &[],
            label: None,
        });
        let upscale_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Upscale"),
            layout: Some(&upscale_pll),
            vertex: wgpu::VertexState {
                module: &upscale_mod,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &upscale_mod,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            depth_stencil: None,
            multisample: Default::default(),
            primitive: Default::default(),
            multiview: None,
            cache: None,
        });

        // Targets
        let render_scale = 0.75;
        let (sw, sh) = (
            (size.width as f32 * render_scale) as u32,
            (size.height as f32 * render_scale) as u32,
        );
        let scene_tex = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: sw,
                height: sh,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            label: None,
            view_formats: &[],
        });
        let scene_view = scene_tex.create_view(&Default::default());
        let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: sw,
                height: sh,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            label: None,
            view_formats: &[],
        });
        let depth_view = depth_tex.create_view(&Default::default());

        let depth_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &depth_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&depth_view),
            }],
            label: None,
        });
        let scene_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &upscale_l,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: None,
        });

        Self {
            device,
            queue,
            surface,
            config,
            shader_module: shader,
            bind_layouts: BindLayouts {
                camera: camera_layout,
                texture: texture_layout,
                depth: depth_layout,
                gen_layout,
                particle: particle_layout,
            },
            render_pipeline,
            sky_pipeline,
            upscale_pipeline,
            rain_render,
            rain_compute,
            depth_tex,
            depth_view,
            scene_tex,
            scene_view,
            scene_bind_group: scene_bg,
            camera_buffer,
            camera_bind_group: camera_bg,
            texture_bind_group: texture_bg,
            depth_bind_group: depth_bg,
            particle_bind_group: particle_bg,
            render_scale,
        }
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.recreate_offscreen_targets();
        }
    }

    fn recreate_offscreen_targets(&mut self) {
        let (sw, sh) = (
            (self.config.width as f32 * self.render_scale) as u32,
            (self.config.height as f32 * self.render_scale) as u32,
        );
        let sw = sw.max(1);
        let sh = sh.max(1);
        self.scene_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: sw,
                height: sh,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            label: None,
            view_formats: &[],
        });
        self.scene_view = self.scene_tex.create_view(&Default::default());
        self.depth_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: sw,
                height: sh,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            label: None,
            view_formats: &[],
        });
        self.depth_view = self.depth_tex.create_view(&Default::default());

        self.depth_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.bind_layouts.depth,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&self.depth_view),
            }],
            label: None,
        });

        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let upscale_l = self.upscale_pipeline.get_bind_group_layout(0);
        self.scene_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &upscale_l,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: None,
        });
    }

    pub fn cycle_render_scale(&mut self) {
        self.render_scale = if self.render_scale >= 1.0 {
            0.5
        } else {
            self.render_scale + 0.25
        };
        println!("Render Scale: {:.2}", self.render_scale);
        self.recreate_offscreen_targets();
    }

    pub fn set_wireframe(&mut self, _wire: bool) {}

    pub fn render(
        &mut self,
        world: &WorldManager,
        player: &Player,
        atmosphere: &AtmosphereState,
    ) -> Result<(), wgpu::SurfaceError> {
        let aspect = self.scene_tex.width() as f32 / self.scene_tex.height() as f32;
        let proj = cgmath::perspective(cgmath::Deg(70.0), aspect, 0.1, 4000.0);
        let (sin_y, cos_y) = player.yaw.to_radians().sin_cos();
        let (sin_p, cos_p) = player.pitch.to_radians().sin_cos();
        let forward = cgmath::Vector3::new(cos_y * cos_p, sin_p, sin_y * cos_p).normalize();
        let view =
            cgmath::Matrix4::look_to_rh(player.get_view_pos(), forward, cgmath::Vector3::unit_y());
        #[rustfmt::skip]
        let correction = cgmath::Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0);
        let view_proj = correction * proj * view;
        let inv = view_proj
            .invert()
            .unwrap_or(cgmath::Matrix4::from_scale(1.0));

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
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

        let uniform = CameraUniform {
            view_proj: view_proj.into(),
            inv_view_proj: inv.into(),
            camera_pos: [player.position.x, player.position.y, player.position.z],
            time: atmosphere.sim_time,
            day_progress: atmosphere.day_time,
            weather_offset: atmosphere.weather_val,
            cloud_type: atmosphere.weather_val,
            lightning_intensity: atmosphere.lightning_active,
            lightning_pos: atmosphere.lightning_pos,
            _pad1: 0.0,
            lightning_color: [0.9, 0.9, 1.0],
            _pad2: 0.0,
            wind: atmosphere.wind_vec,
            rain: atmosphere.rain_intensity,
            _pad3: 0.0,
        };
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[uniform]));

        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&Default::default());
        let mut encoder = self.device.create_command_encoder(&Default::default());

        // 0. Compute Rain
        if atmosphere.rain_intensity > 0.01 {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.rain_compute);
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            pass.set_bind_group(1, &self.texture_bind_group, &[]); // Just to satisfy layout index
            pass.set_bind_group(2, &self.particle_bind_group, &[]);
            let count = 20000u32.div_ceil(64);
            pass.dispatch_workgroups(count, 1, 1);
        }

        // 1. Voxels
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Voxel"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.scene_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&self.render_pipeline);
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            pass.set_bind_group(1, &self.texture_bind_group, &[]);
            for mesh in world.meshes.values() {
                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }
        }

        // 2. Sky
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Sky"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.scene_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(&self.sky_pipeline);
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            pass.set_bind_group(1, &self.texture_bind_group, &[]);
            pass.set_bind_group(2, &self.depth_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // 3. Rain Render
        if atmosphere.rain_intensity > 0.01 {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Rain"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.scene_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&self.rain_render);
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            pass.set_bind_group(1, &self.texture_bind_group, &[]);
            pass.set_bind_group(2, &self.particle_bind_group, &[]);
            pass.draw(0..4, 0..20000);
        }

        // 4. Upscale
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Upscale"),
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
                ..Default::default()
            });
            pass.set_pipeline(&self.upscale_pipeline);
            pass.set_bind_group(0, &self.scene_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
        output.present();

        let _ = self.device.poll(wgpu::PollType::Poll);
        Ok(())
    }
}
