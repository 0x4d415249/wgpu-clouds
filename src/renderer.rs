use crate::data::GameRegistry;
use crate::player::Player;
use crate::shader_gen;
use crate::texture::TextureAtlas;
use crate::world::WorldManager;
use cgmath::{InnerSpace, SquareMatrix};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;
use winit::window::Window;

const FULLSCREEN_SHADER: &str = r#"
struct Out { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }
@group(0) @binding(0) var t_scene: texture_2d<f32>;
@group(0) @binding(1) var s_scene: sampler;

@vertex fn vs_main(@builtin(vertex_index) idx: u32) -> Out {
    var out: Out;
    var uvs = array<vec2<f32>,3>(vec2<f32>(0., 2.), vec2<f32>(0., 0.), vec2<f32>(2., 0.));
    out.uv = uvs[idx];
    out.pos = vec4<f32>(out.uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv.y = 1.0 - out.uv.y;
    return out;
}

@fragment fn fs_main(in: Out) -> @location(0) vec4<f32> {
    let col = textureSample(t_scene, s_scene, in.uv);
    // Simple tone mapping + sharpening
    let mapped = col.rgb / (col.rgb + vec3<f32>(1.0));
    let sharp = mapped * 1.2 - 0.1;
    return vec4<f32>(sharp, 1.0);
}
"#;

pub struct BindLayouts {
    pub camera: wgpu::BindGroupLayout,
    pub texture: wgpu::BindGroupLayout,
    pub depth: wgpu::BindGroupLayout,
    pub gen_layout: wgpu::BindGroupLayout,
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

    depth_tex: wgpu::Texture,
    depth_view: wgpu::TextureView,

    scene_tex: wgpu::Texture,
    scene_view: wgpu::TextureView,
    scene_bind_group: wgpu::BindGroup,

    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    texture_bind_group: wgpu::BindGroup,
    depth_bind_group: wgpu::BindGroup,

    pub render_scale: f32,
}

impl Renderer {
    pub async fn new(window: Arc<Window>, registry: &GameRegistry, atlas: &TextureAtlas) -> Self {
        println!("[Renderer] Initializing...");
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

        println!("[Renderer] Adapter: {:?}", adapter.get_info());

        // Note: Assuming wgpu version where request_device takes 1 argument based on previous errors
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

        // Dynamic Present Mode Selection
        let present_mode = caps
            .present_modes
            .iter()
            .cloned()
            .find(|&mode| mode == wgpu::PresentMode::Mailbox)
            .or_else(|| {
                caps.present_modes
                    .iter()
                    .cloned()
                    .find(|&mode| mode == wgpu::PresentMode::Immediate)
            })
            .unwrap_or(wgpu::PresentMode::Fifo);

        println!("[Renderer] Present Mode: {:?}", present_mode);

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

        let wgsl_source = shader_gen::generate_wgsl(registry);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Generated Shader"),
            source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
        });

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
            label: Some("Camera Layout"),
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
            label: Some("Texture Layout"),
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
            label: Some("Depth Read Layout"),
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
            label: Some("Gen Layout"),
        });

        let atlas_tex = crate::texture::create_atlas_texture(&device, &queue, atlas);
        let atlas_view = atlas_tex.create_view(&Default::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Uniform"),
            contents: &[0u8; 256],
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("Camera Group"),
        });

        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            label: Some("Texture Group"),
        });

        let render_scale = 0.75;
        let (sw, sh) = (
            (size.width as f32 * render_scale) as u32,
            (size.height as f32 * render_scale) as u32,
        );
        let sw = sw.max(1);
        let sh = sh.max(1);

        let scene_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Scene Tex"),
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
            view_formats: &[],
        });
        let scene_view = scene_tex.create_view(&Default::default());

        let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth"),
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

            view_formats: &[],
        });
        let depth_view = depth_tex.create_view(&Default::default());

        let depth_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &depth_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&depth_view),
            }],
            label: Some("Depth Group"),
        });

        // Pipelines
        let voxel_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&camera_layout, &texture_layout],
            push_constant_ranges: &[],
            label: Some("Voxel PL"),
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Voxel Pipeline"),
            layout: Some(&voxel_pl_layout),
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

        let sky_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&camera_layout, &texture_layout, &depth_layout],
            push_constant_ranges: &[],
            label: Some("Sky PL"),
        });

        let sky_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sky Pipeline"),
            layout: Some(&sky_pl_layout),
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

        // Upscale
        let upscale_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Upscale Shader"),
            source: wgpu::ShaderSource::Wgsl(FULLSCREEN_SHADER.into()),
        });

        let upscale_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            label: Some("Upscale Layout"),
        });

        let sampler_upscale = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &upscale_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler_upscale),
                },
            ],
            label: Some("Scene Group"),
        });

        let upscale_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&upscale_layout],
            push_constant_ranges: &[],
            label: None,
        });

        let upscale_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Upscale Pipeline"),
            layout: Some(&upscale_pl_layout),
            vertex: wgpu::VertexState {
                module: &upscale_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &upscale_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
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
            },
            render_pipeline,
            sky_pipeline,
            upscale_pipeline,
            depth_tex,
            depth_view,
            scene_tex,
            scene_view,
            scene_bind_group,
            camera_buffer,
            camera_bind_group,
            texture_bind_group,
            depth_bind_group,
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
            label: Some("Scene Tex"),
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
            view_formats: &[],
        });
        self.scene_view = self.scene_tex.create_view(&Default::default());

        self.depth_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth"),
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

            view_formats: &[],
        });
        self.depth_view = self.depth_tex.create_view(&Default::default());

        // Rebind Depth Group
        self.depth_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.bind_layouts.depth,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&self.depth_view),
            }],
            label: Some("Depth Group"),
        });

        // Rebind Scene Group
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let upscale_layout = self.upscale_pipeline.get_bind_group_layout(0);
        self.scene_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &upscale_layout,
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
            label: Some("Scene Group"),
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
        dt: f32,
    ) -> Result<(), wgpu::SurfaceError> {
        // println!("[R] Frame Start");
        let aspect = self.scene_tex.width() as f32 / self.scene_tex.height() as f32;
        let proj = cgmath::perspective(cgmath::Deg(70.0), aspect, 0.1, 4000.0);
        let view_pos = player.get_view_pos();
        let (sin_yaw, cos_yaw) = player.yaw.to_radians().sin_cos();
        let (sin_pitch, cos_pitch) = player.pitch.to_radians().sin_cos();
        let forward =
            cgmath::Vector3::new(cos_yaw * cos_pitch, sin_pitch, sin_yaw * cos_pitch).normalize();
        let view = cgmath::Matrix4::look_to_rh(view_pos, forward, cgmath::Vector3::unit_y());
        #[rustfmt::skip]
        let correction = cgmath::Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0);
        let view_proj = correction * proj * view;
        let inv_view_proj = view_proj
            .invert()
            .unwrap_or(cgmath::Matrix4::from_scale(1.0));

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct CameraUniform {
            view_proj: [[f32; 4]; 4],
            inv_view_proj: [[f32; 4]; 4],
            camera_pos: [f32; 3],
            time: f32,
            screen_size: [f32; 2],
            render_scale: f32,
            _pad: f32,
        }

        let uniform = CameraUniform {
            view_proj: view_proj.into(),
            inv_view_proj: inv_view_proj.into(),
            camera_pos: [view_pos.x, view_pos.y, view_pos.z],
            time: dt,
            screen_size: [
                self.scene_tex.width() as f32,
                self.scene_tex.height() as f32,
            ],
            render_scale: self.render_scale,
            _pad: 0.0,
        };

        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[uniform]));

        // println!("[R] Acquiring Texture...");
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&Default::default());

        let mut encoder = self.device.create_command_encoder(&Default::default());

        // PASS 1: VOXELS
        // println!("[R] Pass 1: Voxels");
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Voxel Pass"),
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

        // PASS 2: SKYBOX
        // println!("[R] Pass 2: Sky");
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Sky Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.scene_view,
                    resolve_target: None,
                    // LOAD existing voxel color
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None, // NO DEPTH ATTACHMENT
                ..Default::default()
            });

            pass.set_pipeline(&self.sky_pipeline);
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            pass.set_bind_group(1, &self.texture_bind_group, &[]);
            pass.set_bind_group(2, &self.depth_bind_group, &[]); // Bind Depth as Resource
            pass.draw(0..3, 0..1);
        }

        // PASS 3: UPSCALE
        // println!("[R] Pass 3: Upscale");
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Upscale Pass"),
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

        // println!("[R] Submit");
        self.queue.submit(Some(encoder.finish()));
        output.present();

        // println!("[R] Poll");
        let _ = self.device.poll(wgpu::PollType::Poll);

        Ok(())
    }
}
