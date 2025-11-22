use crate::chunk::{CHUNK_SIZE, Chunk};
use crate::data::GameRegistry;
use crate::mesher::{self, VoxelVertex};
use crate::texture::TextureAtlas;
use crossbeam_channel::{Receiver, Sender, unbounded};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use wgpu::util::DeviceExt;

enum MeshResult {
    Meshed {
        coords: [i32; 3],
        vertices: Vec<VoxelVertex>,
        indices: Vec<u32>,
    },
}

pub struct ChunkMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
}

// --- GPU GENERATOR ---
struct GpuGenerator {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuGenerator {
    fn new(device: &wgpu::Device, shader: &wgpu::ShaderModule) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Gen Layout"),
            entries: &[
                // Params
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Block Data
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

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&bind_group_layout],
            ..Default::default()
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Terrain Gen"),
            layout: Some(&layout),
            module: shader,
            entry_point: Some("cs_generate"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
        }
    }
}

// --- WORLD MANAGER ---
pub struct WorldManager {
    pub chunks: HashMap<[i32; 3], Chunk>,
    pub meshes: HashMap<[i32; 3], ChunkMesh>,

    // Channels
    mesh_tx: Sender<MeshResult>,
    mesh_rx: Receiver<MeshResult>,
    gpu_tx: Sender<([i32; 3], Vec<u32>)>,
    gpu_rx: Receiver<([i32; 3], Vec<u32>)>,

    // Async State
    gpu_gen: GpuGenerator,
    pending_gpu_tasks: Vec<([i32; 3], wgpu::Buffer, wgpu::Buffer)>, // Pos, MapBuf, StorageBuf

    registry: Arc<GameRegistry>,
    atlas: Arc<TextureAtlas>,

    render_distance: i32,
    loading_queue: HashSet<[i32; 3]>,
}

impl WorldManager {
    pub fn new(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        registry: GameRegistry,
        atlas: TextureAtlas,
        render_distance: i32,
    ) -> Self {
        let (mesh_tx, mesh_rx) = unbounded();
        let (gpu_tx, gpu_rx) = unbounded();

        Self {
            chunks: HashMap::new(),
            meshes: HashMap::new(),
            mesh_tx,
            mesh_rx,
            gpu_tx,
            gpu_rx,
            gpu_gen: GpuGenerator::new(device, shader),
            pending_gpu_tasks: Vec::new(),
            registry: Arc::new(registry),
            atlas: Arc::new(atlas),
            render_distance,
            loading_queue: HashSet::new(),
        }
    }

    pub fn get_block(&self, x: i32, y: i32, z: i32) -> u16 {
        let cx = x.div_euclid(CHUNK_SIZE as i32);
        let cy = y.div_euclid(CHUNK_SIZE as i32);
        let cz = z.div_euclid(CHUNK_SIZE as i32);
        let lx = x.rem_euclid(CHUNK_SIZE as i32) as u32;
        let ly = y.rem_euclid(CHUNK_SIZE as i32) as u32;
        let lz = z.rem_euclid(CHUNK_SIZE as i32) as u32;

        if let Some(chunk) = self.chunks.get(&[cx, cy, cz]) {
            chunk.get_block(lx, ly, lz)
        } else {
            0
        }
    }

    pub fn set_block(&mut self, x: i32, y: i32, z: i32, id: u16) {
        let cx = x.div_euclid(CHUNK_SIZE as i32);
        let cy = y.div_euclid(CHUNK_SIZE as i32);
        let cz = z.div_euclid(CHUNK_SIZE as i32);
        let lx = x.rem_euclid(CHUNK_SIZE as i32) as u32;
        let ly = y.rem_euclid(CHUNK_SIZE as i32) as u32;
        let lz = z.rem_euclid(CHUNK_SIZE as i32) as u32;

        if let Some(chunk) = self.chunks.get_mut(&[cx, cy, cz]) {
            chunk.set_block(lx, ly, lz, id);
            let chunk_clone = chunk.clone();
            let tx = self.mesh_tx.clone();
            let reg = self.registry.clone();
            let atl = self.atlas.clone();
            let pos = [cx, cy, cz];
            rayon::spawn(move || {
                let (v, i) = mesher::generate_mesh(&chunk_clone, &reg, &atl);
                let _ = tx.send(MeshResult::Meshed {
                    coords: pos,
                    vertices: v,
                    indices: i,
                });
            });
        }
    }

    pub fn update_chunks(
        &mut self,
        player_pos: [f32; 3],
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let cx = (player_pos[0] / CHUNK_SIZE as f32).floor() as i32;
        let cy = (player_pos[1] / CHUNK_SIZE as f32).floor() as i32;
        let cz = (player_pos[2] / CHUNK_SIZE as f32).floor() as i32;

        // 1. Identify and Sort Candidates
        let mut candidates = Vec::new();
        // Expanded vertical range for flying
        let height_range = 4;

        for x in -self.render_distance..=self.render_distance {
            for z in -self.render_distance..=self.render_distance {
                for y in -height_range..=height_range {
                    let pos = [cx + x, cy + y, cz + z];

                    // No hard height limits anymore, sky is the limit (literally)

                    if !self.chunks.contains_key(&pos) && !self.loading_queue.contains(&pos) {
                        let dx = x;
                        let dy = y * 2;
                        let dz = z;
                        let dist_sq = dx * dx + dy * dy + dz * dz;
                        candidates.push((dist_sq, pos));
                    }
                }
            }
        }

        // Sort: Closest first
        candidates.sort_unstable_by_key(|k| k.0);

        // 2. Dispatch Top N Requests
        let max_requests = 16;
        let mut dispatched = 0;
        for (_, pos) in candidates {
            if dispatched >= max_requests {
                break;
            }

            self.dispatch_gen(device, queue, pos);
            self.loading_queue.insert(pos);
            dispatched += 1;
        }

        // 3. Poll GPU Results
        while let Ok((pos, data)) = self.gpu_rx.try_recv() {
            if let Some(idx) = self
                .pending_gpu_tasks
                .iter()
                .position(|(p, _, _)| *p == pos)
            {
                self.pending_gpu_tasks.swap_remove(idx);
            }

            let blocks: Vec<u16> = data.iter().map(|&x| x as u16).collect();
            let chunk = Chunk {
                position: pos,
                blocks,
            };
            self.chunks.insert(pos, chunk.clone());

            let tx = self.mesh_tx.clone();
            let reg = self.registry.clone();
            let atl = self.atlas.clone();
            rayon::spawn(move || {
                let (v, i) = mesher::generate_mesh(&chunk, &reg, &atl);
                let _ = tx.send(MeshResult::Meshed {
                    coords: pos,
                    vertices: v,
                    indices: i,
                });
            });
        }

        // 4. Poll Mesh Results
        while let Ok(result) = self.mesh_rx.try_recv() {
            match result {
                MeshResult::Meshed {
                    coords,
                    vertices,
                    indices,
                } => {
                    self.loading_queue.remove(&coords);
                    if !vertices.is_empty() {
                        let vertex_buffer =
                            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("Chunk VB"),
                                contents: bytemuck::cast_slice(&vertices),
                                usage: wgpu::BufferUsages::VERTEX,
                            });
                        let index_buffer =
                            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("Chunk IB"),
                                contents: bytemuck::cast_slice(&indices),
                                usage: wgpu::BufferUsages::INDEX,
                            });
                        self.meshes.insert(
                            coords,
                            ChunkMesh {
                                vertex_buffer,
                                index_buffer,
                                index_count: indices.len() as u32,
                            },
                        );
                    } else {
                        self.meshes.remove(&coords);
                    }
                }
            }
        }

        // 5. Unload Far Chunks
        self.chunks.retain(|&[x, y, z], _| {
            (x - cx).abs() <= self.render_distance + 2
                && (z - cz).abs() <= self.render_distance + 2
                && (y - cy).abs() <= height_range + 2
        });
        self.meshes.retain(|&[x, y, z], _| {
            (x - cx).abs() <= self.render_distance + 2
                && (z - cz).abs() <= self.render_distance + 2
                && (y - cy).abs() <= height_range + 2
        });
    }

    fn dispatch_gen(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, pos: [i32; 3]) {
        let size = (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE * 4) as u64; // u32 array for shader
        let storage_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gen Storage"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let map_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gen Map"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct GenParams {
            chunk_pos: [i32; 3],
            seed: u32,
        }
        let params = GenParams {
            chunk_pos: pos,
            seed: 1337,
        };
        let param_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM,
            label: None,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.gpu_gen.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: param_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: storage_buf.as_entire_binding(),
                },
            ],
            label: None,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.gpu_gen.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(8, 8, 8);
        }
        encoder.copy_buffer_to_buffer(&storage_buf, 0, &map_buf, 0, size);
        queue.submit(Some(encoder.finish()));

        let slice = map_buf.slice(..);
        let tx = self.gpu_tx.clone();
        let coords = pos;
        let map_buf_cloned = map_buf.clone();

        slice.map_async(wgpu::MapMode::Read, move |res| {
            if res.is_ok() {
                let slice = map_buf_cloned.slice(..);
                let data = slice.get_mapped_range();
                let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
                drop(data);
                let _ = tx.send((coords, result));
            }
        });

        self.pending_gpu_tasks.push((pos, map_buf, storage_buf));
    }
}
