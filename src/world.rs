use crate::chunk::{CHUNK_SIZE, CHUNK_VOL, Chunk};
use crate::data::GameRegistry;
use crate::mesher::{self, VoxelVertex};
use crate::texture::TextureAtlas;
use crossbeam_channel::{Receiver, Sender, unbounded};
use std::collections::{HashMap, HashSet, VecDeque};
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

struct GpuGenerator {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuGenerator {
    fn new(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        bind_group_layout: wgpu::BindGroupLayout,
    ) -> Self {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&bind_group_layout],
            ..Default::default()
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Terrain Gen"),
            layout: Some(&pipeline_layout),
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

// Buffer Pooling
struct GenBuffers {
    storage: wgpu::Buffer,
    output: wgpu::Buffer,
}

struct BufferPool {
    pool: VecDeque<GenBuffers>,
}

impl BufferPool {
    fn new() -> Self {
        Self {
            pool: VecDeque::new(),
        }
    }

    fn get(&mut self, device: &wgpu::Device) -> GenBuffers {
        if let Some(bufs) = self.pool.pop_front() {
            bufs
        } else {
            let size = (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE * 4) as u64;
            let storage = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Gen Storage Pool"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let output = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Gen Map Pool"),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            GenBuffers { storage, output }
        }
    }

    fn return_buffers(&mut self, bufs: GenBuffers) {
        self.pool.push_back(bufs);
    }
}

pub struct WorldManager {
    pub chunks: HashMap<[i32; 3], Chunk>,
    pub meshes: HashMap<[i32; 3], ChunkMesh>,

    mesh_tx: Sender<MeshResult>,
    mesh_rx: Receiver<MeshResult>,
    // GPU Rx: (TaskID, Coords, Data)
    gpu_tx: Sender<([u64; 3], [i32; 3], Vec<u32>)>,
    gpu_rx: Receiver<([u64; 3], [i32; 3], Vec<u32>)>,

    gpu_gen: GpuGenerator,
    // Stores (TaskID, Coords, Buffers)
    pending_gpu_tasks: Vec<(u64, [i32; 3], GenBuffers)>,
    buffer_pool: BufferPool,
    next_task_id: u64,

    registry: Arc<GameRegistry>,
    atlas: Arc<TextureAtlas>,

    render_distance: i32,
    loading_queue: HashSet<[i32; 3]>,
}

impl WorldManager {
    pub fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        shader: &wgpu::ShaderModule,
        layout: wgpu::BindGroupLayout,
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
            gpu_gen: GpuGenerator::new(device, shader, layout),
            pending_gpu_tasks: Vec::new(),
            buffer_pool: BufferPool::new(),
            next_task_id: 0,
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
            
            // Collect neighbors
            let mut neighbors = HashMap::new();
            for x in -1..=1 {
                for y in -1..=1 {
                    for z in -1..=1 {
                        if x == 0 && y == 0 && z == 0 { continue; }
                        if let Some(neighbor) = self.chunks.get(&[cx + x, cy + y, cz + z]) {
                            neighbors.insert([x, y, z], neighbor.clone());
                        }
                    }
                }
            }
            
            let tx = self.mesh_tx.clone();
            let reg = self.registry.clone();
            let atl = self.atlas.clone();
            let pos = [cx, cy, cz];
            rayon::spawn(move || {
                let (v, i) = mesher::generate_mesh(&chunk_clone, &neighbors, &reg, &atl);
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

        // Stop generating if buffers full
        if self.pending_gpu_tasks.len() > 16 {
            self.poll_gpu_tasks();
            return;
        }

        let mut candidates = Vec::new();
        let height_range = 4;

        for x in -self.render_distance..=self.render_distance {
            for z in -self.render_distance..=self.render_distance {
                for y in -height_range..=height_range {
                    let pos = [cx + x, cy + y, cz + z];
                    if pos[1] < -2 { continue; } // Don't generate below bedrock
                    if !self.chunks.contains_key(&pos) && !self.loading_queue.contains(&pos) {
                        let dist = x * x + y * y + z * z;
                        candidates.push((dist, pos));
                    }
                }
            }
        }
        candidates.sort_unstable_by_key(|k| k.0);

        let max_requests = 2;
        let mut dispatched = 0;
        for (_, pos) in candidates {
            if dispatched >= max_requests {
                break;
            }
            self.dispatch_gen(device, queue, pos);
            self.loading_queue.insert(pos);
            dispatched += 1;
        }

        self.poll_gpu_tasks();
        self.poll_mesh_tasks(device);
        self.unload_chunks(cx, cy, cz, height_range);
    }

    fn poll_gpu_tasks(&mut self) {
        // Fix RX signature to match
        while let Ok((_id_arr, pos, data)) = self.gpu_rx.try_recv() {
            let id = _id_arr[0]; // Use dummy array wrapper for simple generic tuple matching if needed, here just ID

            // Match task by ID
            if let Some(index) = self
                .pending_gpu_tasks
                .iter()
                .position(|(tid, _, _)| *tid == id)
            {
                let (_, _, buffers) = self.pending_gpu_tasks.swap_remove(index);

                // CRITICAL FIX: Unmap exactly here, once
                buffers.output.unmap();
                self.buffer_pool.return_buffers(buffers);

                let blocks: Vec<u16> = data.iter().map(|&x| x as u16).collect();
                let chunk = Chunk {
                    position: pos,
                    blocks,
                };
                self.chunks.insert(pos, chunk.clone());

                // Collect neighbors
                // Note: We need to access self.chunks, but we are inside a method that borrows self.
                // We can't easily clone neighbors here without cloning the whole map or doing it before.
                // But we just inserted the new chunk.
                // Let's try to collect neighbors.
                let mut neighbors = HashMap::new();
                let cx = pos[0];
                let cy = pos[1];
                let cz = pos[2];
                
                for x in -1..=1 {
                    for y in -1..=1 {
                        for z in -1..=1 {
                            if x == 0 && y == 0 && z == 0 { continue; }
                            if let Some(neighbor) = self.chunks.get(&[cx + x, cy + y, cz + z]) {
                                neighbors.insert([x, y, z], neighbor.clone());
                            }
                        }
                    }
                }

                let tx = self.mesh_tx.clone();
                let reg = self.registry.clone();
                let atl = self.atlas.clone();
                rayon::spawn(move || {
                    let (v, i) = mesher::generate_mesh(&chunk, &neighbors, &reg, &atl);
                    let _ = tx.send(MeshResult::Meshed {
                        coords: pos,
                        vertices: v,
                        indices: i,
                    });
                });
            }
        }
    }

    fn poll_mesh_tasks(&mut self, device: &wgpu::Device) {
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
    }

    fn unload_chunks(&mut self, cx: i32, cy: i32, cz: i32, height_range: i32) {
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
        let buffers = self.buffer_pool.get(device);
        let size = (CHUNK_VOL * 4) as u64;

        let task_id = self.next_task_id;
        self.next_task_id += 1;

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
            label: Some("Gen Param"),
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.gpu_gen.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: param_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.storage.as_entire_binding(),
                },
            ],
            label: Some("Gen Bind Group"),
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.gpu_gen.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // Dispatch: 64 / 4 = 16 workgroups per axis
            pass.dispatch_workgroups(16, 16, 16);
        }

        encoder.copy_buffer_to_buffer(&buffers.storage, 0, &buffers.output, 0, size);
        queue.submit(Some(encoder.finish()));

        // ... (mapping logic remains same) ...
        let slice = buffers.output.slice(..);
        let tx = self.gpu_tx.clone();
        let coords = pos;
        let map_buf_cloned = buffers.output.clone();

        slice.map_async(wgpu::MapMode::Read, move |res| {
            if res.is_ok() {
                let slice = map_buf_cloned.slice(..);
                let data = slice.get_mapped_range();
                let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
                drop(data);
                let _ = tx.send(([task_id, 0, 0], coords, result));
            }
        });

        self.pending_gpu_tasks.push((task_id, pos, buffers));
    }
}
