use anyhow::Context;
use image::{DynamicImage, GenericImage, RgbaImage};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Rect {
    pub min: [f32; 2],
    pub max: [f32; 2],
}

pub struct TextureAtlas {
    pub image: RgbaImage,
    pub uv_map: HashMap<String, Rect>,
}

impl TextureAtlas {
    pub fn load_from_folder(path: &str) -> anyhow::Result<Self> {
        let mut images: Vec<(String, DynamicImage)> = Vec::new();

        if let Ok(dir) = std::fs::read_dir(path) {
            for entry in dir.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|ext| ext == "png")
                    && let Some(name) = path.file_stem().and_then(|s| s.to_str())
                    && let Ok(img) = image::open(&path)
                {
                    images.push((name.to_string(), img));
                }
            }
        }

        if images.is_empty() {
            return Ok(Self::create_fallback());
        }

        let tile_size = 16;
        let count = images.len() as u32;
        let atlas_width_tiles = (count as f32).sqrt().ceil() as u32;
        let atlas_height_tiles = count.div_ceil(atlas_width_tiles);

        let width = atlas_width_tiles * tile_size;
        let height = atlas_height_tiles * tile_size;

        let mut atlas_img = RgbaImage::new(width, height);
        let mut uv_map = HashMap::new();

        for (i, (name, img)) in images.into_iter().enumerate() {
            let tile_x = (i as u32 % atlas_width_tiles) * tile_size;
            let tile_y = (i as u32 / atlas_width_tiles) * tile_size;

            let resized =
                img.resize_exact(tile_size, tile_size, image::imageops::FilterType::Nearest);

            atlas_img
                .copy_from(&resized, tile_x, tile_y)
                .context("Failed to copy texture to atlas")?;

            let u_min = tile_x as f32 / width as f32;
            let v_min = tile_y as f32 / height as f32;
            let u_max = (tile_x + tile_size) as f32 / width as f32;
            let v_max = (tile_y + tile_size) as f32 / height as f32;

            uv_map.insert(
                name,
                Rect {
                    min: [u_min, v_min],
                    max: [u_max, v_max],
                },
            );
        }

        Ok(Self {
            image: atlas_img,
            uv_map,
        })
    }

    fn create_fallback() -> Self {
        let mut img = RgbaImage::new(16, 16);
        for x in 0..16 {
            for y in 0..16 {
                if (x < 8) ^ (y < 8) {
                    img.put_pixel(x, y, image::Rgba([0, 0, 0, 255]));
                } else {
                    img.put_pixel(x, y, image::Rgba([255, 0, 255, 255]));
                }
            }
        }
        let mut map = HashMap::new();
        map.insert(
            "default".to_string(),
            Rect {
                min: [0.0, 0.0],
                max: [1.0, 1.0],
            },
        );
        Self {
            image: img,
            uv_map: map,
        }
    }

    pub fn get_uv(&self, name: &str) -> Option<&Rect> {
        self.uv_map.get(name).or_else(|| self.uv_map.get("default"))
    }
}

/// Helper to upload the atlas to the GPU
pub fn create_atlas_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    atlas: &TextureAtlas,
) -> wgpu::Texture {
    let texture_size = wgpu::Extent3d {
        width: atlas.image.width(),
        height: atlas.image.height(),
        depth_or_array_layers: 1,
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        size: texture_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        label: Some("Atlas Texture"),
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
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
        texture_size,
    );

    texture
}
