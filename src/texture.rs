//! Handles creating a texture atlas at runtime from a folder of images.

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
    /// Loads all .png files from the specific directory and stitches them into an atlas.
    pub fn load_from_folder(path: &str) -> anyhow::Result<Self> {
        let mut images: Vec<(String, DynamicImage)> = Vec::new();

        // Handle path checking gracefully
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

        // Assume 16x16 textures for simplicity
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

            // GenericImage::copy_from in image 0.25 takes &DynamicImage usually, or &impl GenericImage
            atlas_img
                .copy_from(&resized, tile_x, tile_y)
                .context("Failed to copy texture to atlas")?;

            // Calculate UVs
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
