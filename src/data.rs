//! Data structures for Block and Biome definitions.
//! Formerly definitions.rs.

use serde::Deserialize;
use std::collections::HashMap;

pub type BlockId = u8;

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BlockGeometry {
    Cube,
    Cross,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BlockTextureInfo {
    Uniform {
        texture: String,
    },
    TopBottomSide {
        top: String,
        bottom: String,
        side: String,
    },
    Pillar {
        end: String,
        side: String,
    },
}

#[derive(Debug, Clone, Deserialize)]
pub struct BlockDefinition {
    pub numeric_id: BlockId,
    pub geometry: BlockGeometry,
    pub textures: Option<BlockTextureInfo>,
    pub is_transparent: bool,
    pub is_empty: bool,
    #[serde(default)]
    pub needs_biome_tint: bool,
}

impl BlockDefinition {
    pub fn get_texture_for_face(&self, axis: usize, dir: i32) -> Option<&str> {
        if self.is_empty {
            return None;
        }
        match &self.textures {
            Some(BlockTextureInfo::Uniform { texture }) => Some(texture),
            Some(BlockTextureInfo::TopBottomSide { top, bottom, side }) => match (axis, dir) {
                (1, 1) => Some(top),
                (1, -1) => Some(bottom),
                _ => Some(side),
            },
            Some(BlockTextureInfo::Pillar { end, side }) => match (axis, dir) {
                (1, 1) | (1, -1) => Some(end),
                _ => Some(side),
            },
            None => None,
        }
    }

    pub fn needs_biome_tint_for_face(&self, axis: usize, dir: i32) -> bool {
        if !self.needs_biome_tint {
            return false;
        }
        match &self.textures {
            Some(BlockTextureInfo::TopBottomSide { top, .. }) => match (axis, dir) {
                (1, 1) => top.contains("grass_block_top"),
                (1, -1) => false,
                _ => false, // Simplified: only tint top of grass for now
            },
            Some(BlockTextureInfo::Uniform { .. }) => true, // Leaves, etc
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct BiomeSelector {
    pub weirdness: (Option<f64>, Option<f64>),
    pub temperature: (Option<f64>, Option<f64>),
    pub humidity: (Option<f64>, Option<f64>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct BiomeTerrain {
    pub base_height: i32,
    pub amplitude: i32,
    pub top_block: String,
    pub middle_block: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BiomeDefinition {
    pub selector: BiomeSelector,
    pub terrain: BiomeTerrain,
}

/// Central registry for game data.
#[derive(Default, Clone)]
pub struct GameRegistry {
    pub blocks: HashMap<String, BlockDefinition>,
    pub biomes: HashMap<String, BiomeDefinition>,
    // Fast lookup
    pub blocks_by_id: Vec<Option<BlockDefinition>>,
    pub block_name_to_id: HashMap<String, BlockId>,
}

impl GameRegistry {
    pub fn new_from_json(blocks_json: &str, biomes_json: &str) -> anyhow::Result<Self> {
        let blocks: HashMap<String, BlockDefinition> = serde_json::from_str(blocks_json)?;
        let biomes: HashMap<String, BiomeDefinition> = serde_json::from_str(biomes_json)?;

        let mut registry = Self {
            blocks: blocks.clone(),
            biomes,
            blocks_by_id: vec![],
            block_name_to_id: HashMap::new(),
        };

        // Build lookup tables
        let max_id = blocks.values().map(|b| b.numeric_id).max().unwrap_or(0);
        registry.blocks_by_id.resize(max_id as usize + 1, None);

        for (name, def) in &blocks {
            registry
                .block_name_to_id
                .insert(name.clone(), def.numeric_id);
            registry.blocks_by_id[def.numeric_id as usize] = Some(def.clone());
        }

        Ok(registry)
    }

    pub fn get_block_id(&self, name: &str) -> Option<BlockId> {
        self.block_name_to_id.get(name).copied()
    }

    pub fn get_block_def(&self, id: BlockId) -> Option<&BlockDefinition> {
        self.blocks_by_id
            .get(id as usize)
            .and_then(|opt| opt.as_ref())
    }
}
