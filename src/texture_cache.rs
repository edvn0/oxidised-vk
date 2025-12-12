use std::collections::HashMap;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BindlessTextureIndex(pub u32);

impl BindlessTextureIndex {
    pub const WHITE: Self = Self(0);

    #[inline(always)]
    pub fn raw(self) -> u32 {
        self.0
    }
}

pub struct TextureCache {
    map: HashMap<usize, BindlessTextureIndex>,
    fallback: BindlessTextureIndex,
}

impl TextureCache {
    pub fn new(fallback: BindlessTextureIndex) -> Self {
        Self {
            map: HashMap::new(),
            fallback,
        }
    }

    pub fn insert(&mut self, source_index: usize, texture_id: BindlessTextureIndex) {
        self.map.insert(source_index, texture_id);
    }

    pub fn get(&self, source_index: usize) -> BindlessTextureIndex {
        self.map
            .get(&source_index)
            .copied()
            .unwrap_or(self.fallback)
    }
}

impl TextureCache {
    #[inline(always)]
    pub fn from_gltf_texture(&self, tex: &gltf::texture::Texture) -> BindlessTextureIndex {
        self.get(tex.source().index())
    }

    #[inline(always)]
    pub fn from_opt_info(&self, tex: Option<gltf::texture::Info>) -> BindlessTextureIndex {
        tex.map(|t| self.from_gltf_texture(&t.texture()))
            .unwrap_or(self.fallback)
    }

    #[inline(always)]
    pub fn from_normal_texture(
        &self,
        tex: Option<gltf::material::NormalTexture>,
    ) -> BindlessTextureIndex {
        tex.map(|t| self.from_gltf_texture(&t.texture()))
            .unwrap_or(self.fallback)
    }

    #[inline(always)]
    pub fn from_ao_texture(
        &self,
        tex: Option<gltf::material::OcclusionTexture>,
    ) -> BindlessTextureIndex {
        tex.map(|t| self.from_gltf_texture(&t.texture()))
            .unwrap_or(self.fallback)
    }
}
