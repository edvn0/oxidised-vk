use std::borrow::Cow;

use crate::{bloom_pass::BloomSettings, render_passes::recordings::CompositeSettings};
use dear_imgui_rs::{self as imgui, TreeNodeFlags};

pub trait Settings {
    fn ui(&mut self, ui: &imgui::Ui);
}

impl Settings for BloomSettings {
    fn ui(&mut self, ui: &imgui::Ui) {
        use crate::bloom_pass::bloom_limits;

        ui.checkbox("Enabled", &mut self.enabled);
        ui.slider(
            "Threshold",
            bloom_limits::THRESHOLD_MIN,
            bloom_limits::THRESHOLD_MAX,
            &mut self.threshold,
        );
        ui.slider(
            "Knee",
            bloom_limits::KNEE_MIN,
            bloom_limits::KNEE_MAX,
            &mut self.knee,
        );
        ui.slider(
            "Intensity",
            bloom_limits::INTENSITY_MIN,
            bloom_limits::INTENSITY_MAX,
            &mut self.intensity,
        );
        ui.slider(
            "Filter Radius",
            bloom_limits::FILTER_RADIUS_MIN,
            bloom_limits::FILTER_RADIUS_MAX,
            &mut self.filter_radius,
        );

        if ui.button("Reset") {
            *self = Self::default();
        }
    }
}

mod composite_limits {
    pub const EXPOSURE_MIN: f32 = 0.01;
    pub const EXPOSURE_MAX: f32 = 5.0;

    pub const GAMMA_MIN: f32 = 1.0;
    pub const GAMMA_MAX: f32 = 3.0;

    pub const BLOOM_STRENGTH_MIN: f32 = 0.0;
    pub const BLOOM_STRENGTH_MAX: f32 = 2.5;

    pub const SATURATION_MIN: f32 = 0.0;
    pub const SATURATION_MAX: f32 = 2.0;

    pub const CONTRAST_MIN: f32 = 0.5;
    pub const CONTRAST_MAX: f32 = 2.0;

    pub const BRIGHTNESS_MIN: f32 = -1.0;
    pub const BRIGHTNESS_MAX: f32 = 1.0;

    pub const VIGNETTE_STRENGTH_MIN: f32 = 0.0;
    pub const VIGNETTE_STRENGTH_MAX: f32 = 1.0;

    pub const VIGNETTE_RADIUS_MIN: f32 = 0.0;
    pub const VIGNETTE_RADIUS_MAX: f32 = 1.0;

    pub const GRAIN_STRENGTH_MIN: f32 = 0.0;
    pub const GRAIN_STRENGTH_MAX: f32 = 0.1;
}

#[derive(Copy, Clone)]
enum TonemapOperator {
    Reinhard,
    Aces,
}

impl CompositeSettings {
    const TONEMAP_OPERATORS: [TonemapOperator; 2] =
        [TonemapOperator::Reinhard, TonemapOperator::Aces];
    fn tonemap_label(op: &TonemapOperator) -> Cow<'_, str> {
        match op {
            TonemapOperator::Reinhard => "Reinhard".into(),
            TonemapOperator::Aces => "ACES".into(),
        }
    }
}

impl Settings for CompositeSettings {
    fn ui(&mut self, ui: &imgui::Ui) {
        // ------------------------------------------------------------
        // Tone mapping
        // ------------------------------------------------------------
        let id_token = ui.push_id("tone_mapping");
        if ui.collapsing_header("Tone Mapping", TreeNodeFlags::NONE) {
            ui.slider(
                "Exposure",
                composite_limits::EXPOSURE_MIN,
                composite_limits::EXPOSURE_MAX,
                &mut self.exposure,
            );

            ui.slider(
                "Gamma",
                composite_limits::GAMMA_MIN,
                composite_limits::GAMMA_MAX,
                &mut self.gamma,
            );

            let _combo_token = ui.push_id("combo_tonemap");
            let mut current = self.tonemap_operator as usize;
            if ui.combo(
                "Tone Mapping",
                &mut current,
                &Self::TONEMAP_OPERATORS,
                Self::tonemap_label,
            ) {
                self.tonemap_operator = current as i32;
            }
        }
        id_token.end();

        // ------------------------------------------------------------
        // Bloom
        // ------------------------------------------------------------
        let id_token = ui.push_id("bloom");
        if ui.collapsing_header("Bloom", TreeNodeFlags::NONE) {
            ui.slider(
                "Strength",
                composite_limits::BLOOM_STRENGTH_MIN,
                composite_limits::BLOOM_STRENGTH_MAX,
                &mut self.bloom_strength,
            );
        }
        id_token.end();

        // ------------------------------------------------------------
        // Color grading
        // ------------------------------------------------------------
        if ui.collapsing_header("Color Grading", TreeNodeFlags::NONE) {
            ui.slider(
                "Saturation",
                composite_limits::SATURATION_MIN,
                composite_limits::SATURATION_MAX,
                &mut self.saturation,
            );

            ui.slider(
                "Contrast",
                composite_limits::CONTRAST_MIN,
                composite_limits::CONTRAST_MAX,
                &mut self.contrast,
            );

            ui.slider(
                "Brightness",
                composite_limits::BRIGHTNESS_MIN,
                composite_limits::BRIGHTNESS_MAX,
                &mut self.brightness,
            );
        }

        // ------------------------------------------------------------
        // Vignette
        // ------------------------------------------------------------
        let id_token = ui.push_id("vignette");
        if ui.collapsing_header("Vignette", TreeNodeFlags::NONE) {
            ui.slider(
                "Strength",
                composite_limits::VIGNETTE_STRENGTH_MIN,
                composite_limits::VIGNETTE_STRENGTH_MAX,
                &mut self.vignette_strength,
            );

            ui.slider(
                "Radius",
                composite_limits::VIGNETTE_RADIUS_MIN,
                composite_limits::VIGNETTE_RADIUS_MAX,
                &mut self.vignette_radius,
            );
        }
        id_token.end();

        // ------------------------------------------------------------
        // Film grain
        // ------------------------------------------------------------
        let id_token = ui.push_id("film_grain");
        if ui.collapsing_header("Film Grain", TreeNodeFlags::NONE) {
            ui.slider(
                "Strength",
                composite_limits::GRAIN_STRENGTH_MIN,
                composite_limits::GRAIN_STRENGTH_MAX,
                &mut self.grain_strength,
            );
        }
        id_token.end();

        ui.separator();

        // ------------------------------------------------------------
        // Reset
        // ------------------------------------------------------------
        if ui.button("Reset Composite Settings") {
            *self = Self::default();
        }
    }
}
