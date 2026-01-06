use crate::{bloom_pass::BloomSettings, render_passes::recordings::CompositeSettings};
use dear_imgui_rs as imgui;

impl BloomSettings {
    pub fn ui(&mut self, ui: &imgui::Ui) {
        use crate::bloom_pass::bloom_limits;

        ui.checkbox("Enabled", &mut self.enabled);
        ui.slider(
            "Threshold",
            bloom_limits::THRESHOLD_MIN,
            bloom_limits::THRESHOLD_MAX,
            &mut self.threshold,
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

impl CompositeSettings {
    pub fn ui(&mut self, ui: &imgui::Ui) {
        ui.slider("Exposure", 0.01, 5.0, &mut self.exposure);
        ui.slider("Bloom strength", 0.0, 2.5, &mut self.bloom_strength);

        if ui.button("Reset") {
            self.exposure = 1.0;
            self.bloom_strength = 1.0;
        }
    }
}
