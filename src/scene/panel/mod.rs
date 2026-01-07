use dear_imgui_rs::{self as imgui, Drag, Key};
use legion::*;
use nalgebra::{Matrix4, UnitQuaternion};
use std::any::TypeId;
use std::collections::HashMap;

use crate::camera::Camera;
use crate::components::{MaterialOverride, MeshComponent, Transform, Visible};
use crate::scene::WorldExt;
use crate::scene::entity_uuid::EntityUuid;
use crate::submission::SubmeshSelection;

// Add these imports for ImGuizmo

use dear_imguizmo::{self as imguizmo, GuizmoExt, Operation};

/// Trait for editing a specific component type in ImGui
pub trait ComponentEditor: Send + Sync {
    fn type_name(&self) -> &'static str;
    fn type_id(&self) -> TypeId;

    /// Draw the component editor. Returns true if modified.
    fn draw_editor(
        &self,
        world: &mut World,
        entity: Entity,
        ui: &imgui::Ui,
        ctx: &mut EditorContext,
    ) -> bool;

    /// Check if entity has this component
    fn has_component(&self, world: &World, entity: Entity) -> bool;

    /// Add default instance
    fn add_default(&self, world: &mut World, entity: Entity);

    /// Remove component
    fn remove(&self, world: &mut World, entity: Entity);
}

/// Registry of component editors
pub struct EditorRegistry {
    editors: Vec<Box<dyn ComponentEditor>>,
}

impl EditorRegistry {
    pub fn new() -> Self {
        Self {
            editors: Vec::new(),
        }
    }

    pub fn register<E: ComponentEditor + 'static>(&mut self, editor: E) -> &mut Self {
        self.editors.push(Box::new(editor));
        self
    }

    pub fn editors(&self) -> &[Box<dyn ComponentEditor>] {
        &self.editors
    }
}

/// Commands to execute after UI frame
enum SceneCommand {
    CreateEntity,
    DeleteEntity(EntityUuid),
}

/// Panel state that persists across frames
pub struct ScenePanel {
    selected_entity_uuid: Option<EntityUuid>,
    editor_registry: EditorRegistry,
    commands: Vec<SceneCommand>,
    rotation_deltas: HashMap<EntityUuid, [f32; 3]>,

    // Gizmo state
    gizmo_operation: imguizmo::Operation,
    gizmo_mode: imguizmo::Mode,
    gizmo_enabled: bool,
    gizmo_using: bool,
}

pub struct EditorContext<'a> {
    pub rotation_deltas: &'a mut HashMap<EntityUuid, [f32; 3]>,
}

impl ScenePanel {
    pub fn new() -> Self {
        let mut registry = EditorRegistry::new();

        // Register all component editors
        registry
            .register(TransformEditor)
            .register(MeshComponentEditor)
            .register(VisibleEditor)
            .register(MaterialOverrideEditor);

        Self {
            selected_entity_uuid: None,
            editor_registry: registry,
            commands: Vec::new(),
            rotation_deltas: HashMap::new(),
            gizmo_operation: imguizmo::Operation::TRANSLATE,
            gizmo_mode: imguizmo::Mode::Local,
            gizmo_enabled: true,
            gizmo_using: false,
        }
    }

    pub fn select(&mut self, uuid: EntityUuid) {
        self.selected_entity_uuid = Some(uuid);
    }

    pub fn clear_selection(&mut self) {
        self.selected_entity_uuid = None;
    }

    /// Find Legion Entity from EntityUuid
    fn find_entity(&self, world: &World, uuid: EntityUuid) -> Option<Entity> {
        let mut query = <(Entity, &EntityUuid)>::query();
        query
            .iter(world)
            .find(|(_, eu)| **eu == uuid)
            .map(|(e, _)| *e)
    }

    /// Main draw call - world access is split for borrow checker
    pub fn draw(&mut self, ui: &imgui::Ui, world: &mut World) {
        let commands = std::mem::take(&mut self.commands);

        for cmd in commands {
            match cmd {
                SceneCommand::CreateEntity => {
                    let (_, uuid) = world.push_with_uuid(());
                    self.selected_entity_uuid = Some(uuid);
                }
                SceneCommand::DeleteEntity(uuid) => {
                    if let Some(entity) = self.find_entity(world, uuid) {
                        world.remove(entity);
                        if self.selected_entity_uuid == Some(uuid) {
                            self.selected_entity_uuid = None;
                        }
                    }
                }
            }
        }

        ui.window("Scene")
            .size([350.0, 600.0], imgui::Condition::FirstUseEver)
            .build(|| {
                self.draw_entity_list(ui, world);
                ui.separator();
                self.draw_gizmo_controls(ui);
                ui.separator();
                self.draw_inspector(ui, world);
            });
    }

    fn draw_entity_list(&mut self, ui: &imgui::Ui, world: &World) {
        ui.text("Entities");
        ui.same_line();

        if ui.button("Create") {
            self.commands.push(SceneCommand::CreateEntity);
        }

        ui.separator();

        ui.child_window("entity_list")
            .size([0.0, 200.0])
            .border(true)
            .build(&ui, || {
                let mut query = <(Entity, &EntityUuid)>::query();
                let mut entities: Vec<_> = query.iter(world).collect();

                // Sort for stable order
                entities.sort_by_key(|(_, uuid)| uuid.0);

                for (_, uuid) in entities {
                    let is_selected = self.selected_entity_uuid == Some(*uuid);

                    // Show first 13 chars of UUID
                    let uuid_str = format!("{}", uuid.0);
                    let short_uuid = &uuid_str[..uuid_str.len().min(13)];
                    let label = format!("{}", short_uuid);

                    if ui.selectable_config(&label).selected(is_selected).build() {
                        self.selected_entity_uuid = Some(*uuid);
                    }
                }
            });
    }

    fn draw_gizmo_controls(&mut self, ui: &imgui::Ui) {
        ui.text("Gizmo Controls");

        ui.checkbox("Enable Gizmo", &mut self.gizmo_enabled);

        if !self.gizmo_enabled {
            return;
        }

        ui.text("Operation:");
        if ui.radio_button(
            "Translate (W)",
            self.gizmo_operation == imguizmo::Operation::TRANSLATE,
        ) {
            self.gizmo_operation = imguizmo::Operation::TRANSLATE;
        }
        ui.same_line();
        if ui.radio_button(
            "Rotate (E)",
            self.gizmo_operation == imguizmo::Operation::ROTATE,
        ) {
            self.gizmo_operation = imguizmo::Operation::ROTATE;
        }
        ui.same_line();
        if ui.radio_button(
            "Scale (R)",
            self.gizmo_operation == imguizmo::Operation::SCALE,
        ) {
            self.gizmo_operation = imguizmo::Operation::SCALE;
        }

        ui.text("Mode:");
        if ui.radio_button("Local", self.gizmo_mode == imguizmo::Mode::Local) {
            self.gizmo_mode = imguizmo::Mode::Local;
        }
        ui.same_line();
        if ui.radio_button("World", self.gizmo_mode == imguizmo::Mode::World) {
            self.gizmo_mode = imguizmo::Mode::World;
        }
    }

    fn draw_inspector(&mut self, ui: &imgui::Ui, world: &mut World) {
        ui.text("Inspector");
        ui.separator();

        let Some(selected_uuid) = self.selected_entity_uuid else {
            ui.text_disabled("No entity selected");
            return;
        };

        let Some(entity) = self.find_entity(world, selected_uuid) else {
            ui.text_colored([1.0, 0.5, 0.0, 1.0], "Entity was deleted");
            self.selected_entity_uuid = None;
            return;
        };

        // Entity header with UUID
        let uuid_str = format!("{}", selected_uuid.0);
        let short_uuid = &uuid_str[..uuid_str.len().min(13)];
        ui.text(format!("Entity {}", short_uuid));

        if ui.button("Delete") {
            self.commands
                .push(SceneCommand::DeleteEntity(selected_uuid));
            return;
        }

        ui.same_line();
        if ui.button("Add Component") {
            ui.open_popup("add_component_popup");
        }

        ui.separator();

        // Draw components
        self.draw_components(ui, world, entity);

        // Add component popup
        ui.popup("add_component_popup", || {
            ui.text("Add Component");
            ui.separator();

            for editor in self.editor_registry.editors() {
                let has_component = editor.has_component(world, entity);
                let selected = false;

                let label = if has_component {
                    format!("{} (exists)", editor.type_name())
                } else {
                    editor.type_name().to_string()
                };

                if ui.menu_item_enabled_selected(label, None::<&str>, selected, !has_component) {
                    editor.add_default(world, entity);
                    ui.close_current_popup();
                }
            }
        });
    }

    fn draw_components(&mut self, ui: &imgui::Ui, world: &mut World, entity: Entity) {
        let mut ctx = EditorContext {
            rotation_deltas: &mut self.rotation_deltas,
        };

        ui.child_window("components_scroll")
            .size([0.0, 0.0])
            .build(&ui, || {
                // Collect which components exist
                let has_components: Vec<(usize, &'static str)> = self
                    .editor_registry
                    .editors()
                    .iter()
                    .enumerate()
                    .filter(|(_, editor)| editor.has_component(world, entity))
                    .map(|(i, editor)| (i, editor.type_name()))
                    .collect();

                for (idx, type_name) in has_components {
                    let editor = &self.editor_registry.editors()[idx];

                    let header_id = format!("{}##{}", type_name, idx);
                    if ui.collapsing_header(&header_id, imgui::TreeNodeFlags::DEFAULT_OPEN) {
                        ui.indent();

                        // Draw editor
                        editor.draw_editor(world, entity, ui, &mut ctx);

                        ui.spacing();

                        // Remove button (EntityUuid cannot be removed)
                        if editor.type_id() != TypeId::of::<EntityUuid>() {
                            let remove_id = format!("Remove##remove_{}", idx);
                            if ui.button(&remove_id) {
                                editor.remove(world, entity);
                            }
                        }

                        ui.unindent();
                    }

                    ui.spacing();
                }
            });
    }

    /// Draw the gizmo in viewport space - call this after all other UI
    pub fn draw_gizmo(
        &mut self,
        ui: &imgui::Ui,
        camera: &Camera,
        world: &mut World,
        viewport_rect: Option<([f32; 2], [f32; 2])>,
    ) {
        if !self.gizmo_enabled {
            return;
        }

        let Some((vp_pos, vp_size)) = viewport_rect else {
            return;
        };

        let Some(selected_uuid) = self.selected_entity_uuid else {
            return;
        };

        let Some(entity) = self.find_entity(world, selected_uuid) else {
            return;
        };

        let has_transform = world
            .entry_ref(entity)
            .map_or(false, |e| e.get_component::<Transform>().is_ok());

        if !has_transform {
            return;
        }

        let model_matrix = {
            let entry = world.entry_ref(entity).unwrap();
            let transform = entry.get_component::<Transform>().unwrap();
            transform_to_matrix4(transform)
        };

        let gizmo = ui.guizmo();
        gizmo.set_rect(vp_pos[0], vp_pos[1], vp_size[0], vp_size[1]);

        gizmo.set_drawlist_background();
        gizmo.set_orthographic(false);
        gizmo.set_gizmo_size_clip_space(0.1);

        let view = camera.view_matrix();

        // Use LEFT-HANDED projection for ImGuizmo
        let proj_matrix = imgui_perspective(72.0, vp_size[0] / vp_size[1], 0.1, 1000.0);

        fn matrix_to_array(m: Matrix4<f32>) -> [f32; 16] {
            m.as_slice().try_into().unwrap()
        }

        let view_slice = matrix_to_array(view);
        let proj_slice = matrix_to_array(proj_matrix);
        let mut model_slice = matrix_to_array(model_matrix);

        let used: bool = gizmo
            .manipulate_config(&view_slice, &proj_slice, &mut model_slice)
            .operation(self.gizmo_operation)
            .mode(self.gizmo_mode)
            .build();

        self.gizmo_using = used || gizmo.is_using();

        if used {
            // Only update when actively manipulating
            let (translation, rotation, scale) = imguizmo::decompose_matrix(&model_slice);

            if let Some(mut entry) = world.entry(entity) {
                if let Ok(transform) = entry.get_component_mut::<Transform>() {
                    transform.position = glm::vec3(translation[0], translation[1], translation[2]);
                    transform.scale = glm::vec3(scale[0], scale[1], scale[2]);
                    transform.rotation =
                        UnitQuaternion::from_euler_angles(rotation[0], rotation[1], rotation[2])
                            .into_inner();
                }
            }
        }
    }

    pub fn is_gizmo_using(&self) -> bool {
        self.gizmo_using
    }

    /// Handle keyboard shortcuts for gizmo
    pub fn handle_input(&mut self, ui: &imgui::Ui) {
        if !self.gizmo_enabled {
            return;
        }

        // Check if any text input is active
        if ui.is_any_item_active() {
            return;
        }

        // W - Translate
        if ui.is_key_pressed(Key::W) {
            self.gizmo_operation = Operation::TRANSLATE;
        }
        // E - Rotate
        if ui.is_key_pressed(Key::E) {
            self.gizmo_operation = Operation::ROTATE;
        }
        // R - Scale
        if ui.is_key_pressed(Key::R) {
            self.gizmo_operation = Operation::SCALE;
        }
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn transform_to_matrix4(transform: &Transform) -> Matrix4<f32> {
    let translation = nalgebra::Translation3::from(transform.position);
    let rotation = nalgebra::UnitQuaternion::from_quaternion(transform.rotation);
    let scale = nalgebra::Scale3::from(transform.scale);

    let matrix = translation.to_homogeneous() * rotation.to_homogeneous() * scale.to_homogeneous();
    matrix
}

// ============================================================================
// COMPONENT EDITORS
// ============================================================================

pub struct TransformEditor;

impl ComponentEditor for TransformEditor {
    fn type_name(&self) -> &'static str {
        "Transform"
    }

    fn type_id(&self) -> TypeId {
        TypeId::of::<Transform>()
    }

    fn draw_editor(
        &self,
        world: &mut World,
        entity: Entity,
        ui: &imgui::Ui,
        ctx: &mut EditorContext,
    ) -> bool {
        let mut modified = false;

        let uuid = match world.entry_ref(entity) {
            Ok(e) => e.get_component::<EntityUuid>().ok().copied(),
            Err(_) => None,
        };

        let Some(uuid) = uuid else {
            return false;
        };

        if let Some(mut entry) = world.entry(entity) {
            if let Ok(transform) = entry.get_component_mut::<Transform>() {
                let mut pos = [
                    transform.position.x,
                    transform.position.y,
                    transform.position.z,
                ];

                if Drag::new("Position").speed(0.1).build_array(ui, &mut pos) {
                    transform.position = glm::Vec3::new(pos[0], pos[1], pos[2]);
                    modified = true;
                }

                let delta = ctx.rotation_deltas.entry(uuid).or_insert([0.0, 0.0, 0.0]);

                if imgui::Drag::new("Rotate Δ (deg)")
                    .speed(0.5)
                    .build_array(ui, delta)
                {
                    let dq = UnitQuaternion::from_euler_angles(
                        delta[0].to_radians(),
                        delta[1].to_radians(),
                        delta[2].to_radians(),
                    );

                    let current = UnitQuaternion::from_quaternion(transform.rotation);
                    transform.rotation = (dq * current).into_inner();

                    *delta = [0.0, 0.0, 0.0];
                    modified = true;
                }

                let mut scale = [transform.scale.x, transform.scale.y, transform.scale.z];

                if imgui::Drag::new("Scale")
                    .speed(0.01)
                    .build_array(ui, &mut scale)
                {
                    scale = [scale[0].max(0.0), scale[1].max(0.0), scale[2].max(0.0)];

                    transform.scale = glm::Vec3::new(scale[0], scale[1], scale[2]);
                    modified = true;
                }
            }
        }

        modified
    }

    fn has_component(&self, world: &World, entity: Entity) -> bool {
        world.has_component::<Transform>(entity)
    }

    fn add_default(&self, world: &mut World, entity: Entity) {
        if let Some(mut entry) = world.entry(entity) {
            entry.add_component(Transform {
                position: glm::Vec3::zeros(),
                scale: glm::Vec3::new(1.0, 1.0, 1.0),
                rotation: glm::Quat::identity(),
            });
        }
    }

    fn remove(&self, world: &mut World, entity: Entity) {
        if let Some(mut entry) = world.entry(entity) {
            let _ = entry.remove_component::<Transform>();
        }
    }
}

pub struct MeshComponentEditor;

impl MeshComponentEditor {
    fn draw_mesh_component_editor(
        &self,
        world: &mut World,
        entity: Entity,
        ui: &imgui::Ui,
    ) -> bool {
        let mut modified = false;

        let mut entry = match world.entry(entity) {
            Some(e) => e,
            None => return false,
        };

        let mesh_comp = match entry.get_component_mut::<MeshComponent>() {
            Ok(c) => c,
            Err(_) => return false,
        };

        /* ---------------- Mesh handle ---------------- */

        let mut handle_i32 = mesh_comp.mesh.0 as i32;
        if ui.input_int("Mesh Handle", &mut handle_i32) {
            let new_id = handle_i32.max(0) as u32;
            if new_id != mesh_comp.mesh.0 {
                mesh_comp.mesh.0 = new_id;
                modified = true;
            }
        }

        ui.text_disabled(format!("(Current: {})", mesh_comp.mesh.0));

        ui.separator();

        /* ---------------- Submesh selection ---------------- */

        let mut new_selection = None;

        match mesh_comp.submeshes {
            SubmeshSelection::All => {
                if ui.radio_button("All submeshes", true) {
                    // already selected
                }

                if ui.radio_button("One submesh", false) {
                    new_selection = Some(SubmeshSelection::One(0));
                }
            }

            SubmeshSelection::One(index) => {
                if ui.radio_button("All submeshes", false) {
                    new_selection = Some(SubmeshSelection::All);
                }

                if ui.radio_button("One submesh", true) {
                    // already selected
                }

                let mut idx_i32 = index as i32;
                if ui.input_int("Submesh index", &mut idx_i32) {
                    let new_idx = idx_i32.max(0) as u32;
                    if new_idx != index {
                        new_selection = Some(SubmeshSelection::One(new_idx));
                    }
                }
            }
        }

        if let Some(sel) = new_selection {
            if sel != mesh_comp.submeshes {
                mesh_comp.submeshes = sel;
                modified = true;
            }
        }

        modified
    }
}

impl ComponentEditor for MeshComponentEditor {
    fn type_name(&self) -> &'static str {
        "MeshComponent"
    }

    fn type_id(&self) -> TypeId {
        TypeId::of::<MeshComponent>()
    }

    fn draw_editor(
        &self,
        world: &mut World,
        entity: Entity,
        ui: &imgui::Ui,
        _context: &mut EditorContext,
    ) -> bool {
        self.draw_mesh_component_editor(world, entity, ui)
    }

    fn has_component(&self, world: &World, entity: Entity) -> bool {
        world.has_component::<MeshComponent>(entity)
    }

    fn add_default(&self, world: &mut World, entity: Entity) {
        if let Some(mut entry) = world.entry(entity) {
            entry.add_component(MeshComponent {
                mesh: crate::mesh_registry::MeshHandle(0),
                submeshes: SubmeshSelection::All,
            });
        }
    }

    fn remove(&self, world: &mut World, entity: Entity) {
        if let Some(mut entry) = world.entry(entity) {
            let _ = entry.remove_component::<MeshComponent>();
        }
    }
}

pub struct VisibleEditor;

impl ComponentEditor for VisibleEditor {
    fn type_name(&self) -> &'static str {
        "Visible"
    }

    fn type_id(&self) -> TypeId {
        TypeId::of::<Visible>()
    }

    fn draw_editor(
        &self,
        _world: &mut World,
        _entity: Entity,
        ui: &imgui::Ui,
        _context: &mut EditorContext,
    ) -> bool {
        ui.text_disabled("(Marker component)");
        false
    }

    fn has_component(&self, world: &World, entity: Entity) -> bool {
        world.has_component::<Visible>(entity)
    }

    fn add_default(&self, world: &mut World, entity: Entity) {
        if let Some(mut entry) = world.entry(entity) {
            entry.add_component(Visible);
        }
    }

    fn remove(&self, world: &mut World, entity: Entity) {
        if let Some(mut entry) = world.entry(entity) {
            let _ = entry.remove_component::<Visible>();
        }
    }
}

pub struct MaterialOverrideEditor;

impl ComponentEditor for MaterialOverrideEditor {
    fn type_name(&self) -> &'static str {
        "MaterialOverride"
    }

    fn type_id(&self) -> TypeId {
        TypeId::of::<MaterialOverride>()
    }

    fn draw_editor(
        &self,
        world: &mut World,
        entity: Entity,
        ui: &imgui::Ui,
        _context: &mut EditorContext,
    ) -> bool {
        let mut modified = false;

        if let Some(mut entry) = world.entry(entity) {
            if let Ok(mat_override) = entry.get_component_mut::<MaterialOverride>() {
                let mut mat_id = mat_override.material_id as i32;

                if ui.input_int("Material ID", &mut mat_id) {
                    mat_override.material_id = mat_id.max(0) as u32;
                    modified = true;
                }
            }
        }

        modified
    }

    fn has_component(&self, world: &World, entity: Entity) -> bool {
        world.has_component::<MaterialOverride>(entity)
    }

    fn add_default(&self, world: &mut World, entity: Entity) {
        if let Some(mut entry) = world.entry(entity) {
            entry.add_component(MaterialOverride { material_id: 0 });
        }
    }

    fn remove(&self, world: &mut World, entity: Entity) {
        if let Some(mut entry) = world.entry(entity) {
            let _ = entry.remove_component::<MaterialOverride>();
        }
    }
}

fn imgui_perspective(vertical_fov_deg: f32, aspect: f32, near: f32, far: f32) -> Matrix4<f32> {
    let fov_rad = vertical_fov_deg.to_radians();
    let f = 1.0 / (fov_rad * 0.5).tan();

    // OpenGL-style projection for ImGuizmo
    Matrix4::from_column_slice(&[
        f / aspect,
        0.0,
        0.0,
        0.0,
        0.0,
        f, // No flip needed for ImGuizmo
        0.0,
        0.0,
        0.0,
        0.0,
        -(far + near) / (far - near), // Changed from + to -
        -1.0,
        0.0,
        0.0,
        -(2.0 * far * near) / (far - near), // Changed from + to -
        0.0,
    ])
}

#[cfg(test)]
mod test {
    use super::*;

    // Helper function to compare matrices with epsilon tolerance
    fn matrices_approx_equal(a: &Matrix4<f32>, b: &Matrix4<f32>, epsilon: f32) -> bool {
        for i in 0..4 {
            for j in 0..4 {
                if (a[(i, j)] - b[(i, j)]).abs() > epsilon {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn compare_projection_matrices_parametrized() {
        let test_cases: Vec<(f32, f32, f32, f32, &str)> = vec![
            // (fov_deg, aspect, near, far, description)
            (60.0, 16.0 / 9.0, 0.1, 100.0, "standard case"),
            (45.0, 16.0 / 9.0, 0.1, 1000.0, "narrow fov, far plane"),
            (90.0, 16.0 / 9.0, 0.1, 100.0, "wide fov"),
            (120.0, 16.0 / 9.0, 0.1, 100.0, "ultra-wide fov"),
            (30.0, 16.0 / 9.0, 0.1, 100.0, "telephoto"),
            // Different aspect ratios
            (60.0, 4.0 / 3.0, 0.1, 100.0, "4:3 aspect"),
            (60.0, 1.0, 0.1, 100.0, "square aspect"),
            (60.0, 21.0 / 9.0, 0.1, 100.0, "ultrawide 21:9"),
            (60.0, 32.0 / 9.0, 0.1, 100.0, "super ultrawide"),
            (60.0, 9.0 / 16.0, 0.1, 100.0, "portrait mode"),
            // Different near planes
            (60.0, 16.0 / 9.0, 0.001, 100.0, "very close near"),
            (60.0, 16.0 / 9.0, 0.01, 100.0, "close near"),
            (60.0, 16.0 / 9.0, 1.0, 100.0, "far near"),
            (60.0, 16.0 / 9.0, 5.0, 100.0, "very far near"),
            // Different far planes
            (60.0, 16.0 / 9.0, 0.1, 10.0, "close far"),
            (60.0, 16.0 / 9.0, 0.1, 500.0, "medium far"),
            (60.0, 16.0 / 9.0, 0.1, 5000.0, "far far"),
            (60.0, 16.0 / 9.0, 0.1, 20000.0, "very far far"),
            // Edge cases
            (60.0, 16.0 / 9.0, 0.001, 20000.0, "extreme depth range"),
            (15.0, 16.0 / 9.0, 0.1, 100.0, "very narrow fov"),
            (170.0, 16.0 / 9.0, 0.1, 100.0, "fisheye fov"),
            (60.0, 0.5, 0.1, 100.0, "narrow aspect"),
            (60.0, 3.0, 0.1, 100.0, "wide aspect"),
            // Common game/graphics scenarios
            (72.0, 16.0 / 9.0, 0.1, 1000.0, "typical game camera"),
            (75.0, 16.0 / 9.0, 0.1, 2000.0, "fps game camera"),
            (50.0, 16.0 / 9.0, 0.5, 5000.0, "third person camera"),
            (80.0, 21.0 / 9.0, 0.1, 3000.0, "racing game camera"),
        ];

        const EPSILON: f32 = 1e-5;

        for (fov_deg, aspect, near, far, description) in test_cases {
            let glm_projection = glm::perspective_rh_no(aspect, fov_deg.to_radians(), near, far);
            let imgui_projection = imgui_perspective(fov_deg, aspect, near, far);

            assert!(
                matrices_approx_equal(&glm_projection, &imgui_projection, EPSILON),
                "Failed for case: {} (fov={}, aspect={}, near={}, far={})\nGLM: {:?}\nImGui: {:?}",
                description,
                fov_deg,
                aspect,
                near,
                far,
                glm_projection,
                imgui_projection
            );
        }
    }

    #[test]
    fn compare_projection_matrices_exhaustive() {
        // More exhaustive testing with ranges
        let fov_degrees: Vec<f32> = vec![15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 110.0, 120.0];
        let aspects = vec![0.5, 4.0 / 3.0, 16.0 / 9.0, 1.0, 21.0 / 9.0, 2.0, 3.0];
        let nears = vec![0.001, 0.01, 0.1, 0.5, 1.0, 5.0];
        let fars = vec![10.0, 100.0, 500.0, 1000.0, 5000.0, 20000.0];

        const EPSILON: f32 = 1e-5;
        let mut test_count = 0;

        for &fov in &fov_degrees {
            for &aspect in &aspects {
                for &near in &nears {
                    for &far in &fars {
                        // Skip invalid cases where near >= far
                        if near >= far {
                            continue;
                        }

                        let glm_projection =
                            glm::perspective_rh_no(aspect, fov.to_radians(), near, far);
                        let imgui_projection = imgui_perspective(fov, aspect, near, far);

                        assert!(
                            matrices_approx_equal(&glm_projection, &imgui_projection, EPSILON),
                            "Failed for fov={}, aspect={}, near={}, far={}\nGLM: {:?}\nImGui: {:?}",
                            fov,
                            aspect,
                            near,
                            far,
                            glm_projection,
                            imgui_projection
                        );

                        test_count += 1;
                    }
                }
            }
        }

        println!(
            "Ran {} projection matrix comparison tests successfully!",
            test_count
        );
    }

    #[test]
    fn test_projection_matrix_properties() {
        // Test that the matrices maintain expected properties
        let fov_deg = 60.0f32;
        let aspect = 16.0 / 9.0;
        let near = 0.1;
        let far = 100.0;

        let proj = imgui_perspective(fov_deg, aspect, near, far);

        // Check that the matrix has the expected structure
        // [0,0] should be f/aspect
        let f = 1.0 / (fov_deg.to_radians() * 0.5).tan();
        assert!(
            (proj[(0, 0)] - f / aspect).abs() < 1e-5,
            "x-scale incorrect"
        );

        // [1,1] should be f
        assert!((proj[(1, 1)] - f).abs() < 1e-5, "y-scale incorrect");

        // [2,3] should be -1.0 for right-handed
        assert!(
            (proj[(3, 2)] - (-1.0)).abs() < 1e-5,
            "perspective divide incorrect"
        );

        // [3,3] should be 0 (homogeneous coordinate)
        assert!((proj[(3, 3)] - 0.0).abs() < 1e-5, "w component incorrect");
    }
}
