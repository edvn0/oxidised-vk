use imgui::*;
use legion::*;
use nalgebra::UnitQuaternion;
use std::any::TypeId;
use std::collections::HashMap;

use crate::components::{MaterialOverride, MeshComponent, Transform, Visible};
use crate::scene::WorldExt;
use crate::scene::entity_uuid::EntityUuid;

/// Trait for editing a specific component type in ImGui
pub trait ComponentEditor: Send + Sync {
    fn type_name(&self) -> &'static str;
    fn type_id(&self) -> TypeId;

    /// Draw the component editor. Returns true if modified.
    fn draw_editor(
        &self,
        world: &mut World,
        entity: Entity,
        ui: &Ui,
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
        }
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
    pub fn draw(&mut self, ui: &Ui, world: &mut World) {
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
            .size([350.0, 600.0], Condition::FirstUseEver)
            .build(|| {
                self.draw_entity_list(ui, world);
                ui.separator();
                self.draw_inspector(ui, world);
            });
    }

    fn draw_entity_list(&mut self, ui: &Ui, world: &World) {
        ui.text("Entities");
        ui.same_line();

        if ui.button("Create") {
            self.commands.push(SceneCommand::CreateEntity);
        }

        ui.separator();

        ui.child_window("entity_list")
            .size([0.0, 200.0])
            .border(true)
            .build(|| {
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

    fn draw_inspector(&mut self, ui: &Ui, world: &mut World) {
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

                ui.enabled(!has_component, || {
                    let label = if has_component {
                        format!("{} (exists)", editor.type_name())
                    } else {
                        editor.type_name().to_string()
                    };

                    if ui.button(&label) {
                        editor.add_default(world, entity);
                        ui.close_current_popup();
                    }
                });
            }
        });
    }

    fn draw_components(&mut self, ui: &Ui, world: &mut World, entity: Entity) {
        let mut ctx = EditorContext {
            rotation_deltas: &mut self.rotation_deltas,
        };

        ui.child_window("components_scroll")
            .size([0.0, 0.0])
            .build(|| {
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
                    if ui.collapsing_header(&header_id, TreeNodeFlags::DEFAULT_OPEN) {
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
        ui: &Ui,
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

                if imgui::Drag::new("Position")
                    .speed(0.1)
                    .build_array(ui, &mut pos)
                {
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
        ui: &Ui,
        _context: &mut EditorContext,
    ) -> bool {
        let mut modified = false;

        if let Some(mut entry) = world.entry(entity) {
            if let Ok(mesh_comp) = entry.get_component_mut::<MeshComponent>() {
                let mut handle_id = mesh_comp.mesh.0 as i32;

                if ui.input_int("Mesh Handle", &mut handle_id).build() {
                    mesh_comp.mesh.0 = handle_id.max(0) as u32;
                    modified = true;
                }

                ui.text_disabled(format!("(Current: {})", mesh_comp.mesh.0));
            }
        }

        modified
    }

    fn has_component(&self, world: &World, entity: Entity) -> bool {
        world.has_component::<MeshComponent>(entity)
    }

    fn add_default(&self, world: &mut World, entity: Entity) {
        if let Some(mut entry) = world.entry(entity) {
            entry.add_component(MeshComponent {
                mesh: crate::mesh_registry::MeshHandle(0),
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
        ui: &Ui,
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
        ui: &Ui,
        _context: &mut EditorContext,
    ) -> bool {
        let mut modified = false;

        if let Some(mut entry) = world.entry(entity) {
            if let Ok(mat_override) = entry.get_component_mut::<MaterialOverride>() {
                let mut mat_id = mat_override.material_id as i32;

                if ui.input_int("Material ID", &mut mat_id).build() {
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
