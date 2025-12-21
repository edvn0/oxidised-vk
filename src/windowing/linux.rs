use winit::window::{Icon, Window};

pub fn set_window_icons(window: &Window) {
    const ICON_32: &[u8] = include_bytes!("../../assets/engine/icon_32.rgba");

    let small = Icon::from_rgba(ICON_32.to_vec(), 32, 32);

    if let Ok(icon) = small {
        window.set_window_icon(Some(icon));
    }
}
