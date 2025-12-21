use winit::platform::windows::WindowExtWindows;
use winit::window::{Icon, Window};

pub fn set_window_icons(window: &Window) {
    const ICON_256: &[u8] = include_bytes!("../../assets/engine/icon_256.rgba");
    const ICON_32: &[u8] = include_bytes!("../../assets/engine/icon_32.rgba");

    let big = Icon::from_rgba(ICON_256.to_vec(), 256, 256);
    let small = Icon::from_rgba(ICON_32.to_vec(), 32, 32);

    if let (Ok(big), Ok(small)) = (big, small) {
        window.set_taskbar_icon(Some(big));
        window.set_window_icon(Some(small));
    }
}
