use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=scripts/generate_icons.lua");

    let status = Command::new("lua")
        .arg("scripts/generate_icons.lua")
        .status()
        .expect("Failed to run Lua");

    if !status.success() {
        panic!("Icon generation failed");
    }
}
