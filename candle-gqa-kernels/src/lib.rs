#[cfg(not(target_os = "macos"))]
pub const UNARY: &str = include_str!(concat!(env!("OUT_DIR"), "/unary.ptx"));

#[cfg(target_os = "macos")]
pub const UNARY: &str = "";
