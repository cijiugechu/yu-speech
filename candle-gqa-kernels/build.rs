fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/compatibility.cuh");
    println!("cargo:rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo:rerun-if-changed=src/binary_op_macros.cuh");
    println!("cargo:rerun-if-changed=src/unary.cu");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let unary_ptx_path = std::path::Path::new(&out_dir).join("unary.ptx");

    if target_os == "macos" {
        // No CUDA on macOS: static src/lib.rs provides an empty UNARY.
        println!("cargo:warning=CUDA kernels are stubbed on macOS; this crate will be a no-op.");
        return;
    }

    // Try to build PTX. If it fails (e.g. CUDA not installed), fall back to stub to avoid
    // breaking workspace builds and rust-analyzer.
    let builder = bindgen_cuda::Builder::default();
    println!("cargo:info={builder:?}");
    match builder.build_ptx() {
        Ok(_bindings) => {
            // PTX files are emitted into OUT_DIR by bindgen_cuda. The static src/lib.rs
            // includes OUT_DIR/unary.ptx via include_str!, so nothing else to do here.
        }
        Err(err) => {
            println!(
                "cargo:warning=Failed to build CUDA PTX ({:?}). Falling back to empty PTX.",
                err
            );
            // Ensure an empty file exists at OUT_DIR/unary.ptx to satisfy include_str!.
            let _ = std::fs::write(&unary_ptx_path, "");
        }
    }
}
