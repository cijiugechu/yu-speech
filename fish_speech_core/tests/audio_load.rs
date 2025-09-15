use candle_core::Device;
use fish_speech_core::audio::{load, load_from_memory};

#[test]
fn load_from_file_and_memory_are_consistent() {
    let device = Device::Cpu;
    let fixture =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/resources/sky.wav");

    // Load from file
    let (t_file, sr_file) = load(&fixture, &device).expect("load from file");

    // Load from memory
    let bytes = std::fs::read(&fixture).expect("read wav bytes");
    let (t_mem, sr_mem) = load_from_memory(bytes, &device).expect("load from memory");

    assert_eq!(sr_file, sr_mem);

    let (cf, nf) = t_file.dims2().expect("dims file");
    let (cm, nm) = t_mem.dims2().expect("dims mem");
    assert_eq!((cf, nf), (cm, nm));

    // Compare a small prefix within tolerance (floats)
    let v_file: Vec<f32> = t_file.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let v_mem: Vec<f32> = t_mem.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let take = v_file.len().min(256);
    let mut max_abs = 0.0f32;
    for i in 0..take {
        max_abs = max_abs.max((v_file[i] - v_mem[i]).abs());
    }
    assert!(max_abs <= 1e-5, "max abs diff: {}", max_abs);
}
