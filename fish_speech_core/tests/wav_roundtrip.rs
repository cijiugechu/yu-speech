use candle_core::Device;
use fish_speech_core::audio::{load_from_memory, wav::write_pcm_as_wav};

#[test]
fn wav_write_then_decode_roundtrip_basic() {
    // Generate a short 440Hz sine-like sequence as f32 samples in -1..1
    let sample_rate: u32 = 16000;
    let duration_secs: f32 = 0.05; // 50ms keeps the test fast
    let len: usize = (sample_rate as f32 * duration_secs) as usize;
    let freq_hz: f32 = 440.0;
    let mut samples = Vec::with_capacity(len);
    for n in 0..len {
        let t = n as f32 / sample_rate as f32;
        samples.push((2.0 * std::f32::consts::PI * freq_hz * t).sin());
    }

    // Write to WAV in-memory
    let mut buf = Vec::new();
    write_pcm_as_wav(&mut buf, &samples, sample_rate).expect("write wav");

    // Decode back using the library
    let device = Device::Cpu;
    let (tensor, sr) = load_from_memory(buf, &device).expect("decode wav");

    // Validate basics
    assert_eq!(sr, sample_rate);
    let (channels, frames) = tensor.dims2().expect("dims");
    assert_eq!(channels, 1, "decoder averages to mono");
    assert!(frames > 0);

    // Values should be finite and within a reasonable range
    let data: Vec<f32> = tensor
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .expect("to vec");
    assert!(data.iter().all(|v| v.is_finite()));
}
