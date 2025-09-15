use candle_core::{Device, Tensor};
use fish_speech_core::audio::functional::resample;

fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() <= eps
}

#[test]
fn resample_linear_2x_on_simple_signal() {
    let device = Device::Cpu;

    // 1-channel signal: [0.0, 1.0, 0.0]
    let input = Tensor::from_slice(&[0.0f32, 1.0, 0.0], (1, 3), &device).unwrap();

    // Upsample 2x using from_rate=3 -> to_rate=6
    let up = resample(&input, 3, 6).unwrap();
    let (c, n) = up.dims2().unwrap();
    assert_eq!(c, 1);
    assert_eq!(n, 6);

    // Expected linear interpolation:
    // indices: 0,0.5,1,1.5,2,2.5
    // values:  0, 0.5, 1, 0.5, 0, 0
    let v: Vec<f32> = up.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let expected = [0.0f32, 0.5, 1.0, 0.5, 0.0, 0.0];
    assert_eq!(v.len(), expected.len());
    for (a, b) in v.iter().zip(expected.iter()) {
        assert!(approx_eq(*a, *b, 1e-5), "{} vs {}", a, b);
    }
}
