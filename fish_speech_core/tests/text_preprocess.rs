use fish_speech_core::text::clean::preprocess_text;

#[test]
fn preprocess_mixed_languages_splits_sentences() {
    let input = "Hello world! これはテストです。你好，世界！";
    let chunks = preprocess_text(input);
    assert!(!chunks.is_empty());
    assert!(chunks.len() >= 3);
}
