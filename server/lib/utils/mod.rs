pub mod load;

use candle_core::{Device, Tensor};
use fish_speech_core::config::WhichLM;
use fish_speech_core::text::prompt::{PromptEncoder, load_prompt_text};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tokenizers::Tokenizer;

#[derive(Serialize, Deserialize, Default)]
struct SpeakerIndex {
    // filename (without .npy) -> prompt text
    speakers: HashMap<String, String>,
}

pub fn load_speaker_prompts(
    voice_dir: &Path,
    tokenizer: &Tokenizer,
    device: &Device,
    num_codebooks: usize,
    model_type: WhichLM,
) -> anyhow::Result<(HashMap<String, Tensor>, Tensor)> {
    // Load the index file if present; otherwise start with empty map
    let index_path = voice_dir.join("index.json");
    let mut index: SpeakerIndex = if index_path.exists() {
        serde_json::from_reader(std::fs::File::open(&index_path)?).unwrap_or_default()
    } else {
        SpeakerIndex::default()
    };

    // Discover .npy files in voice_dir and ensure index contains all
    let mut discovered: Vec<String> = Vec::new();
    if voice_dir.exists() {
        for entry in std::fs::read_dir(voice_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map(|e| e == "npy").unwrap_or(false)
                && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
            {
                discovered.push(stem.to_string());
            }
        }
    }
    for name in &discovered {
        index.speakers.entry(name.clone()).or_default();
    }

    // Build prompts
    let prompt_encoder = PromptEncoder::new(tokenizer, device, num_codebooks, model_type);
    let mut speakers = HashMap::new();
    let mut default_prompt: Option<Tensor> = None;

    for (name, prompt_text) in &index.speakers {
        let npy_path = voice_dir.join(format!("{}.npy", name));
        if !npy_path.exists() {
            continue; // skip missing file
        }
        let prompt_tensor = load_prompt_text(&npy_path, device, num_codebooks)?;
        let prompt = prompt_encoder.encode_conditioning_prompt(prompt_text, &prompt_tensor)?;
        if name == "default" {
            default_prompt = Some(prompt.clone());
        }
        speakers.insert(name.clone(), prompt);
    }

    // If no default present, choose any available as default
    let default_prompt = match default_prompt {
        Some(p) => p,
        None => {
            let any_key = speakers
                .keys()
                .next()
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("No speakers found in voices directory"))?;
            speakers.get(&any_key).unwrap().clone()
        }
    };

    // Write back index.json to include discovered entries
    if (!index_path.exists() || !discovered.is_empty())
        && let Ok(file) = std::fs::File::create(&index_path)
    {
        let _ = serde_json::to_writer_pretty(file, &index);
    }

    Ok((speakers, default_prompt))
}
