use super::error::AppError;
use crate::state::AppState;
use axum::{
    extract::{Multipart, Query, State},
    http::StatusCode,
    response::Response,
};
use candle_core::{D, Tensor};
use fish_speech_core::audio as torchaudio;
use fish_speech_core::audio::functional;
use fish_speech_core::text::prompt::PromptEncoder;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tracing::info;

use std::io::{Read, Seek};
use tempfile::NamedTempFile;

pub fn tensor_to_npy_bytes(tensor: &Tensor) -> anyhow::Result<Vec<u8>> {
    // This gives us both a Path for write_npy AND Seek for rewind
    let mut temp = NamedTempFile::new()?;

    // Now we can use write_npy with the path
    tensor.write_npy(temp.path())?;

    // And we can seek because NamedTempFile implements Seek
    temp.rewind()?;

    // Read it back
    let mut bytes = Vec::new();
    temp.read_to_end(&mut bytes)?;

    Ok(bytes)
}

pub async fn encode_speaker(
    State(state): State<Arc<AppState>>,
    Query(params): Query<std::collections::HashMap<String, String>>,
    mut multipart: Multipart,
) -> Result<Response, AppError> {
    let start_total = Instant::now();

    let field = multipart
        .next_field()
        .await?
        .ok_or_else(|| anyhow::anyhow!("No file provided"))?;

    let data = field.bytes().await?.to_vec();

    let (mut audio, sr) = torchaudio::load_from_memory(data, &state.device)?;
    if audio.dim(0)? > 1 {
        audio = audio.mean_keepdim(0)?;
    }
    let audio = functional::resample(&audio, sr, state.sample_rate)?;
    // TODO handle batched audio
    let result = state
        .codec
        .encode_batch(&audio.unsqueeze(0)?)
        .await?
        .squeeze(0)?;

    let start_encode = Instant::now();
    let encode_time = start_encode.elapsed().as_secs_f32();
    // Persist to voices dir if an id is provided (prompt is optional; empty string if missing)
    if let Some(id) = params.get("id") {
        let prompt_text = params.get("prompt").cloned().unwrap_or_default();

        // Write tokens as U32 .npy to voice_dir/id.npy
        let tokens_u32 = result.to_dtype(candle_core::DType::U32)?;
        let voice_dir: PathBuf = state.voice_dir.clone();
        fs::create_dir_all(&voice_dir)?;
        let npy_path = voice_dir.join(format!("{}.npy", id));
        // Do not overwrite existing file to avoid accidental clobbering
        if npy_path.exists() {
            return Err(AppError::Message(format!(
                "Voice '{}' already exists on disk at {}",
                id,
                npy_path.display()
            )));
        }
        tokens_u32.write_npy(&npy_path)?;

        // Update or create index.json with the prompt text
        #[derive(Serialize, Deserialize, Default)]
        struct SpeakerIndex {
            speakers: std::collections::HashMap<String, String>,
        }

        let index_path = voice_dir.join("index.json");
        let mut index: SpeakerIndex = if index_path.exists() {
            let file = fs::File::open(&index_path)?;
            serde_json::from_reader(file).unwrap_or_default()
        } else {
            SpeakerIndex::default()
        };
        index.speakers.insert(id.clone(), prompt_text.clone());
        let file = fs::File::create(&index_path)?;
        serde_json::to_writer_pretty(file, &index)?;

        // Also populate in-memory map for immediate use (without restart)
        {
            let mut speaker_map = state.lm.voices.write().await;
            if speaker_map.contains_key(id) {
                return Err(AppError::Message(format!(
                    "ID already exists on server: {}",
                    id
                )));
            }
            let prompt_encoder = PromptEncoder::new(
                &state.lm.tokenizer,
                &state.device,
                state.lm.config.num_codebooks,
                state.lm.model_type,
            );
            let new_prompt =
                prompt_encoder.encode_conditioning_prompt(&prompt_text, &tokens_u32)?;
            speaker_map.insert(id.to_owned(), new_prompt);
        }
    }

    // Return the U32 token npy bytes
    let npy_bytes = tensor_to_npy_bytes(&result.to_dtype(candle_core::DType::U32)?)?;

    let audio_duration = audio.dim(D::Minus1)? as f32 / state.sample_rate as f32;
    info!("Encoding RTF: {:.1}x", audio_duration / encode_time);
    info!(
        "Total RTF: {:.1}x",
        audio_duration / start_total.elapsed().as_secs_f32()
    );

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/x-npy")
        .body(npy_bytes.into())?)
}
