use axum::serve::ListenerExt;
use axum::{
    Router,
    extract::DefaultBodyLimit,
    routing::{get, post},
};
pub use bytes::Bytes;
use candle_core::{DType, Device};
use clap::Parser;
pub use futures_util::Stream;
use server::handlers::{
    encode_speech::encode_speaker, speech::generate_speech, supported_voices::get_supported_voices,
};
use server::state::AppState;
use server::utils::load::{Args, load_codec, load_lm};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Semaphore;
// Re-export the key types
use tower_http::cors::{Any, CorsLayer};
use tracing::info;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stdout)
        .try_init();

    #[cfg(feature = "cuda")]
    let device = Device::cuda_if_available(0)?;

    #[cfg(feature = "metal")]
    let device = Device::new_metal(0)?;

    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let device = Device::Cpu;

    let checkpoint_dir = args.checkpoint.as_ref().map(|raw_dir| raw_dir.canonicalize().unwrap());
    // TODO: Figure out why BF16 is breaking on Metal
    #[cfg(feature = "cuda")]
    let dtype = DType::BF16;
    #[cfg(not(feature = "cuda"))]
    let dtype = DType::F32;

    info!("Loading {:?} model on {:?}", args.fish_version, device);
    let start_load = Instant::now();
    let lm_state = load_lm(&args, checkpoint_dir, dtype, &device)?;
    let (codec_state, sample_rate) =
        load_codec(&args, dtype, &device, lm_state.config.num_codebooks)?;
    let dt = start_load.elapsed();
    info!("Models loaded in {:.2}s", dt.as_secs_f64());

    let state = Arc::new(AppState {
        lm: Arc::new(lm_state),
        codec: Arc::new(codec_state),
        device,
        model_type: args.fish_version,
        sample_rate,
        voice_dir: args.voice_dir.clone(),
        concurrency: Arc::new(Semaphore::new(1)),
    });

    // Create router
    let app = Router::new()
        .route("/v1/audio/speech", post(generate_speech))
        .route("/v1/audio/encoding", post(encode_speaker))
        .route("/v1/voices", get(get_supported_voices))
        .layer(DefaultBodyLimit::max(32 * 1024 * 1024))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .with_state(state);

    // Run server
    let addr = format!("0.0.0.0:{}", args.port);
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .unwrap()
        .tap_io(|tcp_stream| {
            if let Err(err) = tcp_stream.set_nodelay(true) {
                tracing::trace!("failed to set TCP_NODELAY on incoming connection: {err:#}");
            }
        });
    axum::serve(listener, app).await?;

    Ok(())
}
