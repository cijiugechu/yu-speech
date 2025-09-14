use axum::extract::multipart::MultipartError;
use axum::http::StatusCode;
use axum::response::Response;
use candle_core::Error as CandleError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerdeJson(#[from] serde_json::Error),

    #[error("Zip error: {0}")]
    Zip(#[from] zip::result::ZipError),

    #[error("Candle error: {0}")]
    Candle(#[from] CandleError),

    #[error("Axum error: {0}")]
    Axum(#[from] axum::http::Error),

    #[error("Multipart error: {0}")]
    Multipart(#[from] MultipartError),

    #[error(transparent)]
    Anyhow(#[from] anyhow::Error),

    #[error("Application error: {0}")]
    Message(String),
}

impl axum::response::IntoResponse for AppError {
    fn into_response(self) -> Response {
        // Log the error with its full chain of causes
        tracing::error!("Application error: {self}");

        let (status, kind) = match &self {
            AppError::Io(_) => (StatusCode::INTERNAL_SERVER_ERROR, "io"),
            AppError::SerdeJson(_) => (StatusCode::BAD_REQUEST, "serde_json"),
            AppError::Zip(_) => (StatusCode::INTERNAL_SERVER_ERROR, "zip"),
            AppError::Candle(_) => (StatusCode::INTERNAL_SERVER_ERROR, "candle"),
            AppError::Axum(_) => (StatusCode::INTERNAL_SERVER_ERROR, "axum"),
            AppError::Anyhow(_) => (StatusCode::INTERNAL_SERVER_ERROR, "anyhow"),
            AppError::Message(_) => (StatusCode::INTERNAL_SERVER_ERROR, "message"),
            AppError::Multipart(_) => (StatusCode::INTERNAL_SERVER_ERROR, "multipart"),
        };
        let message = self.to_string();

        (
            status,
            axum::response::Json(serde_json::json!({
                "error": {
                    "kind": kind,
                    "message": message
                }
            })),
        )
            .into_response()
    }
}

impl From<AppError> for std::io::Error {
    fn from(err: AppError) -> Self {
        std::io::Error::other(err.to_string())
    }
}
