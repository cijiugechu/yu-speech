# syntax=docker/dockerfile:1.6

FROM messense/macos-cross-toolchain:latest AS build

ARG RUST_VERSION=1.79.0
ARG MACOS_TARGET=aarch64-apple-darwin
ARG SDK_PATH=/opt/osxcross/SDK/MacOSX14.0.sdk
ENV DEBIAN_FRONTEND=noninteractive \
    MACOSX_DEPLOYMENT_TARGET=13.0 \
    CC_aarch64_apple_darwin=aarch64-apple-darwin-clang \
    CXX_aarch64_apple_darwin=aarch64-apple-darwin-clang++ \
    AR_aarch64_apple_darwin=aarch64-apple-darwin-ar \
    CARGO_TARGET_AARCH64_APPLE_DARWIN_LINKER=aarch64-apple-darwin-clang \
    SDKROOT=${SDK_PATH} \
    BINDGEN_EXTRA_CLANG_ARGS_aarch64_apple_darwin="--sysroot=${SDK_PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        clang \
        cmake \
        curl \
        git \
        libssl-dev \
        pkg-config \
        unzip \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain ${RUST_VERSION}
ENV PATH="/root/.cargo/bin:${PATH}"

RUN rustup target add ${MACOS_TARGET}

WORKDIR /workspace

COPY Cargo.toml Cargo.lock ./
COPY candle-gqa-kernels ./candle-gqa-kernels
COPY fish_speech_core ./fish_speech_core
COPY fish_speech_python ./fish_speech_python
COPY server ./server
COPY configs ./configs
COPY voices-template ./voices-template

RUN cargo fetch --locked
RUN cargo build --release --locked --target ${MACOS_TARGET} --bin server --features metal
RUN ${MACOS_TARGET}-strip target/${MACOS_TARGET}/release/server || true

FROM scratch AS artifact

ARG MACOS_TARGET=aarch64-apple-darwin

COPY --from=build /workspace/target/${MACOS_TARGET}/release/server /server
COPY --from=build /workspace/configs /configs
COPY --from=build /workspace/voices-template /voices
