# ------------------------------------------------------------
# Base image with system dependencies (shared & cached)
# ------------------------------------------------------------
FROM rust:1.91-bookworm AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV CARGO_TERM_COLOR=always
ENV RUST_BACKTRACE=1

RUN apt-get update && apt-get install -y \
    lua5.4 \
    pkg-config \
    cmake \
    clang \
    libvulkan1 \
    vulkan-validationlayers \
    vulkan-tools \
    mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*

ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json

WORKDIR /app


# ------------------------------------------------------------
# Dependency build stage (maximally cacheable)
# ------------------------------------------------------------
FROM base AS dependencies

COPY Cargo.toml Cargo.lock ./

RUN mkdir src && \
    echo "fn main() {}" > src/main.rs

RUN cargo build --release

RUN rm -rf src


# ------------------------------------------------------------
# Application build + test stage
# ------------------------------------------------------------
FROM base AS application

COPY --from=dependencies /usr/local/cargo /usr/local/cargo
COPY --from=dependencies /app/target /app/target

COPY Cargo.toml Cargo.lock ./
COPY build.rs ./
COPY src ./src
COPY assets ./assets
COPY scripts ./scripts

RUN cargo test --release


# ------------------------------------------------------------
# (Optional) runtime image if you later need one
# ------------------------------------------------------------
# FROM debian:bookworm-slim AS runtime
# COPY --from=application /app/target/release/your_binary /usr/local/bin/app
# ENTRYPOINT ["app"]
