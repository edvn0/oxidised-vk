FROM rust:1.91-bookworm

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

# Force Mesa lavapipe (CPU Vulkan)
ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json

WORKDIR /app

# ---- cache-friendly dependency fetch ----
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY build.rs ./

RUN cargo fetch

# ---- full project ----
COPY . .

RUN cargo test -r
