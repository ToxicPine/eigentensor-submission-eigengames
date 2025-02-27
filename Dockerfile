FROM debian:bookworm-slim

# Install base dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Rust via rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Foundry
RUN curl -L https://foundry.paradigm.xyz | bash
ENV PATH="/root/.foundry/bin:${PATH}"

RUN foundryup
RUN cargo install cargo-tangle --git https://github.com/tangle-network/gadget --force

# Set working directory
WORKDIR /app

CMD ["bash"]
