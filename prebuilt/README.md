# Prebuilt llama.cpp binaries

Avoid the ~5-minute clean rebuild on session start by using these.

## Provenance

- Source:   <https://github.com/ggml-org/llama.cpp> (upstream, NOT the
  `PrismML-Eng/llama.cpp` fork — Q1_0 PRs #21273, #21528, #21629, #21539,
  #21636 are all merged into upstream).
- Commit:   `1e5ad35d560b90a8ac447d149c8f8447ae1fcaa0`
- `git describe --tags`:  `b9093`
- Build flags (CPU-only, no CUDA / Metal / Vulkan / curl):

      cmake -B build \
            -DBUILD_SHARED_LIBS=OFF \
            -DGGML_CUDA=OFF \
            -DGGML_METAL=OFF \
            -DGGML_VULKAN=OFF \
            -DGGML_LLAMAFILE=OFF \
            -DGGML_OPENMP=ON \
            -DLLAMA_CURL=OFF \
            -DCMAKE_BUILD_TYPE=Release
      cmake --build build -j$(nproc) --target llama-quantize llama-cli llama-gguf

## What's here

| Path                               | Platform        | glibc / runtime               | Strip |
|------------------------------------|-----------------|-------------------------------|-------|
| `linux-x86_64/llama-cli`           | Linux x86_64    | Ubuntu 24.04, glibc 2.39      |  yes  |
| `linux-x86_64/llama-quantize`      | Linux x86_64    | Ubuntu 24.04, glibc 2.39      |  yes  |
| `linux-x86_64/llama-gguf`          | Linux x86_64    | Ubuntu 24.04, glibc 2.39      |  yes  |

Dynamic deps for the Linux build:
`libssl.so.3, libcrypto.so.3, libgomp.so.1, libstdc++.so.6, libm, libc, ld-linux-x86-64`.
All present in the standard Claude Code web sandbox.

### macOS

Not prebuilt (this VM is Linux). On a Mac, run `scripts/build_llama_cpp.sh`;
it produces native arm64 (Apple Silicon) or x86_64 (Intel) binaries with
Metal acceleration enabled by default.

### Other Linux

Should work on any reasonably recent x86_64 Linux with glibc ≥ 2.34.
If your sandbox uses a different libc (e.g. musl), rebuild with
`scripts/build_llama_cpp.sh`.

## Usage

```sh
# Inspect a Bonsai GGUF
./prebuilt/linux-x86_64/llama-gguf models/gguf/Bonsai-1.7B-Q1_0.gguf r

# Quantize an FP16 GGUF down to Q1_0 (round-trip experiment)
./prebuilt/linux-x86_64/llama-quantize input.gguf out-q1.gguf Q1_0

# Run the model
./prebuilt/linux-x86_64/llama-cli -m models/gguf/Bonsai-1.7B-Q1_0.gguf -p "hello"
```
