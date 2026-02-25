# Building Jarvis from Source

Verified step-by-step on a fresh `nvidia/cuda:12.4.1-devel-ubuntu22.04` Docker image.

## Prerequisites

- Linux (tested on Ubuntu 22.04)
- NVIDIA GPU with CUDA support
- NVIDIA drivers installed on the host
- ~2 GB disk space for build artifacts

## Step 1: Install System Dependencies

```bash
apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget
```

## Step 2: Install the CUDA Toolkit

If you're not using the `nvidia/cuda` Docker image, install the CUDA toolkit for your
distro from [NVIDIA's site](https://developer.nvidia.com/cuda-downloads). You need the
**devel** variant (headers + `nvcc`), not just the runtime.

Verify the install:

```bash
nvcc --version
```

## Step 3: Install Go

```bash
GO_VERSION=1.24.4
wget -q https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz
tar -C /usr/local -xzf go${GO_VERSION}.linux-amd64.tar.gz
rm go${GO_VERSION}.linux-amd64.tar.gz
export PATH=/usr/local/go/bin:$PATH
```

Verify:

```bash
go version
```

## Step 4: Determine Your GPU's CUDA Architecture

Jarvis compiles CUDA kernels for a specific GPU architecture. You must tell the build
system which architecture to target.

Run this on a machine with your target GPU installed:

```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

This prints the compute capability for each GPU in your system (e.g. `8.6` for an
RTX 3090). If you have multiple identical GPUs, they will all report the same value.

Remove the dot to get the `CUDA_ARCHITECTURES` value: `8.6` becomes `86`.

Common architectures:

| Compute Capability | GPUs | `CUDA_ARCHITECTURES` |
|---------------------|------|----------------------|
| `6.1` | GTX 1080, GTX 1070, GTX 1060 (Pascal) | `61` |
| `7.0` | V100, Titan V (Volta) | `70` |
| `7.5` | RTX 2080, RTX 2070, GTX 1660 (Turing) | `75` |
| `8.0` | A100, A30 (Ampere) | `80` |
| `8.6` | RTX 3090, RTX 3080, RTX 3070, A40 (Ampere) | `86` |
| `8.9` | RTX 4090, RTX 4080, RTX 4070, L40 (Ada Lovelace) | `89` |
| `9.0` | H100, H200 (Hopper) | `90` |
| `10.0` | B200, B100 (Blackwell) | `100` |

If you have multiple GPUs of **different** architectures, pass multiple values separated
by semicolons: `CUDA_ARCHITECTURES="86;89"`. This increases compile time but produces
a binary that works on both.

If you don't have access to the target machine, look up your GPU model in NVIDIA's
[CUDA GPUs list](https://developer.nvidia.com/cuda-gpus).

## Step 5: Clone and Build llama-go

```bash
cd ~
git clone https://github.com/tcpipuk/llama-go.git
cd llama-go
git submodule update --init --recursive
```

Build with CUDA support, substituting your architecture value from Step 4:

```bash
export PATH=/usr/local/cuda/bin:$PATH
BUILD_TYPE=cublas CUDA_ARCHITECTURES=86 make libbinding.a
```

> **Note:** This step compiles ~100 CUDA kernel template files with `nvcc` and takes
> a significant amount of time (15-45+ minutes depending on your CPU). This is normal.

When complete, verify the shared libraries were produced:

```bash
ls build/bin/lib*.so
```

Expected output: `libggml-base.so`, `libggml-cpu.so`, `libggml-cuda.so`, `libggml.so`,
`libllama.so`

## Step 6: Clone and Build Jarvis

```bash
cd ~
git clone https://github.com/devon-caron/jarvis.git
cd jarvis
```

Point `go.mod` at your local llama-go checkout:

```bash
go mod edit -replace github.com/tcpipuk/llama-go=$HOME/llama-go
```

Copy the build script template and configure it:

```bash
cp buildscript.sh build.sh
chmod +x build.sh
```

Edit `build.sh` and update two values:

1. Set `LLAMA_LIB` to your llama-go build output directory:

```bash
LLAMA_LIB=$HOME/llama-go/build/bin
```

2. Set the CUDA library path in `CGO_LDFLAGS`. The linker needs to find `libcublas`,
   `libcudart`, and the CUDA driver stub `libcuda.so`. On a standard CUDA toolkit
   install these are in two directories:

```bash
CGO_LDFLAGS="-lcublas -lcudart -lcuda -L/usr/local/cuda/lib64/ -L/usr/local/cuda/lib64/stubs/ -Wl,-rpath,\$ORIGIN"
```

> **Why two `-L` paths?** The CUDA runtime libraries (`libcublas`, `libcudart`) live in
> `/usr/local/cuda/lib64/`, but the CUDA driver stub (`libcuda.so`) needed at link time
> lives in `/usr/local/cuda/lib64/stubs/`. At runtime, `libcuda.so.1` is provided by
> your NVIDIA driver installation and found automatically.

Build:

```bash
./build.sh
```

Verify:

```bash
ls build/bin/jarvis
```

## Step 7: Run

The compiled binary uses `$ORIGIN` rpath, so it finds the symlinked `.so` files in its
own directory automatically. No `LD_LIBRARY_PATH` needed at runtime on a system with
NVIDIA drivers installed.

```bash
export PATH=$HOME/jarvis/build/bin:$PATH

jarvis start
jarvis models register mymodel -p /path/to/model.gguf
jarvis models load mymodel
jarvis "Hello, world!"
```

## Running Tests

```bash
cd ~/jarvis
LD_LIBRARY_PATH=$HOME/llama-go/build/bin go test ./...
```

## Troubleshooting

### `libcuda.so.1: cannot open shared object file`

This means the NVIDIA driver is not installed on the machine. The CUDA **toolkit** (used
for compilation) does not include the driver. Install the NVIDIA driver for your GPU:

```bash
# Ubuntu
apt-get install nvidia-driver-560  # or latest version for your GPU
```

### `libllama.so.0: cannot open shared object file`

The runtime linker can't find the llama-go shared libraries. Either:
- Run from the `build/bin/` directory (the binary has `$ORIGIN` rpath embedded), or
- Set `LD_LIBRARY_PATH` to include the directory containing the `.so` files

### `undefined reference to cuMemCreate` during jarvis build

The linker can't find the CUDA driver stub. Add the stubs directory to `CGO_LDFLAGS`:

```bash
-L/usr/local/cuda/lib64/stubs/
```

And add `-lcuda` to the flags. See Step 6 for the full flags.

### `nvcc` not found during llama-go build

Ensure the CUDA toolkit `bin/` directory is in your `PATH`:

```bash
export PATH=/usr/local/cuda/bin:$PATH
```

### Build takes extremely long

CUDA kernel compilation is CPU-bound and single-threaded per file. Targeting fewer
architectures reduces compile time. Only specify the architecture(s) you actually need.
