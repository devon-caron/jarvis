# Jarvis

A Go-based assistant powered by [llama-go](https://github.com/tcpipuk/llama-go) with CUDA GPU acceleration.

## Prerequisites

- linux (Tested on Ubuntu 22.04 Pop_OS!)
- Go (with CGO support)
- NVIDIA CUDA toolkit (specifically `libcublas` and `libcudart`)
- [llama-go](https://github.com/tcpipuk/llama-go) built from source

## Building

The repository includes a `buildscript.sh` template. Copy it to `build.sh` and edit the placeholder paths to match your system:

```bash
cp buildscript.sh build.sh
```

Open `build.sh` and update the following:

### 1. `LLAMA_LIB` — Path to llama-go shared libraries

```bash
LLAMA_LIB=/path/to/llama-go/build/bin
```

Set this to the directory containing the `lib*.so` files produced by building llama-go. For example:

```bash
LLAMA_LIB=/home/youruser/Repos/llama-go/build/bin
```

### 2. CUDA library path in `CGO_LDFLAGS`

```bash
CGO_LDFLAGS="-lcublas -lcudart -L/path/to/cuda/lib64/ -Wl,-rpath,$ORIGIN"
```

Replace `/path/to/cuda/lib64/` with the location of your CUDA libraries. Common locations:

- `/usr/local/cuda/lib64/` (default CUDA toolkit install)
- `/usr/lib/x86_64-linux-gnu/` (some distro packages)

### 3. Build

```bash
chmod +x build.sh
./build.sh
```

The compiled binary and required shared libraries will be placed in `build/bin/`.

## Running

After building, start Jarvis with:

```bash
LD_LIBRARY_PATH=/path/to/llama-go/build/bin CUDA_VISIBLE_DEVICES=0,1 ./build/bin/jarvis start
```

Adjust `LD_LIBRARY_PATH` to match your `LLAMA_LIB` path, and `CUDA_VISIBLE_DEVICES` to the GPUs you want to use.

## License

MIT
