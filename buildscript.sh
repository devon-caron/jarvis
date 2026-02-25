#!/bin/bash
set -e

LLAMA_LIB=/path/to/llama-go/build/bin

mkdir -p build/bin
CGO_ENABLED=1 CGO_LDFLAGS="-lcublas -lcudart -lcuda -L/path/to/cuda/lib64/ -L/path/to/cuda/lib64/stubs/ -Wl,-rpath,\$ORIGIN" go build -tags cublas -o build/bin/jarvis .

# Symlink llama-go shared libraries so the binary can find them at runtime
for lib in "$LLAMA_LIB"/lib*.so*; do
    ln -sf "$lib" build/bin/
done

echo "Built: build/bin/jarvis"
