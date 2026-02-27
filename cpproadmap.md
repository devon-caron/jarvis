  1. Modern C++ (1-2 weeks)                                                                                                        
  You know basic C++. Bridge the gap to C++17 which ggml/llama.cpp uses:
  - RAII and smart pointers (unique_ptr, shared_ptr) — analogous to Go's defer/GC                                                  
  - Templates — similar to Java generics but compile-time                                                                        
  - Move semantics — no equivalent in Go/Java, critical for performance code
  - std::vector, std::span, std::string_view — the standard containers you'll see everywhere

  2. Memory layout and pointer arithmetic (a few days)
  Go and Java hide this from you. In C++ (and especially CUDA), you need to understand:
  - Stack vs heap, manual allocation (malloc/free, new/delete)
  - Pointer arithmetic, stride, and offset calculations into buffers
  - How a 2D/3D tensor is laid out as a flat 1D array (row-major vs column-major)
  - Memory alignment requirements

  3. CUDA fundamentals (1-2 weeks)
  This is the core of the bug. You need to understand:
  - GPU execution model: grids, blocks, threads — how a kernel launch maps to hardware
  - Device memory vs host memory — cudaMalloc, cudaMemcpy, cudaFree
  - Streams and synchronization — cudaStreamSynchronize is where the crash surfaces
  - Multi-GPU: cudaSetDevice, peer access (cudaDeviceEnablePeerAccess), and how memory on GPU 0 is accessed from GPU 1
  - NVIDIA's CUDA C++ Programming Guide is the canonical resource. Focus on chapters 1-3 and the multi-device section.

  4. ggml's tensor/backend abstraction (1 week of reading)
  ggml is the tensor library under llama.cpp. Read these in order:
  - ggml/include/ggml.h — tensor struct, operations
  - ggml/src/ggml-backend.h — backend interface (CPU, CUDA, etc.)
  - ggml/src/ggml-cuda.cu — the CUDA backend, where the crash is
  - Understand how ggml_backend_buffer allocates memory per-device and how tensors reference offsets into those buffers

  5. llama.cpp's KV cache (a few days of reading)
  This is where the bug lives:
  - src/llama-kv-cache.cpp — how KV cache is allocated, how slots map to buffer regions
  - src/llama-context.cpp — how inference dispatches to the computation graph
  - Understand how -sm graph splits the graph across devices vs -sm layer which splits by layer

  6. Debugging tools
  - CUDA_LAUNCH_BLOCKING=1 — makes kernels synchronous so the crash points to the exact kernel
  - compute-sanitizer (NVIDIA's tool) — CUDA equivalent of valgrind, catches out-of-bounds GPU memory access with the exact address
   and kernel
  - cuda-gdb — GPU debugger, can inspect device memory

  Practical approach: Don't study all of this in isolation. Clone ik_llama.cpp, reproduce the crash, run with
  CUDA_LAUNCH_BLOCKING=1 and compute-sanitizer to identify the exact kernel, then read backward from there. Having a concrete bug
  to chase will make the learning stick faster than studying abstractions first.