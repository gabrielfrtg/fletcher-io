# AMD hipCOMP-core backend

The `original/HIP` backend provides a HIP-portable implementation of Fletcher's
propagation kernels together with checkpoint compression powered by AMD's
[`hipCOMP-core`](https://github.com/ROCm/hipCOMP-core) library.  The port mirrors
the CUDA code path while replacing CUDA runtime calls with their HIP equivalents
and re-implementing the nvCOMP-specific compression manager on top of
hipCOMP's batched LZ4 API.

## Layout

* Device code lives alongside the original CUDA sources, but the GPU toolchain
  is driven by `hipcc`.  See [`original/HIP/Makefile`](../original/HIP/Makefile)
  and [`original/HIP/flags.mk`](../original/HIP/flags.mk) for the build rules and
  compiler flags.
* `cuda_defines.h` now wraps HIP runtime error handling so existing device code
  can keep the `CUDA_CALL()` macro. 【F:original/HIP/cuda_defines.h†L13-L22】
* The compression pipeline is implemented in
  [`cuda_compression.cu`](../original/HIP/cuda_compression.cu).  When
  `USE_HIPCOMP=1` is passed, the file is compiled into the executable; otherwise
  it contributes no symbols.

## Compression format

hipCOMP only exposes the nvCOMP 2.2 batched API.  The port therefore manages
chunk metadata explicitly and stores a small header in front of every
compressed payload:

```
struct HipcompHeader {
  uint32_t magic;        // 'HLZ4'
  uint16_t version;      // currently 1
  uint32_t chunk_size;   // bytes per uncompressed chunk
  uint32_t num_chunks;   // number of chunks following the header
  uint64_t uncompressed_bytes;
  uint64_t total_compressed_bytes;
};
```

The header is followed by an array of `num_chunks` 64-bit compressed chunk
lengths and then the contiguous chunk data.  The compressor grows or shrinks
its staging buffers on demand to accommodate the largest chunk count seen so
far. 【F:original/HIP/cuda_compression.cu†L68-L153】【F:original/HIP/cuda_compression.cu†L207-L296】

During compression the backend:

1. Partitions the wavefield into `chunk_size` pieces and stages pointer/size
   arrays on the host.
2. Copies those arrays to device memory and launches
   `hipcompBatchedLZ4CompressAsync()`.
3. Copies the per-chunk size results back to the host and materialises the
   header + payload layout described above. 【F:original/HIP/cuda_compression.cu†L228-L286】

Decompression reverses the process by parsing the header on the host, ensuring
the staging buffers match the recorded chunk parameters, and invoking
`hipcompBatchedLZ4DecompressAsync()` with rebuilt pointer arrays.
【F:original/HIP/cuda_compression.cu†L298-L378】

## Build flags

Set `backend=HIP` when invoking `make` inside the `original/` directory.  Pass
`USE_HIPCOMP=1` to enable compression and point `HIPCOMP_ROOT` at an installed
hipCOMP-core distribution, e.g.:

```bash
make clean
make -j backend=HIP USE_HIPCOMP=1 HIPCOMP_ROOT=/opt/rocm/hipcomp-core
```

`flags.mk` wires `HIPCOMP_ROOT` into both the host and device compiler include
paths and searches common install locations (e.g. `lib`, `lib64`, `build/lib`)
for `libhipcomp.{so,a}`.  Override `HIPCOMP_LIB_DIR` if your build stores the
library elsewhere. 【F:original/HIP/flags.mk†L1-L30】
