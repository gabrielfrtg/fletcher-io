# fletcher-io


## GPU Compression Support

### NVIDIA nvCOMP (CUDA)

### Build
```bash
make clean ; make -j USE_NVCOMP=1
```

### Execute with file decompression
```bash
DECOMPRESS_FILE=1 ./ModelagemFletcher.exe TTI 312 312 312 16 12.5 12.5 12.5 0.001 0.1
```

#### Compressed file
```bash
checkpoints_compressed.bin
```

#### Decompressed files
```bash
checkpoints_decompressed.rsf
checkpoints_decompressed.rsf@
```

### AMD ROCm hipCOMP-core (HIP)

The `original/HIP` backend provides a HIP port that layers hipCOMP-core onto the
existing propagation kernels.  Build it by selecting the HIP backend and
pointing the build to an installed hipCOMP-core tree:

```bash
make clean
# Override ROCM_PATH if hipconfig cannot detect your installation automatically.
ROCM_PATH=/opt/rocm-7.0.1 \
  make -j backend=HIP USE_HIPCOMP=1 HIPCOMP_ROOT=/opt/rocm/hipcomp-core
```

See [`docs/hip_backend.md`](docs/hip_backend.md) for an architectural overview
of the new compression pipeline and metadata format.
