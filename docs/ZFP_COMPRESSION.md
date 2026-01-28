# ZFP Compression Support for Fletcher-IO

## Overview

This document describes the ZFP compression integration for the HIP_OPT backend in Fletcher-IO, enabling GPU-accelerated lossy compression for seismic wave simulation checkpoints.

## Features

- **GPU-native ZFP compression** via hipCOMP library
- **Configurable compression rate** for quality vs. size trade-off
- **Error validation mode** for scientific validation
- **Compile-time algorithm selection** (LZ4 or ZFP)

## Build Options

### Basic ZFP Build
```bash
make backend=HIP_OPT USE_HIPCOMP=1 COMP_ALGO=ZFP HIPCOMP_ROOT=/path/to/hipcomp/install
```

### With Custom Rate
```bash
# Rate = bits per value (lower = more compression, more loss)
make backend=HIP_OPT USE_HIPCOMP=1 COMP_ALGO=ZFP ZFP_RATE=16.0 HIPCOMP_ROOT=/path/to/hipcomp
```

### LZ4 (Lossless) Build
```bash
make backend=HIP_OPT USE_HIPCOMP=1 COMP_ALGO=LZ4 HIPCOMP_ROOT=/path/to/hipcomp
```

## ZFP Rate Reference

| ZFP_RATE | Compression | Error Level | Use Case |
|----------|-------------|-------------|----------|
| 32.0 | ~1x | ~10⁻⁷ | Near-lossless |
| 24.0 | ~1.33x | ~10⁻⁵ | High quality (default) |
| 16.0 | ~2x | ~10⁻³ | Balanced |
| 8.0 | ~4x | ~10⁻¹ | High compression |

**Note**: Default rate is 24.0 bits/value for scientific accuracy.

## GPU Limitation

**Important**: ZFP on GPU only supports **FIXED_RATE mode** (lossy).

Lossless (REVERSIBLE) mode is NOT supported on GPU. For lossless compression, use:
```bash
make backend=HIP_OPT USE_HIPCOMP=1 COMP_ALGO=LZ4 HIPCOMP_ROOT=/path/to/hipcomp
```

Reference: [ZFP CUDA Limitations](https://zfp.readthedocs.io/en/release1.0.1/execution.html#cuda-limitations)

## Error Validation

To validate compression quality, enable the validation mode:

```bash
# Run with validation enabled
ZFP_VALIDATE=1 ./ModelagemFletcher.exe TTI 216 216 216 16 12.5 12.5 12.5 0.001 0.2 64 1 4
```

This will output error statistics at the end:
```
[ZFP] Compression Error Statistics (rate=16.0 bits/value):
      Max Absolute Error: 1.657754e-07
      Max Relative Error: 1.067210e-01
      RMSE:               6.620990e-10
      PSNR:               160.53 dB
      Samples validated:  16777216
```

### Interpreting Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| **Max Absolute Error** | Largest difference between original and decompressed | < 10⁻⁵ |
| **Max Relative Error** | Largest relative difference | < 10⁻³ |
| **RMSE** | Root Mean Square Error | < 10⁻⁶ |
| **PSNR** | Peak Signal-to-Noise Ratio (dB) | > 60 dB |

**Note**: Validation mode adds overhead (decompress + compare). Use only for benchmarking.

## Algorithm Comparison

| Algorithm | Type | GPU | Ratio | Error | Best For |
|-----------|------|-----|-------|-------|----------|
| LZ4 | Lossless | ✅ | ~1.5-3x | 0 | General data |
| ZFP | Lossy | ✅ | 1.3-8x | Configurable | Floating-point scientific data |

## Scientific Justification

For seismic wave simulations, numerical errors from finite difference schemes are typically O(h²) to O(h⁴), resulting in errors of 10⁻³ to 10⁻⁵.

With `ZFP_RATE=24.0`:
- Compression error: ~10⁻⁵
- **Smaller than numerical scheme error**
- Scientifically valid for most applications

With `ZFP_RATE=16.0`:
- Compression error: ~10⁻³
- 2x compression ratio
- Still acceptable for many applications

## Files Modified

- `original/HIP_OPT/flags.mk` - Build configuration
- `original/HIP_OPT/hip_compression.c` - ZFP implementation
- `original/HIP_OPT/hip_compression.h` - API declarations

## Dependencies

- hipCOMP library with ZFP support
- ROCm/HIP runtime
- ZFP sources (bundled in hipCOMP)
