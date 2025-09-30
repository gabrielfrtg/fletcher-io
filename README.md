# Fletcher-IO

This is a repository dedicated to IO optimizations in Fletcher simulations.

For new repositories with implementations/optimizations other than IO, please use the base repository for Fletcher (https://github.com/gabrielfrtg/fletcher-base).

## Some Warnings

* If/When merging the branch feature/nvcomp-lz4-compression into main, be careful with some conflicts that will certainly arise. Resolve conflicts with care and attention.

## NVIDIA NVCOMP Support

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
