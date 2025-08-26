# fletcher-io


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
