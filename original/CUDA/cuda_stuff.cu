#include "cuda_defines.h"
#include "cuda_stuff.h"
#include "cuda_compression.h"

static size_t sxsy=0;

void CUDA_Initialize(const int sx, const int sy, const int sz, const int bord,
	       float dx, float dy, float dz, float dt,
	       float * restrict ch1dxx, float * restrict ch1dyy, float * restrict ch1dzz, 
	       float * restrict ch1dxy, float * restrict ch1dyz, float * restrict ch1dxz, 
	       float * restrict v2px, float * restrict v2pz, float * restrict v2sz, float * restrict v2pn,
	       float * restrict vpz, float * restrict vsv, float * restrict epsilon, float * restrict delta,
	       float * restrict phi, float * restrict theta, 
	       float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc)
{

   extern float* dev_ch1dxx;
   extern float* dev_ch1dyy;
   extern float* dev_ch1dzz;
   extern float* dev_ch1dxy;
   extern float* dev_ch1dyz;
   extern float* dev_ch1dxz;
   extern float* dev_v2px;
   extern float* dev_v2pz;
   extern float* dev_v2sz;
   extern float* dev_v2pn;
   extern float* dev_pp;
   extern float* dev_pc;
   extern float* dev_qp;
   extern float* dev_qc;

 
  int deviceCount;
  CUDA_CALL(cudaGetDeviceCount(&deviceCount));
  const int device=0;
  cudaDeviceProp deviceProp;
  CUDA_CALL(cudaGetDeviceProperties(&deviceProp, device));
  printf("CUDA source using device(%d) %s with compute capability %d.%d.\n", device, deviceProp.name, deviceProp.major, deviceProp.minor);
  CUDA_CALL(cudaSetDevice(device));


  // Check sx,sy values
  if (sx%BSIZE_X != 0)
  {
     printf("sx(%d) must be multiple of BSIZE_X(%d)\n", sx, (int)BSIZE_X);
     exit(1);
  } 
  if (sy%BSIZE_Y != 0)
  {
     printf("sy(%d) must be multiple of BSIZE_Y(%d)\n", sy, (int)BSIZE_Y);
     exit(1);
  } 

   sxsy=sx*sy; // one plan
   const size_t sxsysz=sxsy*sz;
   const size_t msize_vol=sxsysz*sizeof(float);
   const size_t msize_vol_extra=msize_vol+2*sxsy*sizeof(float); // 2 extra plans for wave fields

   CUDA_CALL(cudaMalloc(&dev_ch1dxx, msize_vol));
   CUDA_CALL(cudaMemcpy(dev_ch1dxx, ch1dxx, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMalloc(&dev_ch1dyy, msize_vol));
   CUDA_CALL(cudaMemcpy(dev_ch1dyy, ch1dyy, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMalloc(&dev_ch1dzz, msize_vol));
   CUDA_CALL(cudaMemcpy(dev_ch1dzz, ch1dzz, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMalloc(&dev_ch1dxy, msize_vol));
   CUDA_CALL(cudaMemcpy(dev_ch1dxy, ch1dxy, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMalloc(&dev_ch1dyz, msize_vol));
   CUDA_CALL(cudaMemcpy(dev_ch1dyz, ch1dyz, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMalloc(&dev_ch1dxz, msize_vol));
   CUDA_CALL(cudaMemcpy(dev_ch1dxz, ch1dxz, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMalloc(&dev_v2px, msize_vol));
   CUDA_CALL(cudaMemcpy(dev_v2px, v2px, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMalloc(&dev_v2pz, msize_vol));
   CUDA_CALL(cudaMemcpy(dev_v2pz, v2pz, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMalloc(&dev_v2sz, msize_vol));
   CUDA_CALL(cudaMemcpy(dev_v2sz, v2sz, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMalloc(&dev_v2pn, msize_vol));
   CUDA_CALL(cudaMemcpy(dev_v2pn, v2pn, msize_vol, cudaMemcpyHostToDevice));

   // Wave field arrays with an extra plan
   CUDA_CALL(cudaMalloc(&dev_pp, msize_vol_extra));
   CUDA_CALL(cudaMemset(dev_pp, 0, msize_vol_extra));
   CUDA_CALL(cudaMalloc(&dev_pc, msize_vol_extra));
   CUDA_CALL(cudaMemset(dev_pc, 0, msize_vol_extra));
   CUDA_CALL(cudaMalloc(&dev_qp, msize_vol_extra));
   CUDA_CALL(cudaMemset(dev_qp, 0, msize_vol_extra));
   CUDA_CALL(cudaMalloc(&dev_qc, msize_vol_extra));
   CUDA_CALL(cudaMemset(dev_qc, 0, msize_vol_extra));
   dev_pp+=sxsy;
   dev_pc+=sxsy;
   dev_qp+=sxsy;
   dev_qc+=sxsy;


  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaDeviceSynchronize());
  printf("GPU memory usage = %ld MiB\n", 15*msize_vol/1024/1024);


  const size_t max_uncompressed = ((size_t)sx*sy)*sz * sizeof(float);
  CUDA_InitCompression(max_uncompressed, 0); // 0=fast compression

}


void CUDA_Finalize()
{

   extern float* dev_ch1dxx;
   extern float* dev_ch1dyy;
   extern float* dev_ch1dzz;
   extern float* dev_ch1dxy;
   extern float* dev_ch1dyz;
   extern float* dev_ch1dxz;
   extern float* dev_v2px;
   extern float* dev_v2pz;
   extern float* dev_v2sz;
   extern float* dev_v2pn;
   extern float* dev_pp;
   extern float* dev_pc;
   extern float* dev_qp;
   extern float* dev_qc;

   dev_pp-=sxsy;
   dev_pc-=sxsy;
   dev_qp-=sxsy;
   dev_qc-=sxsy;

   CUDA_CALL(cudaFree(dev_ch1dxx));
   CUDA_CALL(cudaFree(dev_ch1dyy));
   CUDA_CALL(cudaFree(dev_ch1dzz));
   CUDA_CALL(cudaFree(dev_ch1dxy));
   CUDA_CALL(cudaFree(dev_ch1dyz));
   CUDA_CALL(cudaFree(dev_ch1dxz));
   CUDA_CALL(cudaFree(dev_v2px));
   CUDA_CALL(cudaFree(dev_v2pz));
   CUDA_CALL(cudaFree(dev_v2sz));
   CUDA_CALL(cudaFree(dev_v2pn));
   CUDA_CALL(cudaFree(dev_pp));
   CUDA_CALL(cudaFree(dev_pc));
   CUDA_CALL(cudaFree(dev_qp));
   CUDA_CALL(cudaFree(dev_qc));

   CUDA_FinalizeCompression();

   printf("CUDA_Finalize: SUCCESS\n");
}



void CUDA_Update_pointers(const int sx, const int sy, const int sz, float *pc)
{
   extern float* dev_pc;
   const size_t sxsysz=((size_t)sx*sy)*sz;
   const size_t msize_vol=sxsysz*sizeof(float);
   if (pc) CUDA_CALL(cudaMemcpy(pc, dev_pc, msize_vol, cudaMemcpyDeviceToHost));
}



extern "C" void CUDA_Get_compressed_checkpoint(const int sx, const int sy, const int sz,
                                               void** compressed_data, size_t* compressed_size)
{
    extern float* dev_pc;
    const size_t num_elements = ((size_t)sx*sy)*sz;
    
   // Use high-level nvCOMP API wrapper
   *compressed_size = CUDA_CompressWavefield(dev_pc, num_elements, compressed_data);
    
    if (*compressed_size > 0) {
        float compression_ratio = (num_elements * sizeof(float)) / (float)*compressed_size;
        printf("Compressed checkpoint: %.2f MB -> %.2f MB (ratio: %.2fx)\n",
               (num_elements * sizeof(float))/(1024.0*1024.0),
               *compressed_size/(1024.0*1024.0),
               compression_ratio);
    } else {
        printf("Warning: Compression failed, falling back to uncompressed\n");
    }
}

extern "C" int CUDA_Decompress_to_pc(float* host_compressed, size_t compressed_size,
                    const int sx, const int sy, const int sz)
{
   if (compressed_size == 0) return 0;
   extern float* dev_pc;
   // Upload compressed data to device temp buffer
   void* d_comp = nullptr;
   CUDA_CALL(cudaMalloc(&d_comp, compressed_size));
   CUDA_CALL(cudaMemcpy(d_comp, host_compressed, compressed_size, cudaMemcpyHostToDevice));
   const size_t num_elements = ((size_t)sx*sy)*sz;
   int ok = CUDA_DecompressWavefield(d_comp, dev_pc, num_elements);
   CUDA_CALL(cudaFree(d_comp));
   return ok;
}

extern "C" void CUDA_DecompressCheckpointFile(const char* infile,
                           const char* out_header,
                           const char* out_data,
                           int sx, int sy, int sz, int bord, int absorb,
                           float dx, float dy, float dz, float dt_output)
{
   FILE* in = fopen(infile, "rb");
   if (!in) {
      printf("Could not open compressed checkpoint file %s for decompression.\n", infile);
      return;
   }
   FILE* out_bin = fopen(out_data, "wb");
   if (!out_bin) {
      printf("Could not open output data file %s.\n", out_data);
      fclose(in);
      return;
   }
   typedef struct {
      int iteration;
      int nx, ny, nz;
      size_t original_size;
      size_t compressed_size;
      float timestamp;
   } CheckpointHeader;

   int snapshot_count = 0;
   float first_time = 0.0f, second_time = 0.0f;
   while (1) {
      CheckpointHeader header;
      size_t r = fread(&header, sizeof(header), 1, in);
      if (r != 1) break; // EOF
      if (snapshot_count == 0) first_time = header.timestamp; else if (snapshot_count == 1) second_time = header.timestamp;
      if (header.compressed_size == 0 || header.compressed_size > (1ULL<<40)) {
         printf("Invalid compressed_size in header, aborting decompression loop.\n");
         break;
      }
      void* comp_buf = malloc(header.compressed_size);
      if (!comp_buf) { printf("Alloc fail for compressed buffer.\n"); break; }
      if (fread(comp_buf, 1, header.compressed_size, in) != header.compressed_size) {
         printf("Short read on compressed data.\n");
         free(comp_buf);
         break;
      }
      size_t num_floats = header.original_size / sizeof(float);
      float* d_out = NULL;
      CUDA_CALL(cudaMalloc(&d_out, header.original_size));
      void* d_comp = NULL;
      CUDA_CALL(cudaMalloc(&d_comp, header.compressed_size));
      CUDA_CALL(cudaMemcpy(d_comp, comp_buf, header.compressed_size, cudaMemcpyHostToDevice));
      int ok = CUDA_DecompressWavefield(d_comp, d_out, num_floats);
      CUDA_CALL(cudaFree(d_comp));
      if (!ok) {
         printf("Decompression failed for iteration %d.\n", header.iteration);
         CUDA_CALL(cudaFree(d_out));
         free(comp_buf);
         break;
      }
      float* h_out = (float*)malloc(header.original_size);
      if (!h_out) { printf("Host alloc fail for decompressed output.\n"); }
      else {
         CUDA_CALL(cudaMemcpy(h_out, d_out, header.original_size, cudaMemcpyDeviceToHost));
         fwrite(h_out, 1, header.original_size, out_bin);
         free(h_out);
         snapshot_count++;
      }
      CUDA_CALL(cudaFree(d_out));
      free(comp_buf);
   }
   fclose(in);
   fclose(out_bin);

   // Write RSF header similar to CloseSliceFile FULL
   FILE* out_hdr = fopen(out_header, "w");
   if (!out_hdr) {
      printf("Could not open header file %s for writing.\n", out_header);
      return;
   }
   const int nx_full = sx - 2*bord - 2*absorb;
   const int ny_full = sy - 2*bord - 2*absorb;
   const int nz_full = sz - 2*bord - 2*absorb;
   float inferred_dt = dt_output;
   if (snapshot_count > 1 && second_time > first_time) {
      inferred_dt = second_time - first_time; // time between snapshots
   }
   fprintf(out_hdr, "in=\"%s\"\n", out_data);
   fprintf(out_hdr, "data_format=\"native_float\"\n");
   fprintf(out_hdr, "esize=%lu\n", sizeof(float));
   fprintf(out_hdr, "n1=%d\n", nx_full);
   fprintf(out_hdr, "n2=%d\n", ny_full);
   fprintf(out_hdr, "n3=%d\n", nz_full);
   fprintf(out_hdr, "n4=%d\n", snapshot_count);
   fprintf(out_hdr, "d1=%f\n", dx);
   fprintf(out_hdr, "d2=%f\n", dy);
   fprintf(out_hdr, "d3=%f\n", dz);
   fprintf(out_hdr, "d4=%f\n", inferred_dt);
   fclose(out_hdr);
   printf("File-level decompression complete: %d snapshots -> %s (%s).\n", snapshot_count, out_data, out_header);
}


