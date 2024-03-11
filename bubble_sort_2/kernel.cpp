#include <bsg_manycore.h>
#include <bsg_cuda_lite_barrier.h>

#ifdef WARM_CACHE
#ifndef CACHE_LINE_WORDS
#error "CACHE_LINE_WORDS not defined"
#endif


__attribute__((noinline))
static void warmup(int *A, int *B, int N)
{
  for (int i = __bsg_id*CACHE_LINE_WORDS; i < N; i += bsg_tiles_X*bsg_tiles_Y*CACHE_LINE_WORDS) {
      asm volatile ("lw x0, %[p]" :: [p] "m" (A[i]));
      asm volatile ("lw x0, %[p]" :: [p] "m" (B[i]));
  }
  bsg_fence();
}
#endif

extern "C" __attribute__ ((noinline))
int
kernel_IS(int * A, int * B, int N) {
const int SIZE = 16;
  bsg_barrier_hw_tile_group_init();
#ifdef WARM_CACHE
  warmup(A, B, N);
#endif
  bsg_barrier_hw_tile_group_sync();
  bsg_cuda_print_stat_kernel_start(); // marks the start of the profiling region.

  int *myA = &A[0];
  int *myB = &B[0];

  for (int i = (__bsg_x<<4)+((__bsg_y & 4)<<6)+((__bsg_y&3)<<9); i < N; i += bsg_tiles_X*bsg_tiles_Y*16) {
  	  
 // Bubble Sort
        for (int j = i; j < i + SIZE - 1; ++j) {
            for (int k = i; k < i + SIZE - j - 1; ++k) {
                if (myA[k] > myA[k + 1]) {
                    // Swap elements
                    int temp = myA[k];
                    myA[k] = myA[k + 1];
                    myA[k + 1] = temp;
                }
            }
	}

    //Store the sorted array myA back to array myB
    for (int j = 0; j < SIZE; j++) {
            myB[i + j] = myA[i + j];
            }
}  
  bsg_fence();
  bsg_cuda_print_stat_kernel_end(); // marks the end of the profiling region.
  bsg_fence();
  bsg_barrier_hw_tile_group_sync();
  return 0;
}
