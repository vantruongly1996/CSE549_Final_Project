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

#define SIZE 16

  bsg_barrier_hw_tile_group_init();
#ifdef WARM_CACHE
  warmup(A, B, N);
#endif
  bsg_barrier_hw_tile_group_sync();
  bsg_cuda_print_stat_kernel_start(); // marks the start of the profiling region.

  int *myA = &A[0];
  int *myB = &B[0];

  for (int i = (__bsg_x<<4)+((__bsg_y & 4)<<6)+((__bsg_y&3)<<9); i < N; i += bsg_tiles_X*bsg_tiles_Y*16) {
  // for (int i = __bsg_id; i < N; i += bsg_tiles_X*bsg_tiles_Y){  	  
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
    //------CASE 1---------
    //Load 16 elements at the same time

    register int tmp00 = myA[i+0];
    register int tmp01 = myA[i+1];
    register int tmp02 = myA[i+2];
    register int tmp03 = myA[i+3];
    register int tmp04 = myA[i+4];
    register int tmp05 = myA[i+5];
    register int tmp06 = myA[i+6];
    register int tmp07 = myA[i+7];
    register int tmp08 = myA[i+8];
    register int tmp09 = myA[i+9];
    register int tmp10 = myA[i+10];
    register int tmp11 = myA[i+11];
    register int tmp12 = myA[i+12];
    register int tmp13 = myA[i+13];
    register int tmp14 = myA[i+14];
    register int tmp15 = myA[i+15];
    asm volatile("": : :"memory");
    myB[i+0] = tmp00;
    myB[i+1] = tmp01;
    myB[i+2] = tmp02;
    myB[i+3] = tmp03;
    myB[i+4] = tmp04;
    myB[i+5] = tmp05;
    myB[i+6] = tmp06;
    myB[i+7] = tmp07;
    myB[i+8] = tmp08;
    myB[i+9] = tmp09;
    myB[i+10] = tmp10;
    myB[i+11] = tmp11;
    myB[i+12] = tmp12;
    myB[i+13] = tmp13;
    myB[i+14] = tmp14;
    myB[i+15] = tmp15;
}  
  bsg_fence();
  bsg_cuda_print_stat_kernel_end(); // marks the end of the profiling region.
  bsg_fence();
  bsg_barrier_hw_tile_group_sync();
  return 0;
}
