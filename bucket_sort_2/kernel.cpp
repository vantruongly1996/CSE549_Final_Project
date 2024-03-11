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


  const int TILE_SIZE = bsg_tiles_X * bsg_tiles_Y * 16;
  const int NUM_BUCKETS = 4; // Number of buckets for 16 elements
  const int BUCKET_SIZE = SIZE / NUM_BUCKETS;
 
// Initialize buckets
    int buckets[NUM_BUCKETS][BUCKET_SIZE];
    int bucket_sizes[NUM_BUCKETS] = {0};

  for (int i = (__bsg_x<<4)+((__bsg_y & 4)<<6)+((__bsg_y&3)<<9); i < N; i += TILE_SIZE) {
	  for (int j = i; j < i + SIZE; ++j) {
		  int bucket_index = myA[j] / BUCKET_SIZE;
		  if (bucket_sizes[bucket_index] < BUCKET_SIZE) {
			  buckets[bucket_index][bucket_sizes[bucket_index]++] = myA[j];
		  } else {
			  // Bucket overflow, handle if needed
			 }
	  }
    // Sort buckets using insertion sort
   for (int i = 0; i < NUM_BUCKETS; ++i) {
	   // Use insertion sort for sorting each bucket
           for (int j = 1; j < bucket_sizes[i]; ++j) {
           int key = buckets[i][j];
           int k = j - 1;
           while (k >= 0 && buckets[i][k] > key) {
		   buckets[i][k + 1] = buckets[i][k];
		   k--;
	   }
	   buckets[i][k + 1] = key;
	   }
   }

// Merge sorted buckets into the output array
    int index = 0;
    for (int i = 0; i < NUM_BUCKETS; ++i) {
        for (int j = 0; j < bucket_sizes[i]; ++j) {
            myB[index++] = buckets[i][j];
        }
    }
}  
  bsg_fence();
  bsg_cuda_print_stat_kernel_end(); // marks the end of the profiling region.
  bsg_fence();
  bsg_barrier_hw_tile_group_sync();
  return 0;
}
