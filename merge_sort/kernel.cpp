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

void Merge(int *arr, int start, int mid, int end) {
    int n1 = mid - start + 1;
    int n2 = end - mid;

    // Create temporary arrays
    int L[n1], R[n2];

    // Copy data to temporary arrays L[] and R[]
    for (int i = 0; i < n1; i++)
        L[i] = arr[start + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];

    // Merge the temporary arrays back into arr[start..end]
    int i = 0, j = 0, k = start;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    // Copy the remaining elements of L[], if there are any
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    // Copy the remaining elements of R[], if there are any
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}


// Define the MergeSort function
void MergeSort(int *arr, int start, int end) {
    if (start < end) {
        // Find the middle point
        int mid = start + (end - start) / 2;

        // Can still optimize the performance by parallelize mergesort on tww different threads
	// by using one Tile sorts first half while another tile sorts second half
	
            MergeSort(arr, start, mid);
            MergeSort(arr, mid + 1, end);
   
        // Merge the sorted thread together
        Merge(arr, start, mid, end);
    }
}

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
  	      
      //Perform sorting input array A 16 elements at a time	  
      MergeSort(myA, i, i + (SIZE -1));
    //------CASE 2---------
    // Load sorted array A into array B in original way, which consists lots of overhead stalling
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
