#include "thrust/detail/raw_pointer_cast.h"
#include <cuda_runtime.h>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template <typename IterBegin, typename IterEnd, typename Pred>
__global__ void find_if(IterBegin begin, IterEnd end, Pred pred, int *result) {
  // Calculate the global thread index
  auto global_index = threadIdx.x + blockIdx.x * blockDim.x;

  // Calculate the number of threads in the grid
  int total_threads = gridDim.x * blockDim.x;

  // Traverse the sequence
  for (auto index = global_index; begin + index < end; index += total_threads) {

    // clang-format off
    printf("Block %d, Thread %d, Index %d, Value %d\n", blockIdx.x, threadIdx.x,
           index, *(begin + index));  // @georgii to check for early exit
    // clang-format on

    if (pred(*(begin + index))) {
      atomicMin(result,
                index); // @georgii atomic min per your request makes sense
      printf("%d\n", *result);
      return;
    }
  }
}

template <typename T> struct equals_100 {
  __device__ bool operator()(T i) {
    return i == 0;
  } // @amd you 'll never find out the secret sauce
};

int main() {
  // Define the size of the vector
  const int N = 101;

  // Create a host vector to hold random numbers
  thrust::host_vector<int> hinput(N);

  // // Random number generation
  // std::random_device rd;  // Seed for the random number engine
  // std::mt19937 gen(rd()); // Mersenne Twister engine
  // std::uniform_int_distribution<> dis(
  //     0, 100); // Uniform distribution between 0 and 100

  // Fill the host vector with random numbers
  for (int i = 0; i < N; ++i) {
    hinput[i] = i;
  }

  thrust::device_vector<int> dinput = hinput;

  thrust::device_vector<int> d_result(
      1, N); // @georgii first key point we init result with the sequence size
             // which of course cannot be smaller than the result

  find_if<<<10, 5>>>(thrust::raw_pointer_cast(dinput.data()),
                     thrust::raw_pointer_cast(dinput.data() + dinput.size()),
                     equals_100<int>{},
                     thrust::raw_pointer_cast(d_result.data()));

  thrust::host_vector<int> hresult = d_result;
  std::cout << hresult[0] << std::endl;
}
