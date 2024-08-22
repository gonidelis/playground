#include "nvbench/launch.cuh"
#include "nvbench_helper.cuh"
#include "thrust/detail/raw_pointer_cast.h"
#include <cuda/std/__functional/invoke.h>
#include <cuda_runtime.h>
#include <nvbench/nvbench.cuh>
#include <thrust/async/scan.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template <typename IterBegin, typename IterEnd, typename Pred>
__global__ void find_if(IterBegin begin, IterEnd end, Pred pred, int *result) {

  extern __shared__ int sresult[];
  sresult[0] = INT_MAX;
  __syncthreads();

  auto global_index = threadIdx.x + blockIdx.x * blockDim.x;
  int total_threads = gridDim.x * blockDim.x;

  // traverse the sequence
  for (auto index = global_index; begin + index < end; index += total_threads) {

    // Only one thread reads atomically and propagates it to the
    // the rest threads of the block through shared memory
    if (threadIdx.x == 0) {
      sresult[0] = atomicAdd(result, 0);
    }
    __syncthreads();

    if (sresult[0] < index) { // @georgii early exit!!!
      // printf("early exit!!!");
      return; // this returns the whole block
    }

    if (pred(*(begin + index))) {
      atomicMin(result,
                index); // @georgii atomic min per your request makes sense
      // printf("%d\n", *result);
      return;
    }
  }
}

template <typename ValueType, typename OutputIteratorT>
__global__ void
write_final_result_in_output_iterator_already(ValueType *d_temp_storage,
                                              OutputIteratorT d_out) {
  ValueType *temp_storage = static_cast<ValueType *>(d_temp_storage);
  *d_out = *temp_storage;
}

template <typename ValueType, typename NumItemsT>
__global__ void cuda_mem_set_async_dtemp_storage(ValueType *d_temp_storage,
                                                 NumItemsT num_items) {
  *d_temp_storage = num_items;
}

template <typename T> struct equals_100 {
  __device__ bool operator()(T i) {
    return i == 1;
  } // @amd you 'll never find out the secret sauce
};

template <typename InputIteratorT, typename OutputIteratorT, typename NumItemsT>
void cub_Device_FindIf(void *d_temp_storage, size_t &temp_storage_bytes,
                       InputIteratorT d_in, OutputIteratorT d_out,
                       NumItemsT num_items, cudaStream_t stream = 0) {

  int block_threads = 128;
  // first cub API call
  if (d_temp_storage == nullptr) {
    temp_storage_bytes = sizeof(int);
    return;
  }
  int *int_temp_storage = static_cast<int *>(d_temp_storage);

  // Get device ordinal
  int device_ordinal;
  cudaError error = CubDebug(cudaGetDevice(&device_ordinal));
  if (cudaSuccess != error) {
    return;
  }

  // Get SM count
  int sm_count;
  error = CubDebug(cudaDeviceGetAttribute(
      &sm_count, cudaDevAttrMultiProcessorCount, device_ordinal));
  if (cudaSuccess != error) {
    return;
  }

  int find_if_sm_occupancy;
  error = CubDebug(cub::MaxSmOccupancy(
      find_if_sm_occupancy,
      find_if<decltype(d_in), decltype(d_in + num_items), equals_100<int>>,
      block_threads));
  if (cudaSuccess != error) {
    return;
  }

  int findif_device_occupancy = find_if_sm_occupancy * sm_count;

  // Even-share work distribution
  int max_blocks = findif_device_occupancy * CUB_SUBSCRIPTION_FACTOR(0);

  // use d_temp_storage as the intermediate device result
  // to read and write from. Then store the final result in the output iterator.
  cuda_mem_set_async_dtemp_storage<<<1, 1>>>(int_temp_storage, num_items);

  find_if<<<max_blocks, block_threads, 0, stream>>>(
      d_in, d_in + num_items, equals_100<int>{}, int_temp_storage);

  write_final_result_in_output_iterator_already<int>
      <<<1, 1>>>(int_temp_storage, d_out);
}

//////////////////////////////////////////////////////
void giannis_find_if(nvbench::state &state) {
  const int N = state.get_int64("Elements");
  thrust::device_vector<int> dinput(N, 0);
  dinput[0] = 1;
  thrust::device_vector<int> d_result(1);

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes{};

  cub_Device_FindIf(d_temp_storage, temp_storage_bytes, dinput.begin(),
                    d_result.begin(), dinput.size(), 0);

  thrust::device_vector<uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  state.exec(nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch &launch) {
               cub_Device_FindIf(d_temp_storage, temp_storage_bytes,
                                 dinput.begin(), d_result.begin(),
                                 dinput.size(), launch.get_stream());
             });
}
NVBENCH_BENCH(giannis_find_if)
    .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4))
    .set_timeout(1); // Limit to one second per measurement.

//////////////////////////////////////////////////////
void thrust_find_if(nvbench::state &state) {
  const int N = state.get_int64("Elements");
  thrust::device_vector<int> dinput(N, 0);
  dinput[0] = 1;

  caching_allocator_t alloc;
  thrust::find_if(policy(alloc), dinput.begin(), dinput.end(),
                  equals_100<int>{});

  state.exec(nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch &launch) {
               thrust::find_if(policy(alloc, launch), dinput.begin(),
                               dinput.end(), equals_100<int>{});
             });
}
NVBENCH_BENCH(thrust_find_if)
    .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4))
    .set_timeout(1); // Limit to one second per measurement.

//////////////////////////////////////////////////////
void thrust_count_if(nvbench::state &state) {
  const int N = state.get_int64("Elements");
  thrust::device_vector<int> dinput(N, 0);
  dinput[0] = 1;

  caching_allocator_t alloc;
  thrust::count_if(policy(alloc), dinput.begin(), dinput.end(),
                   equals_100<int>{});

  state.exec(nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch &launch) {
               thrust::count_if(policy(alloc, launch), dinput.begin(),
                                dinput.end(), equals_100<int>{});
             });
}
NVBENCH_BENCH(thrust_count_if)
    .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4))
    .set_timeout(1); // Limit to one second per measurement.