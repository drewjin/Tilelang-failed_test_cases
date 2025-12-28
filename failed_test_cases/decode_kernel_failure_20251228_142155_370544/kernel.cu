#include <tl_templates/cuda/instruction/mma.h>
#include <math_constants.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void kernel_kernel(const bfloat16_t* __restrict__ K, const bfloat16_t* __restrict__ K_Cache, bfloat16_t* __restrict__ O, const bfloat16_t* __restrict__ Q, const bfloat16_t* __restrict__ V, const bfloat16_t* __restrict__ V_Cache, const int* __restrict__ block_tables, const int* __restrict__ context_lens, const int* __restrict__ cu_seqlens_k, const int* __restrict__ cu_seqlens_q, int MAX_SEQ_NUM_BLOCKS);
extern "C" __global__ void __launch_bounds__(128, 1) kernel_kernel(const bfloat16_t* __restrict__ K, const bfloat16_t* __restrict__ K_Cache, bfloat16_t* __restrict__ O, const bfloat16_t* __restrict__ Q, const bfloat16_t* __restrict__ V, const bfloat16_t* __restrict__ V_Cache, const int* __restrict__ block_tables, const int* __restrict__ context_lens, const int* __restrict__ cu_seqlens_k, const int* __restrict__ cu_seqlens_q, int MAX_SEQ_NUM_BLOCKS) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float acc_output[64];
  float acc_score_kv[32];
  float acc_score_kvcache[16];
  float log_sum[2];
  float scores_max[2];
  float scores_max_prev[2];
  float scores_scale[2];
  float scores_sum[2];
  bfloat16_t acc_score_kvcache_cast[16];
  bfloat16_t A_local[8];
  bfloat16_t B_local[16];
  bfloat16_t B_local_1[64];
  bfloat16_t A_local_1[8];
  bfloat16_t B_local_2[32];
  bfloat16_t acc_score_kv_cast[32];
  bfloat16_t B_local_3[64];
  int q_start_idx = cu_seqlens_q[((int)blockIdx.x)];
  int kv_start_idx = cu_seqlens_k[((int)blockIdx.x)];
  int q_end_idx = cu_seqlens_q[(((int)blockIdx.x) + 1)];
  int kv_end_idx = cu_seqlens_k[(((int)blockIdx.x) + 1)];
  int cur_context_len = context_lens[((int)blockIdx.x)];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint4 condval;
    if (((((((((int)threadIdx.x) >> 4) + q_start_idx) >> 3) + i) < 56) && (0 <= (((i * 8) + (((int)threadIdx.x) >> 4)) + q_start_idx)))) {
      condval = *(uint4*)(Q + (((((((int64_t)i) * (int64_t)28672) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)3584)) + (((int64_t)q_start_idx) * (int64_t)3584)) + (((int64_t)((int)blockIdx.y)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
    } else {
      condval = make_uint4(__pack_nv_bfloat162(bfloat16_t(0x0p+0f/*0.000000e+00*/), bfloat16_t(0x0p+0f/*0.000000e+00*/)), __pack_nv_bfloat162(bfloat16_t(0x0p+0f/*0.000000e+00*/), bfloat16_t(0x0p+0f/*0.000000e+00*/)), __pack_nv_bfloat162(bfloat16_t(0x0p+0f/*0.000000e+00*/), bfloat16_t(0x0p+0f/*0.000000e+00*/)), __pack_nv_bfloat162(bfloat16_t(0x0p+0f/*0.000000e+00*/), bfloat16_t(0x0p+0f/*0.000000e+00*/)));
    }
    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval;
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 32; ++i_1) {
    *(float2*)(acc_output + (i_1 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 16; ++i_2) {
    *(float2*)(acc_score_kv + (i_2 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 8; ++i_3) {
    *(float2*)(acc_score_kvcache + (i_3 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 2; ++i_4) {
    log_sum[i_4] = 0x0p+0f/*0.000000e+00*/;
  }
  #pragma unroll
  for (int i_5 = 0; i_5 < 2; ++i_5) {
    scores_max[i_5] = -CUDART_INF_F;
  }
  for (int page_block_idx_local = 0; page_block_idx_local < MAX_SEQ_NUM_BLOCKS; ++page_block_idx_local) {
    int page_block_idx_global = block_tables[((((int64_t)((int)blockIdx.x)) * ((int64_t)MAX_SEQ_NUM_BLOCKS)) + ((int64_t)page_block_idx_local))];
    __syncthreads();
    if (0 <= page_block_idx_global) {
      #pragma unroll
      for (int i_6 = 0; i_6 < 4; ++i_6) {
        uint4 condval_1;
        if ((page_block_idx_global < 505)) {
          condval_1 = *(uint4*)(K_Cache + (((((((int64_t)page_block_idx_global) * (int64_t)16384) + (((int64_t)i_6) * (int64_t)4096)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)512)) + ((((int64_t)((int)blockIdx.y)) / (int64_t)7) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
        } else {
          condval_1 = make_uint4(__pack_nv_bfloat162(bfloat16_t(0x0p+0f/*0.000000e+00*/), bfloat16_t(0x0p+0f/*0.000000e+00*/)), __pack_nv_bfloat162(bfloat16_t(0x0p+0f/*0.000000e+00*/), bfloat16_t(0x0p+0f/*0.000000e+00*/)), __pack_nv_bfloat162(bfloat16_t(0x0p+0f/*0.000000e+00*/), bfloat16_t(0x0p+0f/*0.000000e+00*/)), __pack_nv_bfloat162(bfloat16_t(0x0p+0f/*0.000000e+00*/), bfloat16_t(0x0p+0f/*0.000000e+00*/)));
        }
        *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_6 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 12288)) = condval_1;
      }
      #pragma unroll
      for (int i_7 = 0; i_7 < 16; ++i_7) {
        float condval_2;
        if ((((q_end_idx - q_start_idx) <= ((((((int)threadIdx.x) >> 5) * 16) + (((i_7 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))) || (cur_context_len <= ((((page_block_idx_local * 32) + ((i_7 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_7 & 1))))) {
          condval_2 = -0x1.dcd65p+29f/*-1.000000e+09*/;
        } else {
          condval_2 = 0x0p+0f/*0.000000e+00*/;
        }
        acc_score_kvcache[i_7] = condval_2;
      }
      __syncthreads();
      for (int ki = 0; ki < 8; ++ki) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])) + 0, A_local + 0);
        for (int i_8 = 0; i_8 < 2; ++i_8) {
          tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((ki >> 2) * 2048) + (i_8 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 12288)])) + 0, B_local + (i_8 * 8));
        }
        for (int j = 0; j < 2; ++j) {
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_score_kvcache + (j * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_score_kvcache + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
        }
      }
      #pragma unroll
      for (int i_9 = 0; i_9 < 2; ++i_9) {
        scores_max_prev[i_9] = scores_max[i_9];
      }
      #pragma unroll
      for (int i_10 = 0; i_10 < 2; ++i_10) {
        scores_max[i_10] = -CUDART_INF_F;
      }
      #pragma unroll
      for (int i_11 = 0; i_11 < 2; ++i_11) {
        #pragma unroll
        for (int rv = 0; rv < 8; ++rv) {
          scores_max[i_11] = max(scores_max[i_11], acc_score_kvcache[((((rv & 3) * 4) + (i_11 * 2)) + (rv >> 2))]);
        }
        scores_max[i_11] = tl::AllReduce<tl::MaxOp, 4, 1, 0>::run(scores_max[i_11]);
      }
      #pragma unroll
      for (int i_12 = 0; i_12 < 2; ++i_12) {
        scores_max[i_12] = max(scores_max[i_12], scores_max_prev[i_12]);
      }
      #pragma unroll
      for (int i_13 = 0; i_13 < 2; ++i_13) {
        scores_scale[i_13] = exp2f(((scores_max_prev[i_13] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_13] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      }
      #pragma unroll
      for (int i_14 = 0; i_14 < 16; ++i_14) {
        acc_score_kvcache[i_14] = exp2f(((acc_score_kvcache[i_14] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_14 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      }
      #pragma unroll
      for (int i_15 = 0; i_15 < 2; ++i_15) {
        scores_sum[i_15] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_1 = 0; rv_1 < 8; ++rv_1) {
          scores_sum[i_15] = (scores_sum[i_15] + acc_score_kvcache[((((rv_1 & 3) * 4) + (i_15 * 2)) + (rv_1 >> 2))]);
        }
        scores_sum[i_15] = tl::AllReduce<tl::SumOp, 4, 1, 0>::run(scores_sum[i_15]);
      }
      #pragma unroll
      for (int i_16 = 0; i_16 < 2; ++i_16) {
        log_sum[i_16] = ((log_sum[i_16] * scores_scale[i_16]) + scores_sum[i_16]);
      }
      #pragma unroll
      for (int i_17 = 0; i_17 < 8; ++i_17) {
        uint1 __1;
        float2 v_ = *(float2*)(acc_score_kvcache + (i_17 * 2));
        (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&v_))[0]);
        *(uint1*)(acc_score_kvcache_cast + (i_17 * 2)) = __1;
      }
      #pragma unroll
      for (int i_18 = 0; i_18 < 64; ++i_18) {
        acc_output[i_18] = (acc_output[i_18] * scores_scale[((i_18 & 3) >> 1)]);
      }
      __syncthreads();
      #pragma unroll
      for (int i_19 = 0; i_19 < 4; ++i_19) {
        uint4 condval_3;
        if ((page_block_idx_global < 505)) {
          condval_3 = *(uint4*)(V_Cache + (((((((int64_t)page_block_idx_global) * (int64_t)16384) + (((int64_t)i_19) * (int64_t)4096)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)512)) + ((((int64_t)((int)blockIdx.y)) / (int64_t)7) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
        } else {
          condval_3 = make_uint4(__pack_nv_bfloat162(bfloat16_t(0x0p+0f/*0.000000e+00*/), bfloat16_t(0x0p+0f/*0.000000e+00*/)), __pack_nv_bfloat162(bfloat16_t(0x0p+0f/*0.000000e+00*/), bfloat16_t(0x0p+0f/*0.000000e+00*/)), __pack_nv_bfloat162(bfloat16_t(0x0p+0f/*0.000000e+00*/), bfloat16_t(0x0p+0f/*0.000000e+00*/)), __pack_nv_bfloat162(bfloat16_t(0x0p+0f/*0.000000e+00*/), bfloat16_t(0x0p+0f/*0.000000e+00*/)));
        }
        *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_19 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)) = condval_3;
      }
      __syncthreads();
      for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
        for (int i_20 = 0; i_20 < 8; ++i_20) {
          tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((i_20 >> 2) * 2048) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_20 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_20 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8192)])) + 0, B_local_1 + (i_20 * 8));
        }
        for (int j_1 = 0; j_1 < 8; ++j_1) {
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_output + (j_1 * 8)), reinterpret_cast<const unsigned*>(acc_score_kvcache_cast + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_output + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_score_kvcache_cast + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_21 = 0; i_21 < 8; ++i_21) {
    uint4 condval_4;
    if (((((((((int)threadIdx.x) >> 4) + kv_start_idx) >> 3) + i_21) < 56) && (0 <= (((i_21 * 8) + (((int)threadIdx.x) >> 4)) + kv_start_idx)))) {
      condval_4 = *(uint4*)(K + (((((((int64_t)i_21) * (int64_t)4096) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)512)) + (((int64_t)kv_start_idx) * (int64_t)512)) + ((((int64_t)((int)blockIdx.y)) / (int64_t)7) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
    } else {
      condval_4 = make_uint4(__pack_nv_bfloat162(bfloat16_t(0x0p+0f/*0.000000e+00*/), bfloat16_t(0x0p+0f/*0.000000e+00*/)), __pack_nv_bfloat162(bfloat16_t(0x0p+0f/*0.000000e+00*/), bfloat16_t(0x0p+0f/*0.000000e+00*/)), __pack_nv_bfloat162(bfloat16_t(0x0p+0f/*0.000000e+00*/), bfloat16_t(0x0p+0f/*0.000000e+00*/)), __pack_nv_bfloat162(bfloat16_t(0x0p+0f/*0.000000e+00*/), bfloat16_t(0x0p+0f/*0.000000e+00*/)));
    }
    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_21 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)) = condval_4;
  }
  #pragma unroll
  for (int i_22 = 0; i_22 < 32; ++i_22) {
    float condval_5;
    if ((((q_end_idx - q_start_idx) <= ((((((int)threadIdx.x) >> 5) * 16) + (((i_22 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))) || ((kv_end_idx - kv_start_idx) <= ((((i_22 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_22 & 1))))) {
      condval_5 = -0x1.dcd65p+29f/*-1.000000e+09*/;
    } else {
      condval_5 = 0x0p+0f/*0.000000e+00*/;
    }
    acc_score_kv[i_22] = condval_5;
  }
  __syncthreads();
  for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
    tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki_2 >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])) + 0, A_local_1 + 0);
    for (int i_23 = 0; i_23 < 4; ++i_23) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((ki_2 >> 2) * 4096) + (i_23 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])) + 0, B_local_2 + (i_23 * 8));
    }
    for (int j_2 = 0; j_2 < 4; ++j_2) {
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_score_kv + (j_2 * 8)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_2 + (j_2 * 8)));
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_score_kv + ((j_2 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_2 + ((j_2 * 8) + 4)));
    }
  }
  #pragma unroll
  for (int i_24 = 0; i_24 < 2; ++i_24) {
    scores_max_prev[i_24] = scores_max[i_24];
  }
  #pragma unroll
  for (int i_25 = 0; i_25 < 2; ++i_25) {
    scores_max[i_25] = -CUDART_INF_F;
  }
  #pragma unroll
  for (int i_26 = 0; i_26 < 2; ++i_26) {
    #pragma unroll
    for (int rv_2 = 0; rv_2 < 16; ++rv_2) {
      scores_max[i_26] = max(scores_max[i_26], acc_score_kv[((((rv_2 & 7) * 4) + (i_26 * 2)) + (rv_2 >> 3))]);
    }
    scores_max[i_26] = tl::AllReduce<tl::MaxOp, 4, 1, 0>::run(scores_max[i_26]);
  }
  #pragma unroll
  for (int i_27 = 0; i_27 < 2; ++i_27) {
    scores_max[i_27] = max(scores_max[i_27], scores_max_prev[i_27]);
  }
  #pragma unroll
  for (int i_28 = 0; i_28 < 2; ++i_28) {
    scores_scale[i_28] = exp2f(((scores_max_prev[i_28] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_28] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
  }
  #pragma unroll
  for (int i_29 = 0; i_29 < 32; ++i_29) {
    acc_score_kv[i_29] = exp2f(((acc_score_kv[i_29] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_29 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
  }
  #pragma unroll
  for (int i_30 = 0; i_30 < 2; ++i_30) {
    scores_sum[i_30] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_3 = 0; rv_3 < 16; ++rv_3) {
      scores_sum[i_30] = (scores_sum[i_30] + acc_score_kv[((((rv_3 & 7) * 4) + (i_30 * 2)) + (rv_3 >> 3))]);
    }
    scores_sum[i_30] = tl::AllReduce<tl::SumOp, 4, 1, 0>::run(scores_sum[i_30]);
  }
  #pragma unroll
  for (int i_31 = 0; i_31 < 2; ++i_31) {
    log_sum[i_31] = ((log_sum[i_31] * scores_scale[i_31]) + scores_sum[i_31]);
  }
  #pragma unroll
  for (int i_32 = 0; i_32 < 16; ++i_32) {
    uint1 __2;
    float2 v__1 = *(float2*)(acc_score_kv + (i_32 * 2));
    (reinterpret_cast<__nv_bfloat162*>(&__2))[0] = __float22bfloat162_rn(((float2*)(&v__1))[0]);
    *(uint1*)(acc_score_kv_cast + (i_32 * 2)) = __2;
  }
  #pragma unroll
  for (int i_33 = 0; i_33 < 64; ++i_33) {
    acc_output[i_33] = (acc_output[i_33] * scores_scale[((i_33 & 3) >> 1)]);
  }
  __syncthreads();
  #pragma unroll
  for (int i_34 = 0; i_34 < 8; ++i_34) {
    uint4 condval_6;
    if (((((((((int)threadIdx.x) >> 4) + kv_start_idx) >> 3) + i_34) < 56) && (0 <= (((i_34 * 8) + (((int)threadIdx.x) >> 4)) + kv_start_idx)))) {
      condval_6 = *(uint4*)(V + (((((((int64_t)i_34) * (int64_t)4096) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)512)) + (((int64_t)kv_start_idx) * (int64_t)512)) + ((((int64_t)((int)blockIdx.y)) / (int64_t)7) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
    } else {
      condval_6 = make_uint4(__pack_nv_bfloat162(bfloat16_t(0x0p+0f/*0.000000e+00*/), bfloat16_t(0x0p+0f/*0.000000e+00*/)), __pack_nv_bfloat162(bfloat16_t(0x0p+0f/*0.000000e+00*/), bfloat16_t(0x0p+0f/*0.000000e+00*/)), __pack_nv_bfloat162(bfloat16_t(0x0p+0f/*0.000000e+00*/), bfloat16_t(0x0p+0f/*0.000000e+00*/)), __pack_nv_bfloat162(bfloat16_t(0x0p+0f/*0.000000e+00*/), bfloat16_t(0x0p+0f/*0.000000e+00*/)));
    }
    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_34 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_6;
  }
  __syncthreads();
  for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
    for (int i_35 = 0; i_35 < 8; ++i_35) {
      tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[(((((i_35 >> 2) * 4096) + (ki_3 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_35 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_35 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])) + 0, B_local_3 + (i_35 * 8));
    }
    for (int j_3 = 0; j_3 < 8; ++j_3) {
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_output + (j_3 * 8)), reinterpret_cast<const unsigned*>(acc_score_kv_cast + (ki_3 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + (j_3 * 8)));
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_output + ((j_3 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_score_kv_cast + (ki_3 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + ((j_3 * 8) + 4)));
    }
  }
  #pragma unroll
  for (int i_36 = 0; i_36 < 64; ++i_36) {
    acc_output[i_36] = (acc_output[i_36] / log_sum[((i_36 & 3) >> 1)]);
  }
  __syncthreads();
  #pragma unroll
  for (int i_37 = 0; i_37 < 32; ++i_37) {
    uint1 __3;
    float2 v__2 = *(float2*)(acc_output + (i_37 * 2));
    (reinterpret_cast<__nv_bfloat162*>(&__3))[0] = __float22bfloat162_rn(((float2*)(&v__2))[0]);
    *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((i_37 >> 4) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + ((i_37 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((((i_37 & 15) >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + ((((((((i_37 & 7) >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) >> 4) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_37 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __3;
  }
  __syncthreads();
  #pragma unroll
  for (int i_38 = 0; i_38 < 8; ++i_38) {
    if (((i_38 * 8) + (((int)threadIdx.x) >> 4)) < (q_end_idx - q_start_idx)) {
      if (0 <= (((i_38 * 8) + (((int)threadIdx.x) >> 4)) + q_start_idx)) {
        if (((((((int)threadIdx.x) >> 4) + q_start_idx) >> 3) + i_38) < 56) {
          *(uint4*)(O + (((((((int64_t)i_38) * (int64_t)28672) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)3584)) + (((int64_t)q_start_idx) * (int64_t)3584)) + (((int64_t)((int)blockIdx.y)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8))) = *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_38 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)));
        }
      }
    }
  }
}

