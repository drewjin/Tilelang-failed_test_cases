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
  bfloat16_t acc_score_kv_cast[32];
  bfloat16_t A_local_1[8];
  bfloat16_t B_local_2[32];
  bfloat16_t B_local_3[64];
  int q_start_idx = cu_seqlens_q[0];
  int kv_start_idx = cu_seqlens_k[0];
  int q_end_idx = cu_seqlens_q[1];
  int kv_end_idx = cu_seqlens_k[1];
  int cur_context_len = context_lens[0];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint4 condval;
    if (((((((((int)threadIdx.x) >> 4) + q_start_idx) >> 3) + i) < 4) && (0 <= (((i * 8) + (((int)threadIdx.x) >> 4)) + q_start_idx)))) {
      condval = *(uint4*)(Q + (((((((int64_t)i) * (int64_t)32768) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)4096)) + (((int64_t)q_start_idx) * (int64_t)4096)) + (((int64_t)((int)blockIdx.y)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
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
  int page_block_idx_global = block_tables[0];
  if (0 <= page_block_idx_global) {
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_6 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 16384), K_Cache+(((((((int64_t)page_block_idx_global) * (int64_t)32768) + (((int64_t)i_6) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + ((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)), (page_block_idx_global < 1024));
    }
  }
  tl::cp_async_commit();
  int page_block_idx_global_1 = block_tables[0];
  if (0 <= page_block_idx_global_1) {
    #pragma unroll
    for (int i_7 = 0; i_7 < 4; ++i_7) {
      tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_7 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 24576), V_Cache+(((((((int64_t)page_block_idx_global_1) * (int64_t)32768) + (((int64_t)i_7) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + ((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)), (page_block_idx_global_1 < 1024));
    }
  }
  tl::cp_async_commit();
  for (int page_block_idx_local = 0; page_block_idx_local < (MAX_SEQ_NUM_BLOCKS - 1); ++page_block_idx_local) {
    int page_block_idx_global_2 = block_tables[((int64_t)page_block_idx_local)];
    if (0 <= page_block_idx_global_2) {
      #pragma unroll
      for (int i_8 = 0; i_8 < 16; ++i_8) {
        float condval_1;
        if ((((q_end_idx - q_start_idx) <= ((((((int)threadIdx.x) >> 5) * 16) + (((i_8 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))) || (cur_context_len <= ((((page_block_idx_local * 32) + ((i_8 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_8 & 1))))) {
          condval_1 = -0x1.dcd65p+29f/*-1.000000e+09*/;
        } else {
          condval_1 = 0x0p+0f/*0.000000e+00*/;
        }
        acc_score_kvcache[i_8] = condval_1;
      }
    }
    tl::cp_async_wait<0>();
    __syncthreads();
    int page_block_idx_global_3 = block_tables[((int64_t)page_block_idx_local)];
    if (0 <= page_block_idx_global_3) {
      for (int ki = 0; ki < 8; ++ki) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])) + 0, A_local + 0);
        for (int i_9 = 0; i_9 < 2; ++i_9) {
          tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((ki >> 2) * 2048) + (i_9 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])) + 0, B_local + (i_9 * 8));
        }
        for (int j = 0; j < 2; ++j) {
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_score_kvcache + (j * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_score_kvcache + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
        }
      }
    }
    int page_block_idx_global_4 = block_tables[(((int64_t)page_block_idx_local) + (int64_t)1)];
    __syncthreads();
    if (0 <= page_block_idx_global_4) {
      #pragma unroll
      for (int i_10 = 0; i_10 < 4; ++i_10) {
        tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_10 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 16384), K_Cache+(((((((int64_t)page_block_idx_global_4) * (int64_t)32768) + (((int64_t)i_10) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + ((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)), (page_block_idx_global_4 < 1024));
      }
    }
    tl::cp_async_commit();
    int page_block_idx_global_5 = block_tables[((int64_t)page_block_idx_local)];
    if (0 <= page_block_idx_global_5) {
      #pragma unroll
      for (int i_11 = 0; i_11 < 2; ++i_11) {
        scores_max_prev[i_11] = scores_max[i_11];
      }
    }
    int page_block_idx_global_6 = block_tables[((int64_t)page_block_idx_local)];
    if (0 <= page_block_idx_global_6) {
      #pragma unroll
      for (int i_12 = 0; i_12 < 2; ++i_12) {
        scores_max[i_12] = -CUDART_INF_F;
      }
    }
    int page_block_idx_global_7 = block_tables[((int64_t)page_block_idx_local)];
    if (0 <= page_block_idx_global_7) {
      #pragma unroll
      for (int i_13 = 0; i_13 < 2; ++i_13) {
        #pragma unroll
        for (int rv = 0; rv < 8; ++rv) {
          scores_max[i_13] = max(scores_max[i_13], acc_score_kvcache[((((rv & 3) * 4) + (i_13 * 2)) + (rv >> 2))]);
        }
        scores_max[i_13] = tl::AllReduce<tl::MaxOp, 4, 1, 0>::run(scores_max[i_13]);
      }
    }
    int page_block_idx_global_8 = block_tables[((int64_t)page_block_idx_local)];
    if (0 <= page_block_idx_global_8) {
      #pragma unroll
      for (int i_14 = 0; i_14 < 2; ++i_14) {
        scores_max[i_14] = max(scores_max[i_14], scores_max_prev[i_14]);
      }
    }
    int page_block_idx_global_9 = block_tables[((int64_t)page_block_idx_local)];
    if (0 <= page_block_idx_global_9) {
      #pragma unroll
      for (int i_15 = 0; i_15 < 2; ++i_15) {
        scores_scale[i_15] = exp2f(((scores_max_prev[i_15] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_15] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      }
    }
    int page_block_idx_global_10 = block_tables[((int64_t)page_block_idx_local)];
    if (0 <= page_block_idx_global_10) {
      #pragma unroll
      for (int i_16 = 0; i_16 < 16; ++i_16) {
        acc_score_kvcache[i_16] = exp2f(((acc_score_kvcache[i_16] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_16 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      }
    }
    int page_block_idx_global_11 = block_tables[((int64_t)page_block_idx_local)];
    if (0 <= page_block_idx_global_11) {
      #pragma unroll
      for (int i_17 = 0; i_17 < 2; ++i_17) {
        scores_sum[i_17] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_1 = 0; rv_1 < 8; ++rv_1) {
          scores_sum[i_17] = (scores_sum[i_17] + acc_score_kvcache[((((rv_1 & 3) * 4) + (i_17 * 2)) + (rv_1 >> 2))]);
        }
        scores_sum[i_17] = tl::AllReduce<tl::SumOp, 4, 1, 0>::run(scores_sum[i_17]);
      }
    }
    int page_block_idx_global_12 = block_tables[((int64_t)page_block_idx_local)];
    if (0 <= page_block_idx_global_12) {
      #pragma unroll
      for (int i_18 = 0; i_18 < 2; ++i_18) {
        log_sum[i_18] = ((log_sum[i_18] * scores_scale[i_18]) + scores_sum[i_18]);
      }
    }
    int page_block_idx_global_13 = block_tables[((int64_t)page_block_idx_local)];
    if (0 <= page_block_idx_global_13) {
      #pragma unroll
      for (int i_19 = 0; i_19 < 8; ++i_19) {
        uint1 __1;
        float2 v_ = *(float2*)(acc_score_kvcache + (i_19 * 2));
        (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&v_))[0]);
        *(uint1*)(acc_score_kvcache_cast + (i_19 * 2)) = __1;
      }
    }
    int page_block_idx_global_14 = block_tables[((int64_t)page_block_idx_local)];
    if (0 <= page_block_idx_global_14) {
      #pragma unroll
      for (int i_20 = 0; i_20 < 64; ++i_20) {
        acc_output[i_20] = (acc_output[i_20] * scores_scale[((i_20 & 3) >> 1)]);
      }
    }
    tl::cp_async_wait<0>();
    __syncthreads();
    int page_block_idx_global_15 = block_tables[((int64_t)page_block_idx_local)];
    if (0 <= page_block_idx_global_15) {
      for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
        for (int i_21 = 0; i_21 < 8; ++i_21) {
          tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((i_21 >> 2) * 2048) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_21 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_21 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 12288)])) + 0, B_local_1 + (i_21 * 8));
        }
        for (int j_1 = 0; j_1 < 8; ++j_1) {
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_output + (j_1 * 8)), reinterpret_cast<const unsigned*>(acc_score_kvcache_cast + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_output + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_score_kvcache_cast + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
        }
      }
    }
    int page_block_idx_global_16 = block_tables[(((int64_t)page_block_idx_local) + (int64_t)1)];
    __syncthreads();
    if (0 <= page_block_idx_global_16) {
      #pragma unroll
      for (int i_22 = 0; i_22 < 4; ++i_22) {
        tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_22 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 24576), V_Cache+(((((((int64_t)page_block_idx_global_16) * (int64_t)32768) + (((int64_t)i_22) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + ((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)), (page_block_idx_global_16 < 1024));
      }
    }
    tl::cp_async_commit();
  }
  int page_block_idx_global_17 = block_tables[(((int64_t)MAX_SEQ_NUM_BLOCKS) - (int64_t)1)];
  if (0 <= page_block_idx_global_17) {
    #pragma unroll
    for (int i_23 = 0; i_23 < 16; ++i_23) {
      float condval_2;
      if ((((q_end_idx - q_start_idx) <= ((((((int)threadIdx.x) >> 5) * 16) + (((i_23 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))) || ((cur_context_len + 32) <= ((((MAX_SEQ_NUM_BLOCKS * 32) + ((i_23 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_23 & 1))))) {
        condval_2 = -0x1.dcd65p+29f/*-1.000000e+09*/;
      } else {
        condval_2 = 0x0p+0f/*0.000000e+00*/;
      }
      acc_score_kvcache[i_23] = condval_2;
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  int page_block_idx_global_18 = block_tables[(((int64_t)MAX_SEQ_NUM_BLOCKS) - (int64_t)1)];
  if (0 <= page_block_idx_global_18) {
    for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki_2 >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])) + 0, A_local + 0);
      for (int i_24 = 0; i_24 < 2; ++i_24) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((ki_2 >> 2) * 2048) + (i_24 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])) + 0, B_local + (i_24 * 8));
      }
      for (int j_2 = 0; j_2 < 2; ++j_2) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_score_kvcache + (j_2 * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j_2 * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_score_kvcache + ((j_2 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j_2 * 8) + 4)));
      }
    }
  }
  int page_block_idx_global_19 = block_tables[(((int64_t)MAX_SEQ_NUM_BLOCKS) - (int64_t)1)];
  if (0 <= page_block_idx_global_19) {
    #pragma unroll
    for (int i_25 = 0; i_25 < 2; ++i_25) {
      scores_max_prev[i_25] = scores_max[i_25];
    }
  }
  int page_block_idx_global_20 = block_tables[(((int64_t)MAX_SEQ_NUM_BLOCKS) - (int64_t)1)];
  if (0 <= page_block_idx_global_20) {
    #pragma unroll
    for (int i_26 = 0; i_26 < 2; ++i_26) {
      scores_max[i_26] = -CUDART_INF_F;
    }
  }
  int page_block_idx_global_21 = block_tables[(((int64_t)MAX_SEQ_NUM_BLOCKS) - (int64_t)1)];
  if (0 <= page_block_idx_global_21) {
    #pragma unroll
    for (int i_27 = 0; i_27 < 2; ++i_27) {
      #pragma unroll
      for (int rv_2 = 0; rv_2 < 8; ++rv_2) {
        scores_max[i_27] = max(scores_max[i_27], acc_score_kvcache[((((rv_2 & 3) * 4) + (i_27 * 2)) + (rv_2 >> 2))]);
      }
      scores_max[i_27] = tl::AllReduce<tl::MaxOp, 4, 1, 0>::run(scores_max[i_27]);
    }
  }
  int page_block_idx_global_22 = block_tables[(((int64_t)MAX_SEQ_NUM_BLOCKS) - (int64_t)1)];
  if (0 <= page_block_idx_global_22) {
    #pragma unroll
    for (int i_28 = 0; i_28 < 2; ++i_28) {
      scores_max[i_28] = max(scores_max[i_28], scores_max_prev[i_28]);
    }
  }
  int page_block_idx_global_23 = block_tables[(((int64_t)MAX_SEQ_NUM_BLOCKS) - (int64_t)1)];
  if (0 <= page_block_idx_global_23) {
    #pragma unroll
    for (int i_29 = 0; i_29 < 2; ++i_29) {
      scores_scale[i_29] = exp2f(((scores_max_prev[i_29] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_29] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
  }
  int page_block_idx_global_24 = block_tables[(((int64_t)MAX_SEQ_NUM_BLOCKS) - (int64_t)1)];
  if (0 <= page_block_idx_global_24) {
    #pragma unroll
    for (int i_30 = 0; i_30 < 16; ++i_30) {
      acc_score_kvcache[i_30] = exp2f(((acc_score_kvcache[i_30] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_30 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
  }
  int page_block_idx_global_25 = block_tables[(((int64_t)MAX_SEQ_NUM_BLOCKS) - (int64_t)1)];
  if (0 <= page_block_idx_global_25) {
    #pragma unroll
    for (int i_31 = 0; i_31 < 2; ++i_31) {
      scores_sum[i_31] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_3 = 0; rv_3 < 8; ++rv_3) {
        scores_sum[i_31] = (scores_sum[i_31] + acc_score_kvcache[((((rv_3 & 3) * 4) + (i_31 * 2)) + (rv_3 >> 2))]);
      }
      scores_sum[i_31] = tl::AllReduce<tl::SumOp, 4, 1, 0>::run(scores_sum[i_31]);
    }
  }
  int page_block_idx_global_26 = block_tables[(((int64_t)MAX_SEQ_NUM_BLOCKS) - (int64_t)1)];
  if (0 <= page_block_idx_global_26) {
    #pragma unroll
    for (int i_32 = 0; i_32 < 2; ++i_32) {
      log_sum[i_32] = ((log_sum[i_32] * scores_scale[i_32]) + scores_sum[i_32]);
    }
  }
  int page_block_idx_global_27 = block_tables[(((int64_t)MAX_SEQ_NUM_BLOCKS) - (int64_t)1)];
  if (0 <= page_block_idx_global_27) {
    #pragma unroll
    for (int i_33 = 0; i_33 < 8; ++i_33) {
      uint1 __2;
      float2 v__1 = *(float2*)(acc_score_kvcache + (i_33 * 2));
      (reinterpret_cast<__nv_bfloat162*>(&__2))[0] = __float22bfloat162_rn(((float2*)(&v__1))[0]);
      *(uint1*)(acc_score_kvcache_cast + (i_33 * 2)) = __2;
    }
  }
  int page_block_idx_global_28 = block_tables[(((int64_t)MAX_SEQ_NUM_BLOCKS) - (int64_t)1)];
  if (0 <= page_block_idx_global_28) {
    #pragma unroll
    for (int i_34 = 0; i_34 < 64; ++i_34) {
      acc_output[i_34] = (acc_output[i_34] * scores_scale[((i_34 & 3) >> 1)]);
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  int page_block_idx_global_29 = block_tables[(((int64_t)MAX_SEQ_NUM_BLOCKS) - (int64_t)1)];
  if (0 <= page_block_idx_global_29) {
    for (int ki_3 = 0; ki_3 < 2; ++ki_3) {
      for (int i_35 = 0; i_35 < 8; ++i_35) {
        tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((i_35 >> 2) * 2048) + (ki_3 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_35 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_35 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 12288)])) + 0, B_local_1 + (i_35 * 8));
      }
      for (int j_3 = 0; j_3 < 8; ++j_3) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_output + (j_3 * 8)), reinterpret_cast<const unsigned*>(acc_score_kvcache_cast + (ki_3 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_3 * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_output + ((j_3 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_score_kvcache_cast + (ki_3 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_3 * 8) + 4)));
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_36 = 0; i_36 < 8; ++i_36) {
    tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_36 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 16384), K+(((((((int64_t)i_36) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + (((int64_t)kv_start_idx) * (int64_t)1024)) + ((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)), (((((((((int)threadIdx.x) >> 4) + kv_start_idx) >> 3) + i_36) < 4) && (0 <= (((i_36 * 8) + (((int)threadIdx.x) >> 4)) + kv_start_idx))) && (0 <= (((i_36 * 8) + (((int)threadIdx.x) >> 4)) + kv_start_idx))));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_37 = 0; i_37 < 8; ++i_37) {
    tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_37 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 32768), V+(((((((int64_t)i_37) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + (((int64_t)kv_start_idx) * (int64_t)1024)) + ((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)), ((((((((int)threadIdx.x) >> 4) + kv_start_idx) >> 3) + i_37) < 4) && (0 <= (((i_37 * 8) + (((int)threadIdx.x) >> 4)) + kv_start_idx))));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_38 = 0; i_38 < 32; ++i_38) {
    float condval_3;
    if ((((q_end_idx - q_start_idx) <= ((((((int)threadIdx.x) >> 5) * 16) + (((i_38 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))) || ((kv_end_idx - kv_start_idx) <= ((((i_38 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_38 & 1))))) {
      condval_3 = -0x1.dcd65p+29f/*-1.000000e+09*/;
    } else {
      condval_3 = 0x0p+0f/*0.000000e+00*/;
    }
    acc_score_kv[i_38] = condval_3;
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  for (int ki_4 = 0; ki_4 < 8; ++ki_4) {
    tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki_4 >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_4 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])) + 0, A_local_1 + 0);
    for (int i_39 = 0; i_39 < 4; ++i_39) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((ki_4 >> 2) * 4096) + (i_39 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_4 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])) + 0, B_local_2 + (i_39 * 8));
    }
    for (int j_4 = 0; j_4 < 4; ++j_4) {
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_score_kv + (j_4 * 8)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_2 + (j_4 * 8)));
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_score_kv + ((j_4 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_2 + ((j_4 * 8) + 4)));
    }
  }
  #pragma unroll
  for (int i_40 = 0; i_40 < 2; ++i_40) {
    scores_max_prev[i_40] = scores_max[i_40];
  }
  #pragma unroll
  for (int i_41 = 0; i_41 < 2; ++i_41) {
    scores_max[i_41] = -CUDART_INF_F;
  }
  #pragma unroll
  for (int i_42 = 0; i_42 < 2; ++i_42) {
    #pragma unroll
    for (int rv_4 = 0; rv_4 < 16; ++rv_4) {
      scores_max[i_42] = max(scores_max[i_42], acc_score_kv[((((rv_4 & 7) * 4) + (i_42 * 2)) + (rv_4 >> 3))]);
    }
    scores_max[i_42] = tl::AllReduce<tl::MaxOp, 4, 1, 0>::run(scores_max[i_42]);
  }
  #pragma unroll
  for (int i_43 = 0; i_43 < 2; ++i_43) {
    scores_max[i_43] = max(scores_max[i_43], scores_max_prev[i_43]);
  }
  #pragma unroll
  for (int i_44 = 0; i_44 < 2; ++i_44) {
    scores_scale[i_44] = exp2f(((scores_max_prev[i_44] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_44] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
  }
  #pragma unroll
  for (int i_45 = 0; i_45 < 32; ++i_45) {
    acc_score_kv[i_45] = exp2f(((acc_score_kv[i_45] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_45 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
  }
  #pragma unroll
  for (int i_46 = 0; i_46 < 2; ++i_46) {
    scores_sum[i_46] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_5 = 0; rv_5 < 16; ++rv_5) {
      scores_sum[i_46] = (scores_sum[i_46] + acc_score_kv[((((rv_5 & 7) * 4) + (i_46 * 2)) + (rv_5 >> 3))]);
    }
    scores_sum[i_46] = tl::AllReduce<tl::SumOp, 4, 1, 0>::run(scores_sum[i_46]);
  }
  #pragma unroll
  for (int i_47 = 0; i_47 < 2; ++i_47) {
    log_sum[i_47] = ((log_sum[i_47] * scores_scale[i_47]) + scores_sum[i_47]);
  }
  #pragma unroll
  for (int i_48 = 0; i_48 < 16; ++i_48) {
    uint1 __3;
    float2 v__2 = *(float2*)(acc_score_kv + (i_48 * 2));
    (reinterpret_cast<__nv_bfloat162*>(&__3))[0] = __float22bfloat162_rn(((float2*)(&v__2))[0]);
    *(uint1*)(acc_score_kv_cast + (i_48 * 2)) = __3;
  }
  #pragma unroll
  for (int i_49 = 0; i_49 < 64; ++i_49) {
    acc_output[i_49] = (acc_output[i_49] * scores_scale[((i_49 & 3) >> 1)]);
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  for (int ki_5 = 0; ki_5 < 4; ++ki_5) {
    for (int i_50 = 0; i_50 < 8; ++i_50) {
      tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((i_50 >> 2) * 4096) + (ki_5 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_50 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_50 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16384)])) + 0, B_local_3 + (i_50 * 8));
    }
    for (int j_5 = 0; j_5 < 8; ++j_5) {
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_output + (j_5 * 8)), reinterpret_cast<const unsigned*>(acc_score_kv_cast + (ki_5 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + (j_5 * 8)));
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_output + ((j_5 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_score_kv_cast + (ki_5 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + ((j_5 * 8) + 4)));
    }
  }
  #pragma unroll
  for (int i_51 = 0; i_51 < 64; ++i_51) {
    acc_output[i_51] = (acc_output[i_51] / log_sum[((i_51 & 3) >> 1)]);
  }
  __syncthreads();
  #pragma unroll
  for (int i_52 = 0; i_52 < 32; ++i_52) {
    uint1 __4;
    float2 v__3 = *(float2*)(acc_output + (i_52 * 2));
    (reinterpret_cast<__nv_bfloat162*>(&__4))[0] = __float22bfloat162_rn(((float2*)(&v__3))[0]);
    *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((i_52 >> 4) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + ((i_52 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((((i_52 & 15) >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + ((((((((i_52 & 7) >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) >> 4) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_52 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __4;
  }
  __syncthreads();
  #pragma unroll
  for (int i_53 = 0; i_53 < 8; ++i_53) {
    if (((i_53 * 8) + (((int)threadIdx.x) >> 4)) < (q_end_idx - q_start_idx)) {
      if (0 <= (((i_53 * 8) + (((int)threadIdx.x) >> 4)) + q_start_idx)) {
        if (((((((int)threadIdx.x) >> 4) + q_start_idx) >> 3) + i_53) < 4) {
          *(uint4*)(O + (((((((int64_t)i_53) * (int64_t)32768) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)4096)) + (((int64_t)q_start_idx) * (int64_t)4096)) + (((int64_t)((int)blockIdx.y)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8))) = *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_53 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)));
        }
      }
    }
  }
}

