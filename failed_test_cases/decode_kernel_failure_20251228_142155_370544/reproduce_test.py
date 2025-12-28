"""
Auto-generated test case from failed CHECK_FLASH_ATTN_DECODE.
Generated at: 2025-12-28 11:46:20

To use this test case:
1. Load the data: test_data = pickle.load(open("test_data.pkl", "rb"))
2. Move tensors to device: q = test_data['inputs']['q'].to(device), etc.
3. Call your kernel with the loaded inputs
"""
import torch
import pickle
from pathlib import Path

from einops import rearrange
import torch.nn.functional as F

from diffulex_kernel.python.dllm_flash_attn_kernels import dllm_flash_attn_decode_kernel


def naive_sdpa_with_kvcache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    scale: float,
    num_groups: int,
    page_block_size: int,
) -> torch.Tensor:
    """
    Naive attention reference implementation with KV cache support.
    
    Args:
        q: [Q_LEN, NUM_HEADS, HEAD_DIM]
        k: [KV_LEN, NUM_KV_HEADS, HEAD_DIM]
        v: [KV_LEN, NUM_KV_HEADS, HEAD_DIM]
        k_cache: [NUM_PAGE_BLOCKS, PAGE_BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM]
        v_cache: [NUM_PAGE_BLOCKS, PAGE_BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM]
        block_tables: [NUM_SEQS, MAX_SEQ_NUM_BLOCKS]
        context_lens: [NUM_SEQS]
        cu_seqlens_q: [NUM_SEQS + 1]
        cu_seqlens_k: [NUM_SEQS + 1]
        scale: attention scale
        num_groups: number of GQA groups
        page_block_size: page block size
    
    Returns:
        output: [Q_LEN, NUM_HEADS, HEAD_DIM]
    """
    num_seqs = len(cu_seqlens_q) - 1
    
    output = torch.zeros_like(q)
    for seq_idx in range(num_seqs):
        q_start = cu_seqlens_q[seq_idx].item()
        q_end = cu_seqlens_q[seq_idx + 1].item()
        kv_start = cu_seqlens_k[seq_idx].item()
        kv_end = cu_seqlens_k[seq_idx + 1].item()
        
        q_seq = q[q_start:q_end]  # [seq_q_len, num_heads, head_dim]
        k_seq = k[kv_start:kv_end]  # [seq_kv_len, num_kv_heads, head_dim]
        v_seq = v[kv_start:kv_end]  # [seq_kv_len, num_kv_heads, head_dim]
        
        context_len = context_lens[seq_idx].item()
        
        # Load KV cache for this sequence
        k_cache_seq_list = []
        v_cache_seq_list = []
        
        for block_idx in range(block_tables.shape[1]):
            page_block_idx = block_tables[seq_idx, block_idx].item()
            if page_block_idx >= 0:
                # Calculate how many tokens to take from this block
                block_start = block_idx * page_block_size
                if block_start < context_len:
                    block_end = min(block_start + page_block_size, context_len)
                    num_tokens = block_end - block_start
                    k_cache_seq_list.append(k_cache[page_block_idx, :num_tokens])
                    v_cache_seq_list.append(v_cache[page_block_idx, :num_tokens])
        
        if k_cache_seq_list:
            k_cache_seq = torch.cat(k_cache_seq_list, dim=0)  # [context_len, num_kv_heads, head_dim]
            v_cache_seq = torch.cat(v_cache_seq_list, dim=0)  # [context_len, num_kv_heads, head_dim]
            
            # Combine KV cache and current KV
            k_combined = torch.cat([k_cache_seq, k_seq], dim=0)
            v_combined = torch.cat([v_cache_seq, v_seq], dim=0)
        else:
            k_combined = k_seq
            v_combined = v_seq

        q_sdpa = rearrange(q_seq, 's h d -> 1 h s d') # [1, num_heads, seq_q_len, head_dim]
        k_sdpa = rearrange(k_combined, 's h d -> 1 h s d') # [1, num_heads, total_kv_len, head_dim]
        v_sdpa = rearrange(v_combined, 's h d -> 1 h s d') # [1, num_heads, total_kv_len, head_dim]

        attn_out = F.scaled_dot_product_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
            enable_gqa=True,
        )  # [1, num_heads, seq_q_len, head_dim]

        output[q_start:q_end] = rearrange(attn_out, '1 h s d -> s h d').to(output.dtype)
    
    return output

# Load test data
case_dir = Path(__file__).parent
with open(case_dir / "test_data.pkl", "rb") as f:
    test_data = pickle.load(f)

device = "cuda"

# Extract inputs
q = test_data['inputs']['q'].to(device)
k = test_data['inputs']['k'].to(device)
v = test_data['inputs']['v'].to(device)
k_cache = test_data['inputs']['k_cache'].to(device)
v_cache = test_data['inputs']['v_cache'].to(device)
block_tables = test_data['inputs']['block_tables'].to(device)
context_lens = test_data['inputs']['context_lens'].to(device)
cu_seqlens_q = test_data['inputs']['cu_seqlens_q'].to(device)
cu_seqlens_k = test_data['inputs']['cu_seqlens_k'].to(device)

# Extract parameters
params = test_data['parameters']
max_seqlen_q = params['max_seqlen_q']
scale = params['scale']
num_groups = params['num_groups']
page_block_size = params['page_block_size']
diffusion_block_size = params['diffusion_block_size']
is_block_attn = params['is_block_attn']

# Extract expected outputs for comparison
gt_output = test_data['outputs']['gt_output'].to(device)

# Print test case info
print("Test Case Information:")
q_shape = test_data['shapes']['q_shape']
k_shape = test_data['shapes']['k_shape']
v_shape = test_data['shapes']['v_shape']
print(f"  Shapes: q={q_shape}, k={k_shape}, v={v_shape}")
print(f"  Parameters: scale={scale}, num_groups={num_groups}, page_block_size={page_block_size}")
max_diff_val = test_data['statistics']['max_diff']
num_mismatches = test_data['statistics']['num_exceeds_tolerance']
print(f"  Statistics: max_diff={max_diff_val:.6f}, num_mismatches={num_mismatches}")

# 重新计算 gt_output 以验证一致性
gt_sample_output = naive_sdpa_with_kvcache(
    q, k, v, k_cache, v_cache, 
    block_tables, context_lens, 
    cu_seqlens_q, cu_seqlens_k, 
    scale, num_groups, page_block_size
)

# 检查重新计算的 gt_output 与保存的 gt_output 是否一致
gt_diff = torch.abs(gt_sample_output - gt_output)
print(f"\nGT Output Verification:")
print(f"  Max difference between recomputed and saved GT: {gt_diff.max().item():.6f}")
print(f"  Mean difference: {gt_diff.mean().item():.6f}")
print(f"  Elements with difference > 1e-6: {(gt_diff > 1e-6).sum().item()}")

try:
    torch.testing.assert_close(gt_sample_output, gt_output, atol=0.02, rtol=0.05)
    print("  ✓ GT output matches (recomputed vs saved)")
except AssertionError as e:
    print(f"  ✗ GT output mismatch: {e}")
    # 如果 GT 不匹配，使用新计算的 GT
    print("  Warning: Using recomputed GT output instead of saved one")
    gt_output = gt_sample_output

decode_kernel = dllm_flash_attn_decode_kernel(
    len(cu_seqlens_q) - 1,
    num_groups,
    k_cache.shape[0], # num_page_blocks
    q.shape[0],
    k.shape[0],
    q.shape[1], # num_heads
    q.shape[2], # head_dim
    is_block_attn,
    diffusion_block_size,
    block_tables.shape[1], # max_seq_num_blocks
    page_block_size,
    64,
    64,
    1,
    128,
)

# 显式同步 CUDA 以确保之前的所有操作完成
torch.cuda.synchronize()

# 运行 kernel
kernel_output = decode_kernel(
    q, k, v, k_cache, v_cache, 
    block_tables, context_lens, 
    cu_seqlens_q, cu_seqlens_k, 
    max_seqlen_q,
)

# 同步以确保 kernel 完成
torch.cuda.synchronize()

# 检查输出中是否有 NaN 或 Inf
print("\nKernel Output Diagnostics:")
print(f"  Has NaN: {torch.isnan(kernel_output).any().item()}")
print(f"  Has Inf: {torch.isinf(kernel_output).any().item()}")
print(f"  Min value: {kernel_output.min().item():.6f}")
print(f"  Max value: {kernel_output.max().item():.6f}")
print(f"  Mean value: {kernel_output.mean().item():.6f}")

# 计算差异统计
abs_diff = torch.abs(kernel_output - gt_output)
rel_diff = torch.abs((kernel_output - gt_output) / (gt_output + 1e-8))
print(f"\nDifference Statistics:")
print(f"  Max absolute difference: {abs_diff.max().item():.6f}")
print(f"  Mean absolute difference: {abs_diff.mean().item():.6f}")
print(f"  Max relative difference: {rel_diff.max().item():.6f}")
print(f"  Mean relative difference: {rel_diff.mean().item():.6f}")

# 使用更严格的容差进行测试（如 debugger 中）
print("\nTesting with strict tolerance (atol=0.01, rtol=0.01):")
strict_atol = 0.01
strict_rtol = 0.01
strict_exceeds_atol = (abs_diff > strict_atol)
strict_exceeds_rtol = (rel_diff > strict_rtol)
strict_exceeds_tolerance = strict_exceeds_atol | strict_exceeds_rtol
strict_num_exceeds = strict_exceeds_tolerance.sum().item()
print(f"  Elements exceeding strict tolerance: {strict_num_exceeds}")

try:
    torch.testing.assert_close(kernel_output, gt_output, atol=strict_atol, rtol=strict_rtol)
    print("  ✓ Passed with strict tolerance")
except AssertionError as e:
    print(f"  ✗ Failed with strict tolerance")
    print(f"  Note: torch.testing.assert_close only reports the FIRST mismatched element,")
    print(f"        but there are actually {strict_num_exceeds} elements exceeding tolerance.")
    print(f"  Error message: {e}")

# 使用原始容差进行测试
print("\nTesting with original tolerance (atol=0.02, rtol=0.05):")
atol = 0.01
rtol = 0.01

# 计算超过容差的元素
exceeds_atol = (abs_diff > atol)
exceeds_rtol = (rel_diff > rtol)
exceeds_tolerance = exceeds_atol | exceeds_rtol
num_exceeds_tolerance = exceeds_tolerance.sum().item()

print(f"  Elements exceeding absolute tolerance (atol={atol}): {exceeds_atol.sum().item()}")
print(f"  Elements exceeding relative tolerance (rtol={rtol}): {exceeds_rtol.sum().item()}")
print(f"  Elements exceeding either tolerance: {num_exceeds_tolerance}")

if num_exceeds_tolerance > 0:
    # 找出超过容差的元素位置
    mismatch_indices = torch.nonzero(exceeds_tolerance, as_tuple=False)
    num_to_show = min(10, num_exceeds_tolerance)
    print(f"\n  First {num_to_show} mismatched elements:")
    for i in range(num_to_show):
        idx = mismatch_indices[i]
        abs_val = abs_diff[idx[0], idx[1], idx[2]].item()
        rel_val = rel_diff[idx[0], idx[1], idx[2]].item()
        kernel_val = kernel_output[idx[0], idx[1], idx[2]].item()
        gt_val = gt_output[idx[0], idx[1], idx[2]].item()
        print(f"    Index {idx.tolist()}: kernel={kernel_val:.6f}, gt={gt_val:.6f}, "
              f"abs_diff={abs_val:.6f}, rel_diff={rel_val:.6f}")

try:
    torch.testing.assert_close(kernel_output, gt_output, atol=atol, rtol=rtol)
    print("\n  ✓ Passed with original tolerance")
except AssertionError as e:
    print(f"\n  ✗ Failed with original tolerance")
    print(f"  Note: torch.testing.assert_close only reports the FIRST mismatched element,")
    print(f"        but there are actually {num_exceeds_tolerance} elements exceeding tolerance.")
    print(f"  Error message: {e}")