"""
Auto-generated test case from failed CHECK_FLASH_ATTN_DECODE.
Generated at: 2025-12-28 13:56:10

To use this test case:
1. Load the data: test_data = pickle.load(open("test_data.pkl", "rb"))
2. Move tensors to device: q = test_data['inputs']['q'].to(device), etc.
3. Call your kernel with the loaded inputs
"""
import torch
import pickle
from pathlib import Path

# Load test data
case_dir = Path(__file__).parent
with open(case_dir / "test_data.pkl", "rb") as f:
    test_data = pickle.load(f)

# Extract inputs
q = test_data['inputs']['q']
k = test_data['inputs']['k']
v = test_data['inputs']['v']
k_cache = test_data['inputs']['k_cache']
v_cache = test_data['inputs']['v_cache']
block_tables = test_data['inputs']['block_tables']
context_lens = test_data['inputs']['context_lens']
cu_seqlens_q = test_data['inputs']['cu_seqlens_q']
cu_seqlens_k = test_data['inputs']['cu_seqlens_k']

# Extract parameters
params = test_data['parameters']
max_seqlen_q = params['max_seqlen_q']
scale = params['scale']
num_groups = params['num_groups']
page_block_size = params['page_block_size']
diffusion_block_size = params['diffusion_block_size']
is_block_attn = params['is_block_attn']

# Extract expected outputs for comparison
gt_output = test_data['outputs']['gt_output']

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

# TODO: Add your kernel call here
# kernel_output = your_kernel(q, k, v, k_cache, v_cache, block_tables, context_lens, 
#                             cu_seqlens_q, cu_seqlens_k, max_seqlen_q)
# torch.testing.assert_close(kernel_output, gt_output, atol=params['atol'], rtol=params['rtol'])
