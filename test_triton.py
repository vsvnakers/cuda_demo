import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(X, Y, Z, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    y = tl.load(Y + offsets, mask=mask)
    tl.store(Z + offsets, x + y, mask=mask)

N = 1024
x = torch.rand(N, device='cuda')
y = torch.rand(N, device='cuda')
z = torch.empty_like(x)

add_kernel[(triton.cdiv(N, 256),)](x, y, z, N, BLOCK_SIZE=256)
assert torch.allclose(z, x + y)
print("✅ Triton kernel executed successfully!")
