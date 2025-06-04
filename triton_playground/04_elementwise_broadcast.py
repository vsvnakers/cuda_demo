import torch
import triton
import triton.language as tl

# Triton kernel：执行 z[i] = x[i]^2 + y[i]^2
@triton.jit
def square_sum_kernel(X, Y, Z, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # 加载 x 和 y 的值（注意是值，不是指针）
    x_vals = tl.load(X + offsets, mask=mask)
    y_vals = tl.load(Y + offsets, mask=mask)

    # 元素级操作：平方再相加
    z_vals = x_vals * x_vals + y_vals * y_vals

    # 写入结果到输出数组 Z 中
    tl.store(Z + offsets, z_vals, mask=mask)

# 主程序：测试该 kernel
N = 1023  # 特意选不是 BLOCK_SIZE 的倍数
BLOCK_SIZE = 256
NUM_BLOCKS = triton.cdiv(N, BLOCK_SIZE)

# 初始化输入张量
x = torch.randn(N, device='cuda', dtype=torch.float32)
y = torch.randn(N, device='cuda', dtype=torch.float32)
z = torch.empty_like(x)

# 调用 Triton kernel 执行
square_sum_kernel[(NUM_BLOCKS,)](x, y, z, N, BLOCK_SIZE=BLOCK_SIZE)

# 验证 Triton 的计算是否正确
z_ref = x**2 + y**2
assert torch.allclose(z, z_ref, atol=1e-5)

print("✅ Triton Elementwise Kernel 运行成功，计算正确！")
