import torch
import triton
import triton.language as tl

# ============================================
# Triton GPU kernel：实现两个向量 x + y = z 的并行加法
# 使用 @triton.jit 装饰器表示将其 JIT 编译为 GPU kernel
@triton.jit
def add_kernel(X, Y, Z, N, BLOCK_SIZE: tl.constexpr):
    # 获取当前线程块（program）的编号，类似 CUDA 中 blockIdx.x
    pid = tl.program_id(0)

    # 计算当前 block 要处理的数据索引范围（偏移量）
    # tl.arange(0, BLOCK_SIZE) 生成局部线程 ID 向量 [0, 1, ..., BLOCK_SIZE-1]
    # offsets 即当前线程块内每个线程要处理的数据索引
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 创建一个 mask，表示哪些 offset 是合法的（offset < N）
    # 防止处理超出数组长度的非法索引
    mask = offsets < N

    # 使用 mask 安全地从 X 和 Y 中读取数据
    x_vals = tl.load(X + offsets, mask=mask)
    y_vals = tl.load(Y + offsets, mask=mask)

    # 执行加法操作
    z_vals = x_vals + y_vals

    # 将结果写入 Z 中，仍使用 mask 避免越界写
    tl.store(Z + offsets, z_vals, mask=mask)

# ============================================
# Python 主程序入口

# 设置向量长度
N = 1024

# 在 CUDA GPU 上创建输入张量
x = torch.rand(N, device='cuda', dtype=torch.float32)
y = torch.rand(N, device='cuda', dtype=torch.float32)
z = torch.empty_like(x)  # 用于保存输出

# 启动 Triton kernel
# grid 传入一个元组：表示需要启动多少个 program（即线程块）
# triton.cdiv(N, 256)：向上取整，确保所有元素被覆盖
add_kernel[(triton.cdiv(N, 256),)](
    x, y, z, N, BLOCK_SIZE=256
)

# 用 PyTorch 检查 Triton kernel 是否计算正确
assert torch.allclose(z, x + y, atol=1e-6)

print("✅ Triton kernel executed successfully!")
