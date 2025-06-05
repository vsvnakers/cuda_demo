import torch
import triton
import triton.language as tl

# ----------------------------------------
# Triton kernel：每个 block 对一段数据归约求和
@triton.jit
def blockwise_sum_kernel(X, partial_sums, N, BLOCK_SIZE: tl.constexpr):
    # 获取当前 block 的 ID（类似 CUDA 中 blockIdx.x）
    pid = tl.program_id(0)

    # 当前 block 要处理的一段全局索引
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 创建 mask：屏蔽掉越界索引（offset >= N）
    mask = offsets < N

    # 安全加载数据（越界部分不会读）
    x = tl.load(X + offsets, mask=mask)

    # 每个 block 内部进行归约：将 BLOCK_SIZE 个元素求和
    # 这里使用 tl.sum（默认 float32 精度），自动并行处理
    acc = tl.sum(x, axis=0)

    # 将当前 block 的求和结果写入 partial_sums[pid]
    tl.store(partial_sums + pid, acc)

# ----------------------------------------
# Python 主程序部分

# ❗ 实验原理：
# 本实验目的是验证：使用 Triton 进行分块并行归约求和，与 PyTorch 内置 sum() 的数值精度差异。
# 我们将一个大数组 x 拆分成多个 block，在每个 block 内独立完成求和，并最终用 CPU 汇总。
# 由于浮点数的加法顺序不同（即使数学上 a + b + c = c + a + b），在 float32 精度下误差可能出现。
# 所以我们也特意将值乘以 1e3 来放大误差，让你能看到两个实现之间微小的误差差异。

# 向量长度
N = 1000  # 可以增大为 1_000_000 以放大误差现象

# 每个 block 处理的数据量
BLOCK_SIZE = 256

# 总共需要几个 block（向上取整）
NUM_BLOCKS = triton.cdiv(N, BLOCK_SIZE)

# 创建输入数据（值乘以 1e3 放大误差影响）
x = torch.rand(N, device='cuda', dtype=torch.float32) * 1e3

# partial_sums 保存每个 block 的局部和
partial_sums = torch.empty(NUM_BLOCKS, device='cuda', dtype=torch.float32)

# 启动 Triton kernel（NUM_BLOCKS 个并行程序块）
blockwise_sum_kernel[(NUM_BLOCKS,)](
    x, partial_sums, N,
    BLOCK_SIZE=BLOCK_SIZE
)

# Triton 的最终求和结果：把所有局部和加起来
total_sum = partial_sums.sum().item()

# PyTorch 的真实参考值（单次高精度求和）
torch_sum = x.sum().item()

# 计算绝对误差
error = abs(total_sum - torch_sum)

# 打印结果
print(f"✅ Triton 总和：   {total_sum:.6f}")
print(f"✅ PyTorch 总和： {torch_sum:.6f}")
print(f"🎯 误差：           {error:.6e}")
