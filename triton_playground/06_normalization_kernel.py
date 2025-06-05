import torch
import triton
import triton.language as tl

# ----------------------------------------------------
# Triton Kernel：对每一行做标准化 z = (x - mean) / std
# 每一行对应一个 Triton 程序块（program_id）
# ----------------------------------------------------
@triton.jit
def normalize_kernel(X, Y, N_COLS, EPSILON: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # 当前线程块处理的行号（类似 CUDA 中 blockIdx.x）
    row = tl.program_id(0)

    # 本行所有列索引（长度为 BLOCK_SIZE）
    cols = tl.arange(0, BLOCK_SIZE)

    # 当前 row 中该 block 的 global 偏移位置（行起点 + 每列偏移）
    offset = row * N_COLS + cols

    # 避免访问越界位置（如果 BLOCK_SIZE > N_COLS）
    mask = cols < N_COLS

    # 加载当前行数据（带 mask，越界列不读）
    x = tl.load(X + offset, mask=mask)

    # step 1: 计算均值 mean
    mean = tl.sum(x, axis=0) / N_COLS

    # step 2: 计算方差（Var = E[(x - mean)^2]）
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / N_COLS

    # step 3: 计算标准差并添加 epsilon 保证数值稳定
    std = tl.sqrt(var + EPSILON)

    # step 4: 标准化计算
    z = (x - mean) / std

    # 写入输出张量（写回 global memory）
    tl.store(Y + offset, z, mask=mask)

# ----------------------------------------
# Python 主程序部分（用于验证）
# ----------------------------------------

# 模拟数据规模（100 行，每行 256 列）
N_ROWS = 100
N_COLS = 256
BLOCK_SIZE = 256       # Triton 块大小，必须为常量
EPSILON = 1e-5          # 数值稳定项，防止除以 0

# 创建输入数据（模拟有方差的样本行）
x = torch.randn((N_ROWS, N_COLS), device='cuda', dtype=torch.float32) * 1000
y = torch.empty_like(x)

# 启动 Triton kernel：每一行对应一个 block
normalize_kernel[(N_ROWS,)](
    x, y, N_COLS,
    EPSILON=EPSILON,
    BLOCK_SIZE=BLOCK_SIZE
)

# 用 PyTorch 验证标准化是否正确
mean_ref = x.mean(dim=1, keepdim=True)
std_ref = x.std(dim=1, keepdim=True, unbiased=False)  # unbiased=False 匹配我们除以 N
y_ref = (x - mean_ref) / (std_ref + EPSILON)

# 计算最大误差
max_error = torch.max(torch.abs(y - y_ref)).item()

print(f"✅ Triton 标准化完成，最大误差：{max_error:.6e}")
