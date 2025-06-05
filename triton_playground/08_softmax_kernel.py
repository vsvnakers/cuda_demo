import torch
import triton
import triton.language as tl

# ------------------------------------------------------
# Triton kernel：逐行计算 softmax（每一行一个 block）
# 采用 log-sum-exp 技巧避免溢出
# ------------------------------------------------------
@triton.jit
def softmax_kernel(X, Y, N_COLS, EPSILON: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # 每个 block 负责一整行（row）
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)

    # 当前行全局偏移索引
    offset = row * N_COLS + cols
    mask = cols < N_COLS

    # 加载该行元素
    x = tl.load(X + offset, mask=mask)

    # step 1：最大值 max (防止 e^x 爆炸)
    row_max = tl.max(x, axis=0)

    # step 2：减去最大值后做 exp
    x_exp = tl.exp(x - row_max)

    # step 3：求 exp 和
    row_sum = tl.sum(x_exp, axis=0) + EPSILON  # 加 ε 防止除 0

    # step 4：归一化
    softmax = x_exp / row_sum

    # 写入输出
    tl.store(Y + offset, softmax, mask=mask)


# ----------------------------------------
# Python 主程序测试
# ----------------------------------------

N_ROWS = 512       # 总共 512 个样本（行）
N_COLS = 256       # 每行 256 个特征
BLOCK_SIZE = 256   # Triton block size = 每行处理多少列
EPSILON = 1e-6

# 创建输入矩阵
x = torch.randn((N_ROWS, N_COLS), device='cuda', dtype=torch.float32)
y = torch.empty_like(x)

# 启动 Triton kernel：每一行一个程序块
softmax_kernel[(N_ROWS,)](
    x, y, N_COLS,
    EPSILON=EPSILON,
    BLOCK_SIZE=BLOCK_SIZE
)

# PyTorch 真值（用于验证）
y_ref = torch.nn.functional.softmax(x, dim=1)

# 最大误差对比
max_error = torch.max(torch.abs(y - y_ref)).item()
print(f"✅ Softmax kernel 执行成功，最大误差：{max_error:.6e}")
