import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------
# Triton kernel：每一行做 softmax 运算
# softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
# 为了数值稳定性，先减去每行最大值，再进行归一化
# ---------------------------------------------------------------
@triton.jit
def softmax_kernel(X, Y, N_COLS, EPSILON: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # 获取当前程序块的 ID（对应一行）
    row = tl.program_id(0)

    # 当前线程负责的列索引范围（从0到BLOCK_SIZE）
    cols = tl.arange(0, BLOCK_SIZE)

    # 偏移：映射到当前行的列元素（全局 memory 索引）
    offset = row * N_COLS + cols

    # mask：用于避免访问越界（当列索引 ≥ N_COLS 时屏蔽）
    mask = cols < N_COLS

    # 从 global memory 加载当前行数据（只加载合法部分）
    x = tl.load(X + offset, mask=mask)

    # Step 1：减去每行最大值（提升数值稳定性）
    row_max = tl.max(x, axis=0)
    x_exp = tl.exp(x - row_max)

    # Step 2：对每行求 exp(x - max) 的总和（加上 epsilon 防止除以0）
    row_sum = tl.sum(x_exp, axis=0) + EPSILON

    # Step 3：归一化，得到 softmax 输出
    y = x_exp / row_sum

    # 将计算结果写回 global memory
    tl.store(Y + offset, y, mask=mask)


# ---------------------------------------------------------------
# softmax：用于 Python 调用接口
# 输入 X（二维张量：行×列），每一行做 softmax
# ---------------------------------------------------------------
def softmax(x, block_size=128):
    # 创建输出张量（shape 同输入）
    y = torch.empty_like(x)

    # 启动 Triton kernel
    softmax_kernel[
        (x.shape[0],)  # grid = 行数，每行一个 block
    ](
        x,                    # 输入张量
        y,                    # 输出张量
        x.shape[1],           # 每行列数（N_COLS）
        EPSILON=1e-6,         # 避免除以 0
        BLOCK_SIZE=block_size
    )

    return y
