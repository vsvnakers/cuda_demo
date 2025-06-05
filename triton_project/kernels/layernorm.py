import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------
# Triton kernel：实现每一行 LayerNorm 操作
# 计算公式：z = (x - mean) / sqrt(var + epsilon)
# ---------------------------------------------------------------
@triton.jit
def layernorm_kernel(X, Y, N_COLS, EPSILON: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # 当前行索引（每个 block 处理一行）
    row = tl.program_id(0)

    # 每个线程处理 BLOCK_SIZE 中的一列
    cols = tl.arange(0, BLOCK_SIZE)

    # 全局偏移量（将 row 和 col 映射为平铺的 1D 索引）
    offset = row * N_COLS + cols

    # 掩码：仅处理 cols < N_COLS 的合法数据
    mask = cols < N_COLS

    # 从全局 memory 加载当前行的数据（屏蔽非法索引）
    x = tl.load(X + offset, mask=mask)

    # Step 1: 计算均值 mean（按行）
    mean = tl.sum(x, axis=0) / N_COLS

    # Step 2: 计算方差 var（按行）→ E[(x - mean)^2]
    var = tl.sum((x - mean) * (x - mean), axis=0) / N_COLS

    # Step 3: 计算标准差 std（加上 epsilon 避免除 0）
    std = tl.sqrt(var + EPSILON)

    # Step 4: 归一化输出
    normed = (x - mean) / std

    # 写回全局 memory
    tl.store(Y + offset, normed, mask=mask)


# ---------------------------------------------------------------
# Python 包装函数：提供和 PyTorch 类似的 layernorm 接口
# 输入 x：二维张量，每一行做 LayerNorm
# ---------------------------------------------------------------
def layernorm(x, block_size=128):
    # 输出张量：与输入形状相同
    y = torch.empty_like(x)

    # 启动 Triton kernel（每一行对应一个 block）
    layernorm_kernel[
        (x.shape[0],)  # grid：x 的行数
    ](
        x, y, x.shape[1],     # 输入 X，输出 Y，每行列数
        EPSILON=1e-5,         # 稳定项，避免除以 0
        BLOCK_SIZE=block_size
    )

    return y
