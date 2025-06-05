import torch
import triton
import triton.language as tl

# ----------------------------------------------------------
# Triton Kernel：实现 GELU 激活函数（带 tanh 近似公式）
# 原理：GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
# GELU 用于替代 ReLU，具有更平滑的梯度，常见于 Transformer/BERT
# 此处你手写 tanh 的替代公式：tanh(x) ≈ (e^(2x) - 1) / (e^(2x) + 1)
# ----------------------------------------------------------
@triton.jit
def gelu_kernel(X, Y, N_ELEMENTS, BLOCK_SIZE: tl.constexpr):
    # 每个线程块唯一编号，类似 CUDA 中的 blockIdx.x
    pid = tl.program_id(0)

    # 计算当前线程块内的全局元素索引范围
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 避免越界：最后一个 block 可能不足 BLOCK_SIZE
    mask = offsets < N_ELEMENTS

    # 加载输入数据 X[offsets]
    x = tl.load(X + offsets, mask=mask)

    # -----------------------------
    # GELU 核心计算公式（近似推导）
    # -----------------------------

    # √(2/π) 是 GELU 中 tanh 的系数
    sqrt_2_over_pi = 0.7978845608028654

    # 常数项 0.044715 是经验拟合参数
    coeff = 0.044715

    # 计算 x³，用于展开式
    x_cube = x * x * x

    # inner = √(2/π) * (x + 0.044715 * x³)
    inner = sqrt_2_over_pi * (x + coeff * x_cube)

    # tanh 替代写法：tanh(x) ≈ (exp(2x) - 1) / (exp(2x) + 1)
    exp_inner = tl.exp(2 * inner)
    tanh_approx = (exp_inner - 1) / (exp_inner + 1)

    # 最终 GELU 激活计算
    gelu = 0.5 * x * (1 + tanh_approx)

    # 写回输出张量 Y[offsets]
    tl.store(Y + offsets, gelu, mask=mask)


# ----------------------------------------------------------
# Python 端：构造输入张量，运行 kernel，验证结果
# ----------------------------------------------------------

# 输入元素总数（如 1024 x 1024 个元素）
N_ELEMENTS = 1024 * 1024
BLOCK_SIZE = 256  # 每个 Triton block 处理的元素数量

# 构造随机输入张量（float32, GPU 上）
x = torch.randn(N_ELEMENTS, device='cuda', dtype=torch.float32)
y = torch.empty_like(x)

# 启动 kernel：共 ceil(N / BLOCK_SIZE) 个 block，每个处理 BLOCK_SIZE 个元素
gelu_kernel[(triton.cdiv(N_ELEMENTS, BLOCK_SIZE),)](
    x, y, N_ELEMENTS,
    BLOCK_SIZE=BLOCK_SIZE
)

# ----------------------------------------------------------
# 验证结果是否正确（与 PyTorch 的 GELU 对比）
# ----------------------------------------------------------

# 使用 PyTorch 的 GELU（tanh 近似）作为真值
y_ref = torch.nn.functional.gelu(x, approximate='tanh')

# 计算最大误差
max_error = torch.max(torch.abs(y - y_ref)).item()

# 打印误差结果（建议 < 1e-6）
print(f"✅ GELU kernel 执行成功，最大误差：{max_error:.6e}")
