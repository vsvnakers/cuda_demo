import torch
import triton
import triton.language as tl

# ====================================================
# Triton kernel：将每个线程的全局索引写入到 output 中
# 并使用 mask 避免越界写入
@triton.jit
def masked_store_kernel(OUTPUT, N, BLOCK_SIZE: tl.constexpr):
    # 当前线程块（program）编号，类似 CUDA 中 blockIdx.x
    pid = tl.program_id(0)

    # 每个线程块处理 BLOCK_SIZE 个元素
    # offsets 是当前 block 内所有线程负责写入的位置（全局索引）
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # mask 用于判断哪些 offset 是合法的（小于 N）
    mask = offsets < N

    # 安全写入：只在 mask=True 的位置写入 offsets 的值
    # 如果不加 mask，offsets 超出 N 时会导致越界写入
    tl.store(OUTPUT + offsets, offsets, mask=mask)

# ====================================================
# Python 主程序部分

# 我们要处理的数据总长度 N
N = 1001  # 故意选择非 BLOCK_SIZE 整数倍，测试边界处理

# 每个线程块处理多少元素
BLOCK_SIZE = 256

# 计算需要多少个线程块（向上取整覆盖所有数据）
NUM_BLOCKS = triton.cdiv(N, BLOCK_SIZE)

# 创建一个 output 张量，用于保存 kernel 写入的值
# ✅ torch.full 用法：
# - 创建一个形状为 (N+10,) 的一维张量
# - 默认值为 -1（这样写入位置和未写入位置可以区分）
# - 放在 GPU 上（device='cuda'）
# - 数据类型为 int32
output = torch.full((N + 10,), -1, device='cuda', dtype=torch.int32)

# 启动 Triton kernel，传入 grid 大小为 NUM_BLOCKS 个 program
masked_store_kernel[(NUM_BLOCKS,)](
    output,  # 输出张量，将被 Triton kernel 写入
    N,       # 数据总长度
    BLOCK_SIZE=BLOCK_SIZE  # 编译时常量
)

# ✅ output.cpu()：
# - 将 GPU 上的张量复制回 CPU 内存
# - 方便用 print 显示内容
out_cpu = output.cpu()

# 打印前 10 个值，验证前部写入正确
print("✅ 前 10 个元素：", out_cpu[:10].tolist())

# 打印最后部分，观察最后一个 block 是否写到正确位置，越界是否未写
print("✅ 最后 10 个元素：", out_cpu[N-10:N+5].tolist())
