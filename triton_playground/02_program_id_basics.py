import torch
import triton
import triton.language as tl

# ============================================
# Triton kernel：记录每个线程块（program）负责的起始索引
@triton.jit
def record_offsets_kernel(OUTPUT, N, BLOCK_SIZE: tl.constexpr):
    # 获取当前程序块的编号，类似 CUDA 的 blockIdx.x
    pid = tl.program_id(0)

    # 当前线程块要处理的起始全局索引（例如 block 3 就从 3*BLOCK_SIZE 开始）
    start = pid * BLOCK_SIZE

    # 使用 mask 防止写入越界（因为可能最后一个 block 已超出 N）
    mask = start < N

    # 将每个线程块负责的起始索引写入到输出 Tensor 中
    # OUTPUT 是一个大小为 num_blocks 的张量，每个位置保存一个起始 offset
    tl.store(OUTPUT + pid, start, mask=mask)

# ============================================
# Python 主程序

# 要处理的总数据长度
N = 1000

# 每个线程块负责处理多少个元素
BLOCK_SIZE = 128

# Triton 的线程调度：自动向上取整
# 例如：1000 / 128 = 7.8125 → ceil → 8 个线程块
NUM_BLOCKS = triton.cdiv(N, BLOCK_SIZE)

# 创建输出张量，用来存放每个线程块的起始 offset
# 由于我们只需要 1 个值存每个 block 的处理起点，所以张量长度为 NUM_BLOCKS
output_tensor = torch.empty(NUM_BLOCKS, device='cuda', dtype=torch.int32)

# 启动 Triton kernel：
# grid = (NUM_BLOCKS,) 表示启动 NUM_BLOCKS 个线程块
record_offsets_kernel[(NUM_BLOCKS,)](
    output_tensor,  # 每个线程块将自己的起始索引写入这个 Tensor
    N,
    BLOCK_SIZE=BLOCK_SIZE
)

# 把结果从 GPU 拷贝回 CPU，并打印出来观察每个线程块做了什么
print("✅ 每个线程块处理的起始索引：")
print(output_tensor.cpu())
