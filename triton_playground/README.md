
# 🚀 Triton 学习路径：8 讲全览

| 讲次       | 内容主题                          | 目标与核心知识                                                  | 状态             |
| -------- | ----------------------------- | -------------------------------------------------------- | -------------- |
| ✅ 第 1 讲  | Triton 基础结构与 kernel 编写        | `@triton.jit`、`program_id`、`tl.arange`、数据加载与 store       | 已完成            |
| ✅ 第 2 讲  | 程序索引与广播机制                     | `grid`, `tl.broadcast`, `tl.store/load` 配合实现多块 kernel 映射 | 已完成            |
| ✅ 第 3 讲  | 逐元素运算 kernel（elementwise）     | 模拟 PyTorch 的逐元素加法、乘法                                     | 已完成            |
| ✅ 第 4 讲  | blockwise 广播归约                | `tl.sum`, `mask`, 局部和、广播机制                               | 已完成            |
| ✅ 第 5 讲  | block 内归约：求和 sum kernel       | 自己实现分块并行 `sum` 并与 PyTorch 验证精度差异                         | 已完成  |
| 🔜 第 6 讲 | 标准化 Kernel：`(x - mean) / std` | 学习 `mean`, `var`, `归一化`，构建 LayerNorm 的核心组件               | 已完成            |
| 🔜 第 7 讲 | Triton 中实现 GELU / Swish 激活函数  | 模拟 BERT/GPT 中非线性激活层的高效实现                                 | 已完成            |
| 🔜 第 8 讲 | Triton Softmax / Argmax 实现    | 自定义 `Softmax`，实现 fused 版本加速 attention                    | 已完成           |

---
