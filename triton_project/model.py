import torch
import torch.nn as nn

# 从 Triton 自定义 softmax / layernorm 模块中导入
from kernels.softmax import softmax
from kernels.layernorm import layernorm

# ============================================================
# 模型定义：SimpleTritonModel
# - 构建一个最小神经网络：包含 Triton 自定义 LayerNorm 和 Softmax
# - 用于测试整个 Triton 加速推理和训练流程
# ============================================================
class SimpleTritonModel(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()

        # 输入特征维度（例如 128）
        self.dim = dim
        # 分类任务的类别数量（例如 10）
        self.num_classes = num_classes

        # 一个隐藏层（线性变换），输入输出维度一致
        self.linear = nn.Linear(dim, dim)

        # 输出层：将特征投影为类别数
        self.output = nn.Linear(dim, num_classes)

    # 前向传播定义（forward）
    def forward(self, x):
        # Triton 自定义 LayerNorm（替代 nn.LayerNorm）
        x = layernorm(x)

        # 一层线性变换（维度不变）
        x = self.linear(x)

        # Triton 自定义 Softmax（用于行归一化）
        x = softmax(x)

        # 输出分类得分（未加 softmax，因为交叉熵损失自带）
        x = self.output(x)

        return x
