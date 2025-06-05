
# Triton Project - 使用 Triton 编写的 GPU 加速训练模型

本项目展示如何使用 [Triton](https://triton-lang.org) 编写 GPU kernel，并与 PyTorch 集成，构建一个包含 Softmax 和 LayerNorm 的简易神经网络进行训练。

目录结构：

* `train.py`：主训练脚本，负责训练流程与输出结果。
* `model.py`：定义模型结构，集成自定义 Triton kernel。
* `data/dummy_data.py`：用于生成随机训练数据。
* `kernels/layernorm.py`：Triton 实现的 LayerNorm 内核。
* `kernels/softmax.py`：Triton 实现的 Softmax 内核。

依赖环境：

* Ubuntu - 24.02
* Python ≥ 3.8
* CUDA-capable GPU
* 已安装 Triton 与 PyTorch

安装依赖（建议在虚拟环境中操作）：

```
pip install torch triton
```

运行方法：

1. 进入项目目录
2. 执行命令：

```
python3 train.py
```

执行效果：

* 每轮（epoch）训练开始和结束都会输出标识
* 每个 batch 的损失会被打印
* 首个 batch 会详细打印前几维样本的：

  * 原始输入
  * 模型 logits
  * softmax 概率
  * 预测类别 vs 真实标签
* 最后输出每轮平均损失

示例输出片段：

```
✅ Epoch 1/3 开始训练
⚙️  Batch 1/8 - Loss: 2.3035
🔍 输入样本前5维: [0.15, 0.22, ..., -0.03]
📊 Logits: [1.2, 0.9, -0.4]
📈 Softmax 概率: [0.52, 0.39, 0.09]
🎯 预测类别: 0 | 真实标签: 0
...
✅ Epoch 3 结束 - 平均 Loss: 0.0385
```

