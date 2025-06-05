import torch
import torch.nn as nn
import torch.optim as optim

from model import SimpleTritonModel
from data.dummy_data import get_dummy_data

# =============================================
# 超参数设置
# =============================================
dim = 128
num_classes = 10
num_epochs = 3
lr = 1e-3
batch_size = 256

# =============================================
# 数据生成
# =============================================
x, y = get_dummy_data(num_samples=2048, dim=dim, num_classes=num_classes)

# =============================================
# 切分为 batch（手动模拟）
# =============================================
n_batches = x.shape[0] // batch_size
x_batches = x.split(batch_size)
y_batches = y.split(batch_size)

# =============================================
# 初始化模型与优化器
# =============================================
model = SimpleTritonModel(dim=dim, num_classes=num_classes).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# =============================================
# 训练循环
# =============================================
for epoch in range(num_epochs):
    total_loss = 0.0
    print(f"\n🟡 Epoch {epoch + 1}/{num_epochs} 开始训练 =======================")

    for batch_idx, (xb, yb) in enumerate(zip(x_batches, y_batches)):
        preds = model(xb)
        loss = criterion(preds, yb)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 打印每个 batch 的输出示例
        if batch_idx == 0:
            softmax_probs = torch.softmax(preds, dim=1)
            pred_classes = preds.argmax(dim=1)
            print(f"\n🔹 输入样本前5维（xb[0][:5]）: {xb[0][:5].tolist()}")
            print(f"🔹 Logits（preds[0]）:         {preds[0].tolist()}")
            print(f"🔹 Softmax 概率（前3类）:     {softmax_probs[0][:3].tolist()}")
            print(f"🔹 预测类别: {pred_classes[0].item()} | 真实标签: {yb[0].item()}")

        print(f"   🔁 Batch {batch_idx + 1}/{n_batches} - Loss: {loss.item():.4f}")

    avg_loss = total_loss / n_batches
    print(f"✅ Epoch {epoch + 1} 结束 - 平均 Loss: {avg_loss:.4f}")
