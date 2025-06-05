import torch

# -------------------------------------------------------------
# get_dummy_data：生成用于训练的小批量“假数据”（用于测试训练流程）
# -------------------------------------------------------------
def get_dummy_data(num_samples=1024, dim=128, num_classes=10):
    # 设置随机种子，确保每次生成数据一致（可复现）
    torch.manual_seed(0)

    # 随机生成输入特征数据（标准正态分布）
    # x 的形状为 [num_samples, dim]，表示样本数量 × 每个样本的特征维度
    x = torch.randn(num_samples, dim, device="cuda")

    # 随机生成整数标签 y，范围是 [0, num_classes)
    # y 的形状为 [num_samples]，每个值是一个类别编号
    y = torch.randint(0, num_classes, (num_samples,), device="cuda")

    return x, y
