# Triton_from_CUDA: GPU 编程实验项目（基于 OpenAI Triton）

本项目用于探索和实验 **OpenAI Triton 框架**，用 Python 编写高性能 GPU Kernel。项目中使用虚拟环境隔离依赖，适合在 WSL + Ubuntu + NVIDIA GPU 环境下开发运行。

---

## 📦 环境说明

- Python 3.10+
- Triton (`pip install triton`)
- PyTorch (用于张量操作)
- NVIDIA GPU（支持 CUDA 11.4+）
- WSL2 + Ubuntu / Linux

项目使用本地虚拟环境 `.venv/` 进行依赖隔离，并已通过 `.gitignore` 忽略该目录。

---

## 🧰 环境配置步骤

```bash
# 安装 Python 和 venv（如未安装）
sudo apt update
sudo apt install python3.10 python3.10-venv -y

# 创建虚拟环境
python3 -m venv .venv

# 激活虚拟环境
source .venv/bin/activate

# 安装依赖
pip install -U pip
pip install triton torch numpy
```

---

📁 项目结构总览（目录说明）

```
TRITON_FROM_CUDA/
├── .venv/              # Python 虚拟环境（不纳入版本控制）
├── .vscode/            # VSCode 编辑器设置
├── .gitignore          # Git 忽略配置文件
├── README.md           # 项目说明文档

├── cuda_file/          # ✅ CUDA 学习代码文件夹（基础 GPU 并行计算入门）
│   └── *.cu            # 示例：float vs double 精度误差，block reduce 等

├── cuda_build/         # ⚙️ CUDA 编译后生成的可执行文件目录（自动生成，可忽略）

├── triton_playground/  # 🧪 Triton 基础教学实验代码（含 program_id、归约、broadcast 等）
│   └── *.py            # 每讲一个文件，建议逐个执行理解

├── triton_project/     # 🏗️ Triton 实战项目：训练模型（softmax/layernorm kernel 自定义）

```


