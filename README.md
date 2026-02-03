# 🇨🇳 中文 GPT-2 从零预训练

从随机权重开始，使用中文维基百科和知乎数据训练一个中文 GPT-2 语言模型。

[![Demo](https://img.shields.io/badge/🤗%20Demo-Gradio-yellow)](https://huggingface.co/spaces/Wilsonwin/gpt2-chinese-demo)
[![Model](https://img.shields.io/badge/🤗%20Model-HuggingFace-blue)](https://huggingface.co/Wilsonwin/gpt2-chinese-pretrained)

## 📊 训练结果

| 指标 | 值 |
|------|-----|
| 起始 Loss | 7.40 |
| 最终 Loss | 4.25 |
| Loss 下降 | 42.6% |
| 总步数 | 11,838 |
| 训练时长 | ~1.5 小时 (A100) |

## 🚀 快速开始

### 方式一：直接使用训练好的模型

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Wilsonwin/gpt2-chinese-pretrained")
model = AutoModelForCausalLM.from_pretrained("Wilsonwin/gpt2-chinese-pretrained")

text = "人工智能的未来"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100, do_sample=True, temperature=0.8)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 方式二：从零训练自己的模型

参考下方详细教程。

---

## 📚 详细教程

### 1. 环境准备

#### AutoDL（推荐）

1. 在 [AutoDL](https://www.autodl.com/) 租用 GPU 实例
   - 推荐配置：A100-40GB / RTX 4090 / RTX 5090
   - 镜像选择：`PyTorch 2.x + CUDA 12.x`

2. SSH 连接到实例：
```bash
ssh -p [端口] root@[地址]
```

3. 安装依赖：
```bash
pip install transformers datasets sentencepiece accelerate bitsandbytes
pip install flash-attn --no-build-isolation  # 可选，加速训练
```

#### Kaggle（免费）

1. 在 [Kaggle](https://www.kaggle.com/) 创建 Notebook
2. 开启 GPU 加速（Settings → Accelerator → GPU T4 x2）
3. 开启网络访问（Settings → Internet → On）

### 2. 获取 HuggingFace Token

1. 注册 [HuggingFace](https://huggingface.co/) 账号
2. 进入 [Token 设置页面](https://huggingface.co/settings/tokens)
3. 创建 Token（需要 Write 权限）

### 3. 运行训练

#### AutoDL 训练

```bash
# 1. 上传训练脚本
scp -P [端口] a100_train.py root@[地址]:/root/autodl-tmp/

# 2. SSH 登录后运行
cd /root/autodl-tmp
source /etc/network_turbo  # 开启学术加速
export HF_TOKEN="你的token"
python a100_train.py \
    --work_dir /root/autodl-tmp/gpt2-chinese \
    --cache_dir /root/autodl-tmp/cache \
    --batch_size 48 \
    --num_epochs 2
```

#### Kaggle 训练

复制 `kaggle_pretrain_combined.py` 内容到 Kaggle Notebook 运行。

### 4. 模型部署

训练完成后，模型会自动上传到 HuggingFace Hub。你也可以创建 Gradio Space：

```python
from huggingface_hub import create_repo, upload_file

create_repo("你的用户名/gpt2-chinese-demo", repo_type="space", space_sdk="gradio")
upload_file("app.py", "你的用户名/gpt2-chinese-demo", repo_type="space")
```

---

## 📁 项目结构

```
├── a100_train.py              # AutoDL/高端GPU训练脚本（推荐）
├── autodl_train.py            # AutoDL简化版训练脚本
├── kaggle_pretrain_combined.py # Kaggle训练脚本
├── hf_space_app.py            # Gradio演示应用
├── requirements.txt           # 依赖列表
└── README.md                  # 本文档
```

## ⚙️ 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch_size` | 32 | 批次大小 |
| `--num_epochs` | 2 | 训练轮数 |
| `--learning_rate` | 3e-4 | 学习率 |
| `--vocab_size` | 32000 | 词表大小 |
| `--n_embd` | 768 | 嵌入维度 |
| `--n_layer` | 12 | Transformer 层数 |
| `--n_head` | 12 | 注意力头数 |

## 🔧 常见问题

### Q: 训练中断了怎么办？
A: 使用 `--resume` 参数从 checkpoint 恢复训练。

### Q: 显存不够怎么办？
A: 1) 减小 `--batch_size`；2) 开启 gradient checkpointing（默认开启）

### Q: 如何使用自己的数据？
A: 修改 `load_dataset_cached()` 函数，返回包含 "text" 字段的 Dataset。

### Q: 为什么 Loss 下降到 4.x 就不动了？
A: 这是正常的。小模型在有限数据上的收敛点大约在 4-5 之间。增加模型规模和数据量可以继续降低 Loss。

## 📖 技术细节

- **分词器**: SentencePiece Unigram（32K 词表，专为中文优化）
- **模型架构**: GPT-2（12层 Transformer Decoder）
- **训练数据**: 中文维基百科 + 知乎问答（约 126 万条）
- **优化器**: AdamW 8-bit（bitsandbytes）
- **学习率调度**: Cosine with Warmup

## 📄 License

MIT License

## 🙏 致谢

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [AutoDL](https://www.autodl.com/)
- [Kaggle](https://www.kaggle.com/)
