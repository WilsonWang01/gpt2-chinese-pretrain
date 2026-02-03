# GPT-2 中文预训练踩坑指南

> 本文档记录了 A100/5090 训练脚本开发过程中遇到的所有问题和解决方案，避免后续踩坑。

---

## 🔴 严重问题 (会导致崩溃)

### 1. DDP + torch.compile + Gradient Checkpointing 冲突

**报错信息**:
```
RuntimeError: Parameter ... has been marked as ready twice
RuntimeError: expect_autograd_hooks_ INTERNAL ASSERT FAILED
```

**原因**: 三者同时启用会导致 PyTorch 内部状态冲突

**解决方案**:
```python
# 方案1: 二选一 (推荐 torch.compile，A100 40GB 够用)
gradient_checkpointing=not args.use_compile,
torch_compile=args.use_compile,

# 方案2: 禁用 torch.compile
gradient_checkpointing=True,
torch_compile=False,
```

---

### 2. 手动 DDP + Trainer 自动 DDP = 双重包装

**报错信息**:
```
RuntimeError: Expected all tensors to be on the same device
AttributeError: 'DistributedDataParallel' object has no attribute 'config'
```

**原因**: 同时手动调用 `DDP(model)` 和设置 `TrainingArguments(local_rank=...)`

**解决方案**:
```python
# ❌ 错误做法
model = DDP(model, device_ids=[local_rank])
trainer = Trainer(model=model, ...)

# ✅ 正确做法 - 让 Trainer 自动处理
trainer = Trainer(model=model, ...)  # 直接传入，不要手动 DDP
```

---

### 3. torch.compile 与 Checkpoint 保存冲突

**报错信息**:
```
KeyError: '_orig_mod.transformer.wte.weight'
RuntimeError: Error(s) in loading state_dict
```

**原因**: `torch.compile` 会将权重名加上 `_orig_mod` 前缀

**解决方案**:
```python
# TrainingArguments 自动处理
training_args = TrainingArguments(
    torch_compile=True,  # 让 Trainer 管理 compile
    ...
)
```

---

### 4. bitsandbytes CUDA 12 兼容性

**报错信息**:
```
The installed version of bitsandbytes was compiled without GPU support
libbitsandbytes_cuda124.so: cannot open shared object file
```

**原因**: bitsandbytes 二进制版本与 CUDA 12.x 不完全兼容

**解决方案**:
```python
def get_optimizer_name(use_8bit_adam):
    if not use_8bit_adam:
        return "adamw_torch_fused"
    try:
        import bitsandbytes as bnb
        test_param = torch.zeros(1, device='cuda', requires_grad=True)
        bnb.optim.Adam8bit([test_param])  # 测试是否真正支持
        return "adamw_bnb_8bit"
    except Exception:
        return "adamw_torch_fused"  # Fallback
```

---

### 5. resume 无 Checkpoint 时崩溃

**报错信息**:
```
ValueError: No valid checkpoint found in output directory
```

**解决方案**:
```python
resume_path = None
if args.resume:
    checkpoint_dir = Path(args.work_dir) / "checkpoints"
    checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
    if checkpoints:
        resume_path = True
    else:
        print("⚠️ 未找到 checkpoint，从头开始")
trainer.train(resume_from_checkpoint=resume_path)
```

---

## 🟠 中等问题 (性能下降)

### 6. DataLoader num_workers 过低

**现象**: GPU 利用率 50-70%，`nvidia-smi` 显示 GPU 经常空闲

**解决方案**:
```python
dataloader_num_workers=min(8, multiprocessing.cpu_count()),
dataloader_persistent_workers=True,  # 避免每 epoch 重建
dataloader_pin_memory=True,
```

---

### 7. check_environment 返回 None

**问题**: DDP 时非主进程 Flash Attention 失效

```python
# ❌ 错误
def check_environment(local_rank=0):
    if local_rank != 0:
        return  # 返回 None

# ✅ 正确
def check_environment(local_rank=0):
    if local_rank != 0:
        return False  # 明确返回 False
```

---

### 8. Hub 仓库不存在时上传失败

**报错信息**:
```
huggingface_hub.utils._errors.RepositoryNotFoundError
```

**解决方案**:
```python
api.create_repo(repo_id=repo_id, exist_ok=True, private=False)  # 先创建
api.upload_folder(folder_path=..., repo_id=repo_id, ...)
```

---

### 9. SentencePiece 训练慢

**现象**: 分词器训练 10+ 分钟

**解决方案**:
```python
spm.SentencePieceTrainer.train(
    input_sentence_size=200000,  # 限制样本数
    shuffle_input_sentence=True,
    num_threads=multiprocessing.cpu_count(),
    ...
)
```

---

### 10. map batch_size 过大导致 OOM

**报错信息**:
```
Killed (signal 9)
MemoryError
```

**解决方案**:
```python
dataset.map(
    tokenize_fn,
    batch_size=2000,  # 从 5000 降到 2000
    num_proc=min(8, cpu_count()),
)
```

---

## 🟡 注意事项

### 11. Flash Attention 需要 Ampere+ 架构

```python
if torch.cuda.get_device_capability()[0] >= 8:  # SM 8.0+
    # 支持 Flash Attention (A100, A10, RTX 30xx, 40xx, 50xx)
else:
    # 不支持 (V100, T4, P100)
```

---

### 12. TF32 只有 Ampere+ 支持

```python
# A100/A800/H100/5090 可用，V100/T4 不可用
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

---

### 13. 环境变量设置顺序

```python
# 必须在任何 HuggingFace 导入之前设置
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/root/autodl-tmp/cache")

# 然后才能 import
from transformers import ...
```

---

### 14. 完整随机种子设置

```python
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

---

## 📊 GPU 推荐配置

| GPU | batch_size | compile_mode | 特殊优化 |
|-----|-----------|--------------|---------|
| A100-40GB | 48 | reduce-overhead | TF32 |
| A100-80GB | 64 | reduce-overhead | TF32 |
| RTX 5090 | 48 | max-autotune | - |
| RTX 4090 | 32 | max-autotune | - |
| T4 (Kaggle) | 8 | 不推荐 | DataParallel |

---

## 🔧 调试命令

```bash
# 查看 GPU 状态
watch -n 1 nvidia-smi

# 查看 CUDA 版本
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# 测试 bitsandbytes
python -c "import bitsandbytes; print(bitsandbytes.__version__)"

# 测试 Flash Attention
python -c "import flash_attn; print(flash_attn.__version__)"

# 查看 GPU 架构
python -c "import torch; print(torch.cuda.get_device_capability())"
```

---

## 📝 版本要求

| 依赖 | 最低版本 | 推荐版本 |
|-----|---------|---------|
| Python | 3.10 | 3.11 |
| PyTorch | 2.0 | 2.2+ |
| CUDA | 11.8 (A100) | 12.1+ |
| Transformers | 4.36 | 4.40+ |
| flash-attn | 2.0 | 2.5+ |

---

*最后更新: 2026-02-03*
