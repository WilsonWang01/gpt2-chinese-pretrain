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

---

## 🔴 新增严重问题 (2026-02-03 训练补充)

### 15. GenerationCallback 中 datetime 未导入

**报错信息**:
```
NameError: name 'datetime' is not defined. Did you forget to import 'datetime'?
```

**原因**: 在 callback 类中使用 `datetime.now()` 但忘记在脚本顶部导入

**解决方案**:
```python
# 在脚本顶部添加
from datetime import datetime
```

---

### 16. Tokenizer 词表大小错误 (unk_id 问题) - "Final Boss"

**报错信息**:
```
ValueError: unk_id 0 is out of range (vocab_size = 3)
# 或
RuntimeError: Vocabulary size mismatch: config=32000, actual=3
```

**原因**: HuggingFace 的 `LlamaTokenizerFast` / `AutoTokenizer` 加载 SentencePiece 模型时解析错误，返回错误的词表大小

**解决方案** (手动重建 tokenizer.json):
```python
from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import Unigram
from transformers import PreTrainedTokenizerFast

# 1. 读取 SP 词表文件
vocab_path = Path(work_dir) / "chinese_sp.vocab"
vocab_list = []
with open(vocab_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            vocab_list.append((parts[0], float(parts[1])))

# 2. 用 tokenizers 库重建
tokenizer_obj = Tokenizer(Unigram(vocab_list))
tokenizer_obj.decoder = decoders.Metaspace()
tokenizer_obj.pre_tokenizer = pre_tokenizers.Metaspace()
tokenizer_obj.save(str(tokenizer_dir / "tokenizer.json"))

# 3. 用 PreTrainedTokenizerFast 加载
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=str(tokenizer_dir / "tokenizer.json"),
    bos_token="<s>", eos_token="</s>",
    unk_token="<unk>", pad_token="<pad>",
)
```

---

### 17. Checkpoint 只保留最后 3 个，无法对比早期效果

**现象**: 训练结束后只有 checkpoint-11000, 11500, 11838，无法对比 step 100/500/1000 时的随机生成效果

**原因**: `save_total_limit=3` 导致早期 checkpoint 被自动删除

**解决方案**:
```python
# 增加保留数量
training_args = TrainingArguments(
    save_steps=500,
    save_total_limit=15,  # 保留更多 checkpoint
    ...
)
```

---

### 18. 生成样本只打印到控制台，训练后无法找回

**问题**: `GenerationCallback` 只 `print()` 生成结果，训练结束后无法查看历史样本

**解决方案** (保存到日志文件):
```python
class GenerationCallback(TrainerCallback):
    def __init__(self, tokenizer, work_dir, prompts=None):
        self.log_file = Path(work_dir) / "generation_samples.log"
        # 初始化日志
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"# 创建时间: {datetime.now()}\n")
    
    def on_evaluate(self, args, state, control, model, **kwargs):
        # ... 生成逻辑 ...
        
        # 保存到文件
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"Step {step} | Loss: {loss}\n")
            for prompt, result in results:
                f.write(f"[{prompt}] → {result}\n")
```

---

### 19. argparse 参数名格式错误

**报错信息**:
```
error: unrecognized arguments: --work-dir --batch-size
```

**原因**: Python argparse 定义时用的 `work_dir`，调用时用 `work-dir` (连字符 vs 下划线)

**解决方案**:
```bash
# ❌ 错误
python a100_train.py --work-dir gpt2-chinese --batch-size 48

# ✅ 正确 (用下划线)
python a100_train.py --work_dir gpt2-chinese --batch_size 48
```

**预防**: 在 argparse 定义时同时添加两种格式:
```python
parser.add_argument("--work_dir", "--work-dir", type=str, default="gpt2-chinese")
```

---

### 20. eval_loss 不显示

**现象**: 训练日志只有 `train_loss`，没有 `eval_loss`

**原因**: 设置了 `prediction_loss_only=True`

**解决方案**:
```python
training_args = TrainingArguments(
    eval_strategy="steps",
    eval_steps=500,
    # prediction_loss_only=False,  # 注释掉或设为 False
    ...
)
```

---

## 🟠 新增中等问题

### 21. SCP 文件传输需要密码交互

**现象**: `scp` 命令阻塞等待密码输入

**解决方案**:
1. 设置 SSH 免密登录
2. 或使用 Base64 编码通过 SSH 传输:
```bash
# 服务器端: 编码压缩
tar czf - files/ | base64 > transfer.b64

# 本地: 解码
cat transfer.b64 | base64 -d | tar xzf -
```

---

### 22. Generation 时 results 变量未定义

**报错信息**:
```
NameError: name 'results' is not defined
```

**原因**: 在循环前忘记初始化 `results = []`

**解决方案**:
```python
def on_evaluate(...):
    results = [header]  # 初始化列表
    for prompt in self.prompts:
        # ...
        results.append(line)
```

---

## 📋 训练日志保存检查清单

运行训练前确保以下项目已配置:

- [ ] `save_total_limit >= 10` (保留足够多的 checkpoint)
- [ ] `GenerationCallback` 保存到文件而非只 print
- [ ] `logging_steps=10` (频繁记录 loss)
- [ ] `eval_strategy="steps"` + `eval_steps` 已设置
- [ ] `report_to="tensorboard"` 或其他日志工具
- [ ] 训练结束后下载 `trainer_state.json` + `generation_samples.log`

---

## 🔧 远程服务器训练完整流程

```bash
# 1. 上传脚本
scp -P 35036 a100_train.py root@server:/root/autodl-tmp/

# 2. SSH 连接
ssh -p 35036 root@server

# 3. 启动训练 (用 nohup 防止断连)
cd /root/autodl-tmp
nohup python3 a100_train.py --work_dir gpt2-chinese --batch_size 48 --use_bf16 > train.log 2>&1 &

# 4. 查看进度
tail -f train.log

# 5. 训练完成后下载
scp -P 35036 root@server:/root/autodl-tmp/gpt2-chinese/generation_samples.log ./
scp -P 35036 -r root@server:/root/autodl-tmp/gpt2-chinese/checkpoints/checkpoint-*/trainer_state.json ./
```

---

*最后更新: 2026-02-03 (补充训练日志保存经验)*
