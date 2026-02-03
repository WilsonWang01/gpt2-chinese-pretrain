# Kaggle Mini GPT-2 中文预训练 - 多数据集版本
# ===================================================
# 使用 维基百科 + 知乎 合并数据集 (~625M tokens)
# 设置: Accelerator = GPU T4 x2, Internet = On
# ===================================================

# ================== Cell 1: 环境设置 ==================
print("=" * 60)
print("🔧 第一阶段：环境配置")
print("=" * 60)

import subprocess
subprocess.run(["nvidia-smi"], check=True)

subprocess.run([
    "pip", "install", "-q",
    "transformers", "datasets", "accelerate",
    "huggingface_hub", "sentencepiece", "tokenizers",
    "bitsandbytes",      # 8-bit 优化器
    "liger-kernel",      # Triton LayerNorm 加速
], check=True)

import torch
print(f"\n✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA: {torch.cuda.is_available()}")
print(f"✅ GPU 数量: {torch.cuda.device_count()}")

# ================== Cell 2: HuggingFace 登录 ==================
from huggingface_hub import login

# 从 Kaggle Secrets 获取 Token
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
    login(token=hf_token)
    print("✅ 使用 Kaggle Secret 登录成功")
except Exception as e:
    print(f"⚠️ Secret 获取失败: {e}")
    from huggingface_hub import notebook_login
    notebook_login()

from huggingface_hub import whoami
try:
    HF_USERNAME = whoami()["name"]
    print(f"✅ 当前用户: {HF_USERNAME}")
except:
    HF_USERNAME = "YOUR_USERNAME"
    print(f"⚠️ 使用默认用户名: {HF_USERNAME}")

# ================== Cell 3: 多数据集加载 ==================
print("\n" + "=" * 60)
print("📥 第二阶段：多数据集加载")
print("=" * 60)

from datasets import load_dataset, concatenate_datasets
import os

print("📥 正在下载多个数据集...")

# === 1. 加载维基百科（完整）===
print("   [1/2] 加载维基百科...")
wiki = load_dataset(
    "pleisto/wikipedia-cn-20230720-filtered",
    split="train",
)
print(f"   ✅ 维基百科: {len(wiki)} 条")

# === 2. 加载知乎高赞回答 ===
print("   [2/2] 加载知乎...")
zhihu = load_dataset(
    "wangrui6/Zhihu-KOL",
    split="train",
)
print(f"   ✅ 知乎: {len(zhihu)} 条")

# === 3. 统一字段名并合并 ===
def process_wiki(example):
    return {"text": example["completion"]}

def process_zhihu(example):
    text = f"{example['INSTRUCTION']}\n{example['RESPONSE']}"
    return {"text": text}

print("🔄 正在处理数据格式...")
wiki_processed = wiki.map(process_wiki, remove_columns=wiki.column_names, num_proc=4)
zhihu_processed = zhihu.map(process_zhihu, remove_columns=zhihu.column_names, num_proc=4)

dataset = concatenate_datasets([wiki_processed, zhihu_processed])
dataset = dataset.shuffle(seed=42)

print(f"\n✅ 数据集合并完成!")
print(f"   维基百科: {len(wiki)} 条")
print(f"   知乎: {len(zhihu)} 条")
print(f"   合计: {len(dataset)} 条")

# === 4. 导出纯文本 ===
CORPUS_FILE = "/kaggle/working/combined_corpus.txt"
print("📝 正在导出纯文本...")
# === 优化：采样导出语料（仅用于分词器训练）===
SAMPLE_SIZE = 500000  # 50 万句足够训练 32K 词表

# 打乱并采样
import random
random.seed(42)
indices = list(range(len(dataset)))
random.shuffle(indices)
sampled_indices = indices[:SAMPLE_SIZE]

# 批量写入，加速 I/O
print(f"📝 采样 {SAMPLE_SIZE} 句用于分词器训练...")
batch_size = 10000
with open(CORPUS_FILE, "w", encoding="utf-8") as f:
    batch = []
    count = 0
    for idx in sampled_indices:
        text = dataset[idx]["text"].strip()
        if len(text) > 50:
            batch.append(text)
            count += 1
            if len(batch) >= batch_size:
                f.write("\n".join(batch) + "\n")
                batch = []
    if batch:
        f.write("\n".join(batch) + "\n")

file_size_mb = os.path.getsize(CORPUS_FILE) / (1024 * 1024)
print(f"✅ 语料导出完成: {file_size_mb:.1f} MB ({count} 句)")

# ================== Cell 4: 分词器训练 ==================
print("\n" + "=" * 60)
print("🔤 第三阶段：分词器训练")
print("=" * 60)

import sentencepiece as spm

MODEL_PREFIX = "/kaggle/working/chinese_sp"
VOCAB_SIZE = 32000

spm.SentencePieceTrainer.train(
    input=CORPUS_FILE,
    model_prefix=MODEL_PREFIX,
    vocab_size=VOCAB_SIZE,
    model_type="unigram",
    character_coverage=0.9995,
    
    # === 优化后的采样参数 ===
    input_sentence_size=500000,       # 50 万句（已预采样，无需更多）
    shuffle_input_sentence=True,
    max_sentence_length=2048,         # 减少单句长度限制
    
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    pad_piece="<pad>", unk_piece="<unk>",
    bos_piece="<s>", eos_piece="</s>",
    num_threads=os.cpu_count(),
    normalization_rule_name="nmt_nfkc_cf",
    split_by_unicode_script=True,
    split_by_number=True,
)
print(f"✅ 分词器训练完成")

# 转换为 HuggingFace 格式
from transformers import LlamaTokenizerFast

TOKENIZER_DIR = "/kaggle/working/tokenizer"
tokenizer = LlamaTokenizerFast(
    vocab_file=f"{MODEL_PREFIX}.model",
    bos_token="<s>", eos_token="</s>",
    unk_token="<unk>", pad_token="<pad>",
    add_bos_token=False, add_eos_token=True,
)
tokenizer.save_pretrained(TOKENIZER_DIR)
print(f"✅ 词表大小: {len(tokenizer)}")

# ================== Cell 5: 模型初始化 ==================
print("\n" + "=" * 60)
print("🧠 第四阶段：模型初始化")
print("=" * 60)

from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

config = GPT2Config(
    vocab_size=len(tokenizer),
    # === 优化后的模型配置 ===
    n_positions=1024, n_ctx=1024,  # 增大 context（显存优化后可支持）
    n_embd=768, n_layer=6, n_head=12, n_inner=3072,
    activation_function="gelu_new",
    # Dropout: 大数据集可适当降低
    resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,  # 预训练阶段关闭 dropout
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

# === 优化 0: SDPA (Scaled Dot Product Attention) ===
# T4 使用 memory-efficient (CUTLASS) 后端，提供 10-30% 加速
try:
    config._attn_implementation = "sdpa"
    print("✅ 已设置 SDPA attention（Memory-Efficient 后端）")
except Exception as e:
    print(f"⚠️ SDPA 设置失败，使用默认 attention: {e}")

model = GPT2LMHeadModel(config)
print(f"✅ 参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# === 优化 1: Liger Kernel LayerNorm（Triton 加速 +30%）===
try:
    from liger_kernel.transformers import LigerLayerNorm
    import torch.nn as nn
    
    def patch_layernorm(model):
        """将所有 LayerNorm 替换为 Liger Triton 优化版本"""
        patched_count = 0
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.LayerNorm):
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model.get_submodule(parent_name) if parent_name else model
                
                liger_ln = LigerLayerNorm(
                    module.normalized_shape,
                    eps=module.eps
                )
                liger_ln.weight = module.weight
                if module.bias is not None:
                    liger_ln.bias = module.bias
                
                setattr(parent, child_name, liger_ln)
                patched_count += 1
        return model, patched_count
    
    model, count = patch_layernorm(model)
    print(f"✅ 已替换 {count} 个 LayerNorm 为 Liger Triton 版本（+30% 速度）")
except ImportError:
    print("⚠️ liger-kernel 未安装，跳过 LayerNorm 优化")
except Exception as e:
    print(f"⚠️ Liger LayerNorm 替换失败: {e}")

# 保存 lm_head.weight 引用，供 FusedLinearCrossEntropy 使用
# 必须在 torch.compile 之前保存，否则可能无法访问
LM_HEAD_WEIGHT = model.lm_head.weight

# === 优化 2: torch.compile 加速（PyTorch 2.0+）===
try:
    model = torch.compile(model)
    print("✅ 已启用 torch.compile（预计加速 50-100%）")
except Exception as e:
    print(f"⚠️ torch.compile 不可用: {e}")

# ================== Cell 6: 数据处理 ==================
print("\n" + "=" * 60)
print("📦 第五阶段：数据处理")
print("=" * 60)

BLOCK_SIZE = 1024  # 与模型 n_positions 匹配

# 自动检测字段名
text_column = "text" if "text" in dataset.column_names else "completion"

def tokenize_function(examples):
    return tokenizer(examples[text_column], truncation=False, return_attention_mask=False)

tokenized = dataset.map(
    tokenize_function, batched=True, batch_size=1000,
    remove_columns=dataset.column_names, num_proc=4
)

def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = (len(concatenated["input_ids"]) // BLOCK_SIZE) * BLOCK_SIZE
    result = {k: [t[i:i+BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)] for k, t in concatenated.items()}
    result["labels"] = result["input_ids"].copy()
    return result

lm_dataset = tokenized.map(group_texts, batched=True, batch_size=1000, num_proc=4)

# 分割训练集和验证集
split_dataset = lm_dataset.train_test_split(test_size=0.02, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print(f"✅ 训练集: {len(train_dataset)} 样本")
print(f"✅ 验证集: {len(eval_dataset)} 样本")
print(f"   总 Token: {len(lm_dataset) * BLOCK_SIZE / 1e6:.1f}M")

# ================== Cell 7: 训练配置 ==================
print("\n" + "=" * 60)
print("⚙️ 第六阶段：训练配置")
print("=" * 60)

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

MODEL_NAME = "gpt2-chinese-mini-v2"
REPO_ID = f"{HF_USERNAME}/{MODEL_NAME}"

# === 优化后的训练参数 ===
# 基于 Liger Kernel 显存优化，可使用更大 batch size
NUM_EPOCHS = 2       # 625M tokens / 67.5M params ≈ 9 tokens/param，需要多次遍历
BATCH_SIZE = 24      # 显存优化后可增大（原 16）
GRAD_ACCUM = 2       # 减少累积步数，加快更新（原 4）
# Effective batch size: 24 * 2 * 2 GPU = 96 samples = 98K tokens/step

total_steps = (len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM * 2)) * NUM_EPOCHS

training_args = TrainingArguments(
    output_dir=f"/kaggle/working/{MODEL_NAME}",
    overwrite_output_dir=True,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    
    # 学习率调度 - 针对大 batch 优化
    learning_rate=3e-4,              # 大 batch 时适当降低 LR
    warmup_steps=2000,               # 增加 warmup（约 5% 总步数）
    lr_scheduler_type="cosine",
    
    # === 优化器配置 ===
    optim="adamw_bnb_8bit",          # 8-bit AdamW，显存 -75%
    weight_decay=0.1,                # 标准 GPT-2 weight decay
    adam_beta1=0.9, adam_beta2=0.95, # GPT-2 标准配置
    max_grad_norm=1.0,
    
    # 精度
    fp16=True,
    bf16=False,  # T4 不支持 BF16
    
    # === 优化 3: 数据加载优化 ===
    dataloader_num_workers=4,
    dataloader_pin_memory=True,           # 锁页内存，加速传输
    dataloader_prefetch_factor=4,         # 预取 4 个 batch
    dataloader_persistent_workers=True,   # 持久化 worker，避免重启开销
    
    # Checkpoint
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    
    # 日志
    logging_steps=100,
    logging_first_step=True,
    report_to="none",
    
    # Hub
    push_to_hub=True,
    hub_model_id=REPO_ID,
    hub_strategy="checkpoint",
    
    # DDP
    ddp_find_unused_parameters=False,
    seed=42,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# === 自定义回调：训练时生成样本文本 ===
from transformers import TrainerCallback

class GenerationCallback(TrainerCallback):
    """每次评估时生成样本文本，直观观察模型进步"""
    
    def __init__(self, tokenizer, prompts=None):
        self.tokenizer = tokenizer
        self.prompts = prompts or ["中国的历史", "人工智能是", "知乎上有人问"]
    
    def on_evaluate(self, args, state, control, model, **kwargs):
        print("\n" + "=" * 50)
        print(f"📝 Step {state.global_step} - 生成样本:")
        print("=" * 50)
        
        model.eval()
        device = next(model.parameters()).device
        
        for prompt in self.prompts:
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=True,
                        temperature=0.8,
                        top_k=50,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"  [{prompt}] → {generated}")
            except Exception as e:
                print(f"  [{prompt}] → 生成失败: {e}")
        
        print("=" * 50 + "\n")
        model.train()

# === 自定义 Trainer：使用 Liger FusedLinearCrossEntropy ===
# 显存减少 60-80%，速度提升 50-100%（大词表效果更明显）
USE_FUSED_CE = True  # 设置为 False 可禁用

try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    
    class LigerTrainer(Trainer):
        """使用 FusedLinearCrossEntropy 的自定义 Trainer
        
        原理：将 LM Head (Linear) + CrossEntropy Loss 融合为一个 kernel
        优势：
        - 不需要 materialize 完整的 logits tensor (vocab_size × batch × seq)
        - 分块计算，显存占用大幅降低
        - 减少内存访问，GPU 利用率提高
        """
        
        def __init__(self, *args, lm_head_weight=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.fused_loss_fn = LigerFusedLinearCrossEntropyLoss()
            self.lm_head_weight = lm_head_weight
            print("✅ LigerTrainer: 已启用 FusedLinearCrossEntropy")
        
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            """重写 loss 计算，使用 Fused kernel"""
            labels = inputs.pop("labels")
            
            # 获取 hidden states（不经过 lm_head）
            # 处理 torch.compile 包装的模型
            base_model = getattr(model, '_orig_mod', model)
            transformer = getattr(base_model, 'transformer', None)
            if transformer is None:
                # Fallback: 使用标准 forward 然后忽略 logits
                raise RuntimeError("无法访问 model.transformer，请禁用 USE_FUSED_CE")
            
            outputs = transformer(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
            hidden_states = outputs.last_hidden_state  # [batch, seq, hidden]
            
            # Shift: LM 任务中 labels 需要左移一位
            # hidden_states[:-1] 预测 labels[1:]
            shift_hidden = hidden_states[..., :-1, :].contiguous()  # [batch, seq-1, hidden]
            shift_labels = labels[..., 1:].contiguous()  # [batch, seq-1]
            
            # Flatten for FusedLinearCrossEntropy
            batch_size, seq_len, hidden_size = shift_hidden.shape
            shift_hidden = shift_hidden.view(-1, hidden_size)  # [batch*seq-1, hidden]
            shift_labels = shift_labels.view(-1)  # [batch*seq-1]
            
            # 使用 FusedLinearCrossEntropy
            # 它接受: (weight, input, target)
            loss = self.fused_loss_fn(
                self.lm_head_weight,  # [vocab_size, hidden_size]
                shift_hidden,          # [batch*seq-1, hidden_size]
                shift_labels           # [batch*seq-1]
            )
            
            if return_outputs:
                # 构造一个假的 outputs 对象用于其他回调
                from transformers.modeling_outputs import CausalLMOutputWithPast
                fake_outputs = CausalLMOutputWithPast(loss=loss)
                return loss, fake_outputs
            return loss
    
    if USE_FUSED_CE:
        TrainerClass = LigerTrainer
        trainer_kwargs = {"lm_head_weight": LM_HEAD_WEIGHT}  # 使用预保存的权重
        print("✅ 将使用 LigerTrainer + FusedLinearCrossEntropy")
    else:
        TrainerClass = Trainer
        trainer_kwargs = {}
        print("⚠️ FusedLinearCrossEntropy 已禁用，使用标准 Trainer")
        
except ImportError:
    print("⚠️ liger-kernel 未安装，使用标准 Trainer")
    TrainerClass = Trainer
    trainer_kwargs = {}
except Exception as e:
    print(f"⚠️ LigerTrainer 初始化失败: {e}")
    print("   使用标准 Trainer 作为 fallback")
    TrainerClass = Trainer
    trainer_kwargs = {}

generation_callback = GenerationCallback(
    tokenizer=tokenizer,
    prompts=["中国的历史", "人工智能是", "知乎上有人问", "科学技术的发展"]
)

trainer = TrainerClass(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[generation_callback],
    **trainer_kwargs,
)

print(f"✅ 训练配置完成")
print(f"   预估步数: {total_steps}")
print(f"   Warmup: {training_args.warmup_steps}")
print(f"   ✅ 已启用生成回调（每 {training_args.eval_steps} 步测试 prompt）")
if TrainerClass.__name__ == "LigerTrainer":
    print(f"   ✅ FusedLinearCrossEntropy: 显存 -60~80%, 速度 +50~100%")

# ================== Cell 8: 开始训练 ==================
print("\n" + "=" * 60)
print("🚀 第七阶段：开始训练")
print("=" * 60)

import time
print(f"⏰ 开始: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📍 模型: https://huggingface.co/{REPO_ID}")
print("\n📈 Loss 参考: 初始 ~10, 目标 <3.5")
print("=" * 60)

try:
    result = trainer.train()
    print(f"\n✅ 完成! 最终 Loss: {result.training_loss:.4f}")
except KeyboardInterrupt:
    print("\n⚠️ 中断，保存中...")
except Exception as e:
    print(f"\n❌ 错误: {e}")
finally:
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    try:
        trainer.push_to_hub(commit_message="Training complete")
        print(f"✅ 已上传至 HuggingFace")
    except:
        print("⚠️ 上传失败，请手动上传")

print(f"\n🎉 模型: https://huggingface.co/{REPO_ID}")

# ================== Cell 9: 生成测试 ==================
print("\n" + "=" * 60)
print("🔬 生成测试")
print("=" * 60)

model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

prompts = ["中国的历史", "人工智能是", "科学技术", "知乎上有人问"]

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_length=80, do_sample=True,
            temperature=0.8, top_k=50, top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    print(f"\n[{prompt}]")
    print(f"  → {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
