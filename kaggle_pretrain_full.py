# Kaggle Mini GPT-2 中文预训练 - 完整 Notebook
# ===================================================
# 复制整个文件到 Kaggle Notebook 运行
# 设置: Accelerator = GPU T4 x2, Internet = On
# ===================================================

# ================== Cell 1: 环境设置 ==================
print("=" * 60)
print("🔧 第一阶段：环境配置")
print("=" * 60)

# GPU 检查
import subprocess
subprocess.run(["nvidia-smi"], check=True)

# 安装依赖
subprocess.run([
    "pip", "install", "-q",
    "transformers==4.37.2",
    "datasets==2.16.1",
    "accelerate==0.26.1",
    "huggingface_hub==0.20.3",
    "sentencepiece==0.1.99",
    "tokenizers==0.15.1",
], check=True)

import torch
print(f"\n✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA: {torch.cuda.is_available()}")
print(f"✅ GPU 数量: {torch.cuda.device_count()}")

# ================== Cell 2: HuggingFace 登录 ==================
from huggingface_hub import notebook_login, whoami

notebook_login()  # 需要输入 token

try:
    HF_USERNAME = whoami()["name"]
    print(f"✅ 登录成功: {HF_USERNAME}")
except:
    HF_USERNAME = "YOUR_USERNAME"  # 手动填写
    print(f"⚠️ 请手动设置 HF_USERNAME")

# ================== Cell 3: 数据准备 ==================
print("\n" + "=" * 60)
print("📥 第二阶段：数据准备")
print("=" * 60)

from datasets import load_dataset
import os

# 加载数据 (调整比例控制数据量)
DATA_RATIO = "10%"  # 可选: 10%, 50%, 100%
dataset = load_dataset(
    "pleisto/wikipedia-cn-20230720-filtered",
    split=f"train[:{DATA_RATIO}]",
    trust_remote_code=True
)
print(f"✅ 数据样本: {len(dataset)}")

# 导出纯文本
CORPUS_FILE = "/kaggle/working/wiki_corpus.txt"
with open(CORPUS_FILE, "w", encoding="utf-8") as f:
    for item in dataset:
        text = item["completion"].strip()
        if len(text) > 50:
            f.write(text + "\n")

print(f"✅ 语料导出: {os.path.getsize(CORPUS_FILE) / 1e6:.1f} MB")

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

# 测试
test = "人工智能是未来的发展方向"
print(f"   测试: {test}")
print(f"   分词: {tokenizer.tokenize(test)}")

# ================== Cell 5: 模型初始化 ==================
print("\n" + "=" * 60)
print("🧠 第四阶段：模型初始化")
print("=" * 60)

from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=512, n_ctx=512,
    n_embd=768, n_layer=6, n_head=12, n_inner=3072,
    activation_function="gelu_new",
    resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

model = GPT2LMHeadModel(config)
print(f"✅ 参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# ================== Cell 6: 数据处理 ==================
print("\n" + "=" * 60)
print("📦 第五阶段：数据处理")
print("=" * 60)

BLOCK_SIZE = 512

def tokenize_function(examples):
    return tokenizer(examples["completion"], truncation=False, return_attention_mask=False)

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
print(f"✅ 训练样本: {len(lm_dataset)} 个 512-token 块")
print(f"   总 Token: {len(lm_dataset) * BLOCK_SIZE / 1e6:.1f}M")

# ================== Cell 7: 训练配置 ==================
print("\n" + "=" * 60)
print("⚙️ 第六阶段：训练配置")
print("=" * 60)

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

MODEL_NAME = "gpt2-chinese-mini"
REPO_ID = f"{HF_USERNAME}/{MODEL_NAME}"

NUM_EPOCHS = 10
BATCH_SIZE = 16
GRAD_ACCUM = 2

total_steps = (len(lm_dataset) // (BATCH_SIZE * GRAD_ACCUM * 2)) * NUM_EPOCHS

training_args = TrainingArguments(
    output_dir=f"/kaggle/working/{MODEL_NAME}",
    overwrite_output_dir=True,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    
    # 学习率调度
    learning_rate=6e-4,
    warmup_steps=min(500, total_steps // 10),
    lr_scheduler_type="cosine",
    
    # 优化器
    optim="adamw_torch",
    weight_decay=0.1,
    adam_beta1=0.9, adam_beta2=0.95,
    max_grad_norm=1.0,
    
    # 精度
    fp16=True,
    dataloader_num_workers=4,
    
    # Checkpoint
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,
    
    # 日志
    logging_steps=50,
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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print(f"✅ 训练配置完成")
print(f"   预估步数: {total_steps}")
print(f"   Warmup: {training_args.warmup_steps}")

# ================== Cell 8: 开始训练 ==================
print("\n" + "=" * 60)
print("🚀 第七阶段：开始训练")
print("=" * 60)

import time
print(f"⏰ 开始: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📍 模型: https://huggingface.co/{REPO_ID}")
print("\n📈 Loss 参考: 初始 ~10, 目标 <4.0")
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
        trainer.push_to_hub(commit_message="Training checkpoint")
        print(f"✅ 已上传至 HuggingFace")
    except:
        print("⚠️ 上传失败，请手动上传")

print(f"\n🎉 模型: https://huggingface.co/{REPO_ID}")

# ================== Cell 9: 生成测试 ==================
print("\n" + "=" * 60)
print("🔬 生成测试")
print("=" * 60)

model.eval()
prompts = ["中国的历史", "人工智能是", "科学技术"]

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_length=80, do_sample=True,
            temperature=0.8, top_k=50, top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    print(f"\n提示: {prompt}")
    print(f"生成: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
