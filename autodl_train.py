#!/usr/bin/env python3
"""
GPT-2 Chinese Pretraining Script for AutoDL (5090/4090)
========================================================
优化版本，适配 AutoDL 环境和高端消费级显卡

与 Kaggle 版本的主要区别：
1. 使用 DDP (DistributedDataParallel) 而非 DataParallel
2. 启用 torch.compile 加速 (+50-100%)
3. 路径改为 AutoDL 标准路径 (/root/autodl-tmp/)
4. 批量大小针对 32GB 显存优化
5. 支持命令行参数配置

运行方式（单卡）：
    python autodl_train.py

运行方式（多卡 DDP）：
    torchrun --nproc_per_node=2 autodl_train.py

环境安装：
    pip install torch transformers datasets accelerate sentencepiece tokenizers bitsandbytes liger-kernel huggingface_hub flash-attn
"""

import os
import sys
import time
import gc
import json
import random
import argparse
import multiprocessing
from pathlib import Path
from itertools import chain

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


# ============================================================
# 版本兼容性检查
# ============================================================

def check_environment():
    """检查环境版本兼容性"""
    print("=" * 60)
    print("🔍 环境检查")
    print("=" * 60)
    
    # Python 版本
    py_version = sys.version_info
    print(f"   Python: {py_version.major}.{py_version.minor}.{py_version.micro}", end="")
    if py_version.major >= 3 and py_version.minor >= 10:
        print(" ✅")
    else:
        print(" ⚠️ 推荐 3.10+")
    
    # PyTorch 版本
    print(f"   PyTorch: {torch.__version__}", end="")
    torch_major = int(torch.__version__.split('.')[0])
    if torch_major >= 2:
        print(" ✅")
    else:
        print(" ⚠️ 推荐 2.0+")
    
    # CUDA
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        print(f"   CUDA: {cuda_version}", end="")
        if cuda_version and float(cuda_version.split('.')[0]) >= 12:
            print(" ✅ (5090 需要 CUDA 12.8+)")
        else:
            print(f" ⚠️ 5090 推荐 CUDA 12.8+")
        
        # GPU 信息
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({gpu_mem:.0f}GB)")
            
            # 检测 5090/4090
            if "5090" in gpu_name:
                print(f"         → Blackwell 架构，BF16/FlashAttn2 推荐 ✅")
            elif "4090" in gpu_name:
                print(f"         → Ada 架构，BF16 推荐 ✅")
    else:
        print("   CUDA: ❌ 未检测到 GPU!")
        sys.exit(1)
    
    # Transformers 版本
    try:
        import transformers
        print(f"   Transformers: {transformers.__version__}", end="")
        tf_major = int(transformers.__version__.split('.')[0])
        if tf_major >= 4:
            print(" ✅")
        else:
            print(" ⚠️ 推荐 4.36+")
    except ImportError:
        print("   Transformers: ❌ 未安装!")
        sys.exit(1)
    
    # Flash Attention
    try:
        import flash_attn
        print(f"   Flash Attention: {flash_attn.__version__} ✅")
    except ImportError:
        print("   Flash Attention: ⚠️ 未安装 (将使用 SDPA)")
    
    # BF16 支持
    if torch.cuda.is_bf16_supported():
        print("   BF16: ✅ 支持")
    else:
        print("   BF16: ❌ 不支持 (将使用 FP16)")
    
    print("=" * 60)


# ============================================================
# 配置参数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="GPT-2 Chinese Pretraining")
    
    # 路径配置
    parser.add_argument("--work_dir", type=str, default="/root/autodl-tmp/gpt2-chinese",
                       help="工作目录")
    parser.add_argument("--cache_dir", type=str, default="/root/autodl-tmp/cache",
                       help="HuggingFace 缓存目录")
    
    # 模型配置
    parser.add_argument("--vocab_size", type=int, default=32000, help="词表大小")
    parser.add_argument("--n_positions", type=int, default=1024, help="最大序列长度")
    parser.add_argument("--n_embd", type=int, default=768, help="隐藏层维度")
    parser.add_argument("--n_layer", type=int, default=6, help="层数")
    parser.add_argument("--n_head", type=int, default=12, help="注意力头数")
    
    # 训练配置
    parser.add_argument("--batch_size", type=int, default=48, 
                       help="每 GPU 批量大小 (5090 32GB 可用 48, 4090 24GB 用 32)")
    parser.add_argument("--gradient_accumulation", type=int, default=2,
                       help="梯度累积步数")
    parser.add_argument("--num_epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="学习率")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="预热步数")
    
    # === 5090/Blackwell 优化配置 ===
    parser.add_argument("--use_compile", action="store_true", default=True,
                       help="使用 torch.compile 加速")
    parser.add_argument("--compile_mode", type=str, default="max-autotune",
                       choices=["default", "reduce-overhead", "max-autotune"],
                       help="torch.compile 模式 (5090 推荐 max-autotune)")
    parser.add_argument("--use_bf16", action="store_true", default=True,
                       help="使用 BF16 (5090/4090 推荐，比 FP16 更稳定)")
    parser.add_argument("--use_flash_attn", action="store_true", default=True,
                       help="使用 Flash Attention 2")
    parser.add_argument("--use_liger", action="store_true", default=True,
                       help="使用 Liger Kernel 优化")
    parser.add_argument("--use_8bit_adam", action="store_true", default=True,
                       help="使用 8-bit AdamW")
    
    # HuggingFace 配置
    parser.add_argument("--hf_token", type=str, default=None,
                       help="HuggingFace Token (或设置 HF_TOKEN 环境变量)")
    parser.add_argument("--push_to_hub", action="store_true", default=True,
                       help="训练完成后上传到 Hub")
    parser.add_argument("--hub_model_id", type=str, default=None,
                       help="Hub 模型 ID (默认: 用户名/gpt2-chinese-mini)")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--resume", action="store_true", help="从 checkpoint 恢复")
    
    return parser.parse_args()


def setup_distributed():
    """设置分布式训练环境"""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return local_rank, world_size, True
    else:
        return 0, 1, False


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(local_rank):
    """是否为主进程"""
    return local_rank == 0


def print_rank0(msg, local_rank=0):
    """只在主进程打印"""
    if is_main_process(local_rank):
        print(msg)


# ============================================================
# 数据加载
# ============================================================

def load_datasets(args, local_rank):
    """加载并合并数据集"""
    from datasets import load_dataset, concatenate_datasets
    
    print_rank0("📥 加载数据集...", local_rank)
    
    # 加载维基百科
    print_rank0("   [1/2] 加载维基百科...", local_rank)
    wiki = load_dataset(
        "pleisto/wikipedia-cn-20230720-filtered",
        split="train",
        cache_dir=args.cache_dir,
    )
    print_rank0(f"   ✅ 维基百科: {len(wiki)} 条", local_rank)
    
    # 加载知乎
    print_rank0("   [2/2] 加载知乎...", local_rank)
    zhihu = load_dataset(
        "wangrui6/Zhihu-KOL",
        split="train",
        cache_dir=args.cache_dir,
    )
    print_rank0(f"   ✅ 知乎: {len(zhihu)} 条", local_rank)
    
    # 统一字段名
    def process_wiki(example):
        return {"text": example["completion"]}
    
    def process_zhihu(example):
        return {"text": f"{example['INSTRUCTION']}\n{example['RESPONSE']}"}
    
    wiki_processed = wiki.map(process_wiki, remove_columns=wiki.column_names, num_proc=4)
    zhihu_processed = zhihu.map(process_zhihu, remove_columns=zhihu.column_names, num_proc=4)
    
    # 合并
    dataset = concatenate_datasets([wiki_processed, zhihu_processed])
    dataset = dataset.shuffle(seed=args.seed)
    
    print_rank0(f"✅ 数据集合并完成: {len(dataset)} 条", local_rank)
    return dataset


# ============================================================
# 分词器
# ============================================================

def train_or_load_tokenizer(args, dataset, local_rank):
    """训练或加载分词器"""
    import sentencepiece as spm
    from transformers import LlamaTokenizerFast, AutoTokenizer
    
    tokenizer_dir = Path(args.work_dir) / "tokenizer"
    sp_model_path = Path(args.work_dir) / "chinese_sp.model"
    
    # 缓存检测
    if (tokenizer_dir / "tokenizer.json").exists():
        print_rank0("✅ 检测到已有分词器，从缓存加载", local_rank)
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        print_rank0(f"   词表大小: {len(tokenizer)}", local_rank)
        return tokenizer
    
    # 只在主进程训练分词器
    if is_main_process(local_rank):
        print_rank0("🔤 训练 SentencePiece 分词器...", local_rank)
        
        # 采样数据
        sample_size = min(500000, len(dataset))
        random.seed(args.seed)
        indices = random.sample(range(len(dataset)), sample_size)
        
        corpus_file = Path(args.work_dir) / "corpus.txt"
        with open(corpus_file, "w", encoding="utf-8") as f:
            for idx in indices:
                text = dataset[idx]["text"].strip()
                if 50 < len(text) < 5000:
                    f.write(text + "\n")
        
        # 训练
        spm.SentencePieceTrainer.train(
            input=str(corpus_file),
            model_prefix=str(sp_model_path).replace(".model", ""),
            vocab_size=args.vocab_size,
            model_type="unigram",
            character_coverage=0.9995,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            pad_piece="<pad>", unk_piece="<unk>",
            bos_piece="<s>", eos_piece="</s>",
            num_threads=os.cpu_count() or 4,
        )
        
        # 转换为 HuggingFace 格式
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        tokenizer = LlamaTokenizerFast(
            vocab_file=str(sp_model_path),
            bos_token="<s>", eos_token="</s>",
            unk_token="<unk>", pad_token="<pad>",
            add_bos_token=False, add_eos_token=True,
        )
        tokenizer.save_pretrained(str(tokenizer_dir))
        
        # 上传到 Hub
        if args.push_to_hub and args.hf_token:
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=args.hf_token)
                user = api.whoami()["name"]
                tokenizer.push_to_hub(f"{user}/chinese-sp-32k", token=args.hf_token)
                print_rank0(f"🚀 分词器已上传至 HuggingFace Hub", local_rank)
            except Exception as e:
                print_rank0(f"⚠️ 上传失败: {e}", local_rank)
        
        print_rank0(f"✅ 分词器训练完成，词表大小: {len(tokenizer)}", local_rank)
    
    # 等待主进程完成
    if dist.is_initialized():
        dist.barrier()
    
    # 所有进程加载分词器
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    return tokenizer


# ============================================================
# 数据预处理
# ============================================================

def prepare_dataset(args, dataset, tokenizer, local_rank):
    """Tokenize 和 Packing"""
    from datasets import load_from_disk
    
    lm_dataset_path = Path(args.work_dir) / "lm_dataset"
    
    # 缓存检测
    if lm_dataset_path.exists():
        print_rank0("✅ 检测到已处理的数据集，从缓存加载", local_rank)
        lm_dataset = load_from_disk(str(lm_dataset_path))
        print_rank0(f"   样本数: {len(lm_dataset)}", local_rank)
        return lm_dataset
    
    # 只在主进程处理
    if is_main_process(local_rank):
        print_rank0("🔄 Tokenize 数据...", local_rank)
        
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=False,
            )
        
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=5000,
            remove_columns=dataset.column_names,
            num_proc=multiprocessing.cpu_count(),
            desc="Tokenizing",
        )
        
        print_rank0("📦 Packing 数据...", local_rank)
        block_size = args.n_positions
        
        def group_texts(examples):
            concatenated = {k: list(chain.from_iterable(examples[k])) for k in examples.keys()}
            total_length = len(concatenated["input_ids"])
            total_length = (total_length // block_size) * block_size
            
            result = {
                k: [t[i:i+block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result
        
        lm_dataset = tokenized.map(
            group_texts,
            batched=True,
            batch_size=5000,
            num_proc=multiprocessing.cpu_count(),
            desc="Packing",
        )
        
        # 保存缓存
        lm_dataset.save_to_disk(str(lm_dataset_path))
        print_rank0(f"✅ 数据处理完成: {len(lm_dataset)} 样本", local_rank)
    
    # 等待主进程
    if dist.is_initialized():
        dist.barrier()
    
    lm_dataset = load_from_disk(str(lm_dataset_path))
    return lm_dataset


# ============================================================
# 模型创建
# ============================================================

def create_model(args, tokenizer, local_rank):
    """创建并优化模型"""
    from transformers import GPT2Config, GPT2LMHeadModel
    
    print_rank0("🏗️ 创建模型...", local_rank)
    
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=args.n_positions,
        n_ctx=args.n_positions,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_inner=args.n_embd * 4,
        activation_function="gelu_new",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # === 5090/Blackwell 优化：Flash Attention 2 ===
    if args.use_flash_attn:
        try:
            # 尝试使用 Flash Attention 2
            config._attn_implementation = "flash_attention_2"
            print_rank0("✅ Flash Attention 2 已启用", local_rank)
        except Exception:
            # 回退到 SDPA
            config._attn_implementation = "sdpa"
            print_rank0("⚠️ Flash Attention 2 不可用，使用 SDPA", local_rank)
    else:
        config._attn_implementation = "sdpa"
    
    model = GPT2LMHeadModel(config)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print_rank0(f"✅ 模型创建完成: {param_count:.1f}M 参数", local_rank)
    
    # Liger Kernel 优化
    if args.use_liger:
        try:
            from liger_kernel.transformers import LigerLayerNorm
            import torch.nn as nn
            
            count = 0
            for name, module in list(model.named_modules()):
                if isinstance(module, nn.LayerNorm):
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    parent = model.get_submodule(parent_name) if parent_name else model
                    liger_ln = LigerLayerNorm(module.normalized_shape, eps=module.eps)
                    liger_ln.weight = module.weight
                    if module.bias is not None:
                        liger_ln.bias = module.bias
                    setattr(parent, child_name, liger_ln)
                    count += 1
            print_rank0(f"✅ Liger LayerNorm: 替换了 {count} 个", local_rank)
        except ImportError:
            print_rank0("⚠️ liger-kernel 未安装，跳过优化", local_rank)
    
    # Gradient Checkpointing
    model.gradient_checkpointing_enable()
    print_rank0("✅ Gradient Checkpointing 已启用", local_rank)
    
    # === 5090/Blackwell 优化：torch.compile max-autotune ===
    if args.use_compile:
        try:
            # max-autotune 模式会花更多时间编译，但运行更快
            model = torch.compile(model, mode=args.compile_mode)
            print_rank0(f"✅ torch.compile 已启用 (mode={args.compile_mode})", local_rank)
        except Exception as e:
            print_rank0(f"⚠️ torch.compile 失败: {e}", local_rank)
    
    return model


# ============================================================
# 训练
# ============================================================

def train(args):
    """主训练函数"""
    # === 1. 版本兼容性检查 ===
    check_environment()
    
    # 分布式设置
    local_rank, world_size, is_distributed = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    print_rank0("=" * 60, local_rank)
    print_rank0("🚀 GPT-2 Chinese Pretraining (AutoDL 5090)", local_rank)
    print_rank0("=" * 60, local_rank)
    print_rank0(f"   GPU: {torch.cuda.get_device_name(local_rank)}", local_rank)
    print_rank0(f"   分布式: {is_distributed} (world_size={world_size})", local_rank)
    print_rank0(f"   工作目录: {args.work_dir}", local_rank)
    
    # 创建工作目录
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    
    # HuggingFace 登录
    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)
    elif os.environ.get("HF_TOKEN"):
        args.hf_token = os.environ["HF_TOKEN"]
        from huggingface_hub import login
        login(token=args.hf_token)
    
    # 加载数据
    dataset = load_datasets(args, local_rank)
    
    # 分词器
    tokenizer = train_or_load_tokenizer(args, dataset, local_rank)
    
    # 数据预处理
    lm_dataset = prepare_dataset(args, dataset, tokenizer, local_rank)
    
    # 创建模型
    model = create_model(args, tokenizer, local_rank)
    model = model.to(device)
    
    # DDP 包装
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])
        print_rank0("✅ DDP 已启用", local_rank)
    
    # 训练参数
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    
    effective_batch = args.batch_size * args.gradient_accumulation * world_size
    print_rank0(f"\n📊 训练配置:", local_rank)
    print_rank0(f"   Batch/GPU: {args.batch_size}", local_rank)
    print_rank0(f"   有效 Batch: {effective_batch}", local_rank)
    print_rank0(f"   Epochs: {args.num_epochs}", local_rank)
    
    training_args = TrainingArguments(
        output_dir=str(Path(args.work_dir) / "checkpoints"),
        overwrite_output_dir=True,
        
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        
        optim="adamw_bnb_8bit" if args.use_8bit_adam else "adamw_torch",
        weight_decay=0.1,
        max_grad_norm=1.0,
        
        # === 5090/Blackwell 优化：使用 BF16 ===
        # BF16 比 FP16 数值范围更大，训练更稳定
        fp16=not args.use_bf16,
        bf16=args.use_bf16 and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=False,  # 已手动启用
        
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        prediction_loss_only=True,
        
        logging_steps=10,
        logging_first_step=True,
        report_to="none",
        
        push_to_hub=args.push_to_hub and is_main_process(local_rank),
        hub_model_id=args.hub_model_id,
        hub_strategy="checkpoint",
        
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        seed=args.seed,
        
        # DDP 设置
        ddp_find_unused_parameters=False,
        local_rank=local_rank if is_distributed else -1,
    )
    
    # Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # === Callback: 生成测试（评估/保存时测试 prompt）===
    from transformers import TrainerCallback
    
    class GenerationCallback(TrainerCallback):
        """在评估/保存时测试生成质量"""
        def __init__(self, tokenizer, prompts=None):
            self.tokenizer = tokenizer
            self.prompts = prompts or ["中国的历史", "人工智能是", "今天天气"]
        
        def on_evaluate(self, args, state, control, model, **kwargs):
            print("\n" + "=" * 50)
            print(f"📝 Step {state.global_step} - 生成样本测试:")
            print("=" * 50)
            
            # 处理 DDP 包装
            eval_model = model.module if hasattr(model, 'module') else model
            eval_model.eval()
            device = next(eval_model.parameters()).device
            
            for prompt in self.prompts:
                try:
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = eval_model.generate(
                            **inputs, max_new_tokens=50,
                            do_sample=True, temperature=0.8, top_k=50,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                    generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    print(f"   [{prompt}] → {generated[:80]}...")
                except Exception as e:
                    print(f"   [{prompt}] → 生成失败: {e}")
            print("=" * 50 + "\n")
    
    # === Callback: 前10步详细日志 ===
    class DetailedLoggingCallback(TrainerCallback):
        """前10步详细日志，之后每100步打印一次"""
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                step = state.global_step
                loss = logs.get("loss", logs.get("eval_loss", None))
                lr = logs.get("learning_rate", 0)
                
                # 前10步或每100步打印详细信息
                if step <= 10 or step % 100 == 0:
                    if loss is not None:
                        print(f"📊 Step {step}: loss={loss:.4f}, lr={lr:.2e}")
    
    # 创建回调
    generation_callback = GenerationCallback(tokenizer) if is_main_process(local_rank) else None
    logging_callback = DetailedLoggingCallback()
    
    callbacks = [logging_callback]
    if generation_callback:
        callbacks.append(generation_callback)
    
    # 创建 Trainer
    # 如果是 DDP，需要解包模型
    train_model = model.module if is_distributed else model
    
    trainer = Trainer(
        model=train_model,
        args=training_args,
        train_dataset=lm_dataset,
        eval_dataset=lm_dataset.select(range(min(1000, len(lm_dataset)))),
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    # 开始训练
    print_rank0("\n" + "=" * 60, local_rank)
    print_rank0("🎯 开始训练...", local_rank)
    print_rank0("=" * 60, local_rank)
    
    try:
        trainer.train(resume_from_checkpoint=args.resume)
        print_rank0("\n✅ 训练完成!", local_rank)
        
        # 保存最终模型
        if is_main_process(local_rank):
            final_path = Path(args.work_dir) / "final_model"
            trainer.save_model(str(final_path))
            tokenizer.save_pretrained(str(final_path))
            
            # 上传到 Hub
            if args.push_to_hub and args.hf_token:
                print_rank0("📤 上传到 HuggingFace Hub...", local_rank)
                from huggingface_hub import HfApi
                api = HfApi(token=args.hf_token)
                user = api.whoami()["name"]
                repo_id = args.hub_model_id or f"{user}/gpt2-chinese-mini"
                
                api.upload_folder(
                    folder_path=str(final_path),
                    repo_id=repo_id,
                    commit_message="Training complete",
                )
                print_rank0(f"🎉 模型已上传至: https://huggingface.co/{repo_id}", local_rank)
                
    except KeyboardInterrupt:
        print_rank0("\n⚠️ 训练被中断，保存 checkpoint...", local_rank)
        trainer.save_model()
    
    finally:
        cleanup_distributed()


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    args = parse_args()
    train(args)
