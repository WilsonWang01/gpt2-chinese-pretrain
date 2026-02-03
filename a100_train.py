#!/usr/bin/env python3
"""
GPT-2 Chinese Pretraining Script for A100/A800 (v2 - FIXED)
============================================================
针对 NVIDIA A100/A800 (Ampere 架构) 深度优化
已修复所有已知问题，生产环境就绪

修复记录:
- [FIX-1] 移除 DDP+compile+gradient_checkpointing 冲突
- [FIX-2] 使用 Trainer 内置 DDP，避免 model.module 问题
- [FIX-3] 移除手动 DDP 包装
- [FIX-4] bitsandbytes fallback 逻辑
- [FIX-5] 增加 dataloader_num_workers
- [FIX-6] 启用 dataloader_persistent_workers
- [FIX-7] SentencePiece 训练加速
- [FIX-8] 数据集并行下载
- [FIX-9] 降低 map batch_size
- [FIX-10-15] 其他优化

运行方式（单卡）：
    python a100_train.py

运行方式（多卡 DDP）：
    torchrun --nproc_per_node=2 a100_train.py

环境安装：
    pip install torch transformers datasets accelerate sentencepiece tokenizers bitsandbytes liger-kernel huggingface_hub
    pip install flash-attn --no-build-isolation  # 可选
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
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.distributed as dist

# [FIX-15] 设置 HF 缓存环境变量（在任何 HuggingFace 导入之前）
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/root/autodl-tmp/cache")


# ============================================================
# A100 专属优化
# ============================================================

def enable_a100_optimizations(local_rank=0):
    """启用 A100 专属优化 [FIX-10: 只在主进程打印]"""
    # === 1. TF32 优化 (A100 独有) ===
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # === 2. cuDNN 自动调优 ===
    torch.backends.cudnn.benchmark = True
    
    # === 3. 内存分配优化 ===
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    if local_rank == 0:
        print("=" * 60)
        print("⚡ A100/Ampere 专属优化")
        print("=" * 60)
        print("   ✅ TF32: 已启用 (矩阵乘法加速 2-6x)")
        print("   ✅ cuDNN Autotuner: 已启用")
        print("   ✅ CUDA 内存分配: 已优化")
        print("=" * 60)


# ============================================================
# 环境检查
# ============================================================

def check_environment(local_rank=0):
    """检查 A100 环境兼容性"""
    if local_rank != 0:
        # [FIX] 非主进程返回 False 而非 None，确保 Flash Attention 在 DDP 时正确启用
        return False
    
    print("=" * 60)
    print("🔍 A100 环境检查")
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
        cuda_major = float(cuda_version.split('.')[0]) if cuda_version else 0
        if cuda_major >= 11:
            print(" ✅")
        else:
            print(" ⚠️ 推荐 CUDA 11.8+")
        
        # GPU 信息
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({gpu_mem:.0f}GB)")
            
            if "A100" in gpu_name or "A800" in gpu_name:
                if gpu_mem >= 75:
                    print(f"         → 80GB 版本 (推荐 batch_size=64+)")
                else:
                    print(f"         → 40GB 版本 (推荐 batch_size=48)")
    else:
        print("   CUDA: ❌ 未检测到 GPU!")
        sys.exit(1)
    
    # Transformers
    try:
        import transformers
        print(f"   Transformers: {transformers.__version__} ✅")
    except ImportError:
        print("   Transformers: ❌ 未安装!")
        sys.exit(1)
    
    # [FIX-4] bitsandbytes 检测
    try:
        import bitsandbytes as bnb
        # 测试是否真正支持 GPU
        bnb.optim.Adam8bit([torch.zeros(1, device='cuda', requires_grad=True)])
        print(f"   bitsandbytes: ✅ GPU 支持")
    except Exception as e:
        print(f"   bitsandbytes: ⚠️ 无 GPU 支持 (将使用 fused AdamW)")
    
    # [FIX-13] Flash Attention 检测 - 检查 GPU 架构
    flash_attn_available = False
    if torch.cuda.get_device_capability()[0] >= 8:  # Ampere 或更新
        try:
            import flash_attn
            print(f"   Flash Attention: {flash_attn.__version__} ✅")
            flash_attn_available = True
        except ImportError:
            print("   Flash Attention: ⚠️ 未安装 (将使用 SDPA)")
    else:
        print("   Flash Attention: ⚠️ GPU 不支持 (需要 Ampere+)")
    
    # BF16/TF32 支持
    if torch.cuda.is_bf16_supported():
        print("   BF16: ✅ 支持")
    if torch.cuda.get_device_capability()[0] >= 8:
        print("   TF32: ✅ 支持")
    
    print("=" * 60)
    return flash_attn_available


# ============================================================
# 配置参数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="GPT-2 Chinese Pretraining (A100 优化版 v2)")
    
    # 路径配置
    parser.add_argument("--work_dir", type=str, default="/root/autodl-tmp/gpt2-chinese",
                       help="工作目录")
    parser.add_argument("--cache_dir", type=str, default="/root/autodl-tmp/cache",
                       help="HuggingFace 缓存目录")
    
    # 模型配置 (Tensor Core 对齐: 维度为 8 的倍数)
    parser.add_argument("--vocab_size", type=int, default=32000, help="词表大小")
    parser.add_argument("--n_positions", type=int, default=1024, help="最大序列长度")
    parser.add_argument("--n_embd", type=int, default=768, help="隐藏层维度")
    parser.add_argument("--n_layer", type=int, default=8, help="层数")
    parser.add_argument("--n_head", type=int, default=12, help="注意力头数")
    
    # 训练配置
    parser.add_argument("--batch_size", type=int, default=48, 
                       help="每 GPU 批量大小 (A100-40GB=48, A100-80GB=64)")
    parser.add_argument("--gradient_accumulation", type=int, default=2,
                       help="梯度累积步数")
    parser.add_argument("--num_epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="学习率")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="预热步数")
    
    # A100 优化配置
    parser.add_argument("--use_tf32", action="store_true", default=True,
                       help="使用 TF32 加速")
    parser.add_argument("--use_bf16", action="store_true", default=True,
                       help="使用 BF16 混合精度")
    parser.add_argument("--use_flash_attn", action="store_true", default=True,
                       help="使用 Flash Attention 2")
    parser.add_argument("--use_compile", action="store_true", default=True,
                       help="使用 torch.compile 加速")
    parser.add_argument("--compile_mode", type=str, default="default",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile 模式 (default 兼容 Liger Kernel)")
    parser.add_argument("--use_liger", action="store_true", default=True,
                       help="使用 Liger Kernel 优化")
    parser.add_argument("--use_8bit_adam", action="store_true", default=True,
                       help="使用 8-bit AdamW (如可用)")
    
    # HuggingFace 配置
    parser.add_argument("--hf_token", type=str, default=None,
                       help="HuggingFace Token")
    parser.add_argument("--push_to_hub", action="store_true", default=True,
                       help="训练完成后上传到 Hub")
    parser.add_argument("--hub_model_id", type=str, default=None,
                       help="Hub 模型 ID")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--resume", action="store_true", help="从 checkpoint 恢复")
    
    return parser.parse_args()


# ============================================================
# 工具函数
# ============================================================

def set_seed(seed):
    """[FIX-12] 完整随机种子设置"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_local_rank():
    """获取本地 rank"""
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size():
    """获取 world size"""
    return int(os.environ.get("WORLD_SIZE", 1))


def is_main_process():
    """是否为主进程"""
    return get_local_rank() == 0


def print_rank0(msg):
    """只在主进程打印"""
    if is_main_process():
        print(msg)


# ============================================================
# 数据加载 [FIX-8: 并行下载]
# ============================================================

def load_datasets(args):
    """加载并合并数据集"""
    from datasets import load_dataset, concatenate_datasets
    
    print_rank0("📥 加载数据集...")
    
    # [FIX-8] 并行下载
    def load_wiki():
        return load_dataset(
            "pleisto/wikipedia-cn-20230720-filtered",
            split="train",
            cache_dir=args.cache_dir,
        )
    
    def load_zhihu():
        return load_dataset(
            "wangrui6/Zhihu-KOL",
            split="train",
            cache_dir=args.cache_dir,
        )
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        wiki_future = executor.submit(load_wiki)
        zhihu_future = executor.submit(load_zhihu)
        
        wiki = wiki_future.result()
        zhihu = zhihu_future.result()
    
    print_rank0(f"   ✅ 维基百科: {len(wiki)} 条")
    print_rank0(f"   ✅ 知乎: {len(zhihu)} 条")
    
    # 统一字段名
    def process_wiki(example):
        return {"text": example["completion"]}
    
    def process_zhihu(example):
        return {"text": f"{example['INSTRUCTION']}\n{example['RESPONSE']}"}
    
    # [FIX-5] 使用更多 CPU 核心
    num_proc = min(8, multiprocessing.cpu_count())
    
    wiki_processed = wiki.map(process_wiki, remove_columns=wiki.column_names, num_proc=num_proc)
    zhihu_processed = zhihu.map(process_zhihu, remove_columns=zhihu.column_names, num_proc=num_proc)
    
    # 合并
    dataset = concatenate_datasets([wiki_processed, zhihu_processed])
    dataset = dataset.shuffle(seed=args.seed)
    
    print_rank0(f"✅ 数据集合并完成: {len(dataset)} 条")
    return dataset


# ============================================================
# 分词器 [FIX-7: 加速训练]
# ============================================================

def train_or_load_tokenizer(args, dataset):
    """训练或加载分词器"""
    import sentencepiece as spm
    from transformers import LlamaTokenizerFast, AutoTokenizer
    
    tokenizer_dir = Path(args.work_dir) / "tokenizer"
    sp_model_path = Path(args.work_dir) / "chinese_sp.model"
    
    # 缓存检测
    if (tokenizer_dir / "tokenizer.json").exists():
        print_rank0("✅ 检测到已有分词器，从缓存加载")
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        print_rank0(f"   词表大小: {len(tokenizer)}")
        return tokenizer
    
    # 只在主进程训练
    if is_main_process():
        print_rank0("🔤 训练 SentencePiece 分词器...")
        
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
        
        # [FIX] 验证 corpus 文件大小
        corpus_size = corpus_file.stat().st_size
        if corpus_size < 1_000_000:  # 至少 1MB
            raise RuntimeError(f"❌ corpus.txt 太小 ({corpus_size} bytes)！数据集可能下载不完整。请检查网络并重新运行。")
        print_rank0(f"   📄 Corpus 文件: {corpus_size / 1e6:.1f} MB")
        
        # [FIX-7] 使用 input_sentence_size 加速
        spm.SentencePieceTrainer.train(
            input=str(corpus_file),
            model_prefix=str(sp_model_path).replace(".model", ""),
            vocab_size=args.vocab_size,
            model_type="unigram",
            character_coverage=0.9995,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            pad_piece="<pad>", unk_piece="<unk>",
            bos_piece="<s>", eos_piece="</s>",
            num_threads=multiprocessing.cpu_count() or 4,
            input_sentence_size=200000,  # [FIX-7] 限制训练样本
            shuffle_input_sentence=True,  # [FIX-7] 打乱
        )
        
        # 验证 SP 模型
        sp = spm.SentencePieceProcessor()
        sp.load(str(sp_model_path))
        sp_vocab_size = sp.get_piece_size()
        print_rank0(f"   📊 SentencePiece 词表: {sp_vocab_size} tokens")
        
        if sp_vocab_size < 1000:
            raise RuntimeError(f"❌ SP 模型词表异常: 只有 {sp_vocab_size} tokens！")
        
        # [FIX-FINAL] 使用 tokenizers 库直接从 SP 模型构建 tokenizer.json
        # 然后用 PreTrainedTokenizerFast 加载 - 这是最可靠的方法
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        
        from tokenizers import Tokenizer, decoders, pre_tokenizers
        from tokenizers.models import Unigram
        from tokenizers.trainers import UnigramTrainer
        import json
        
        # 读取 SP vocab 文件构建 tokenizers 格式
        vocab_path = Path(args.work_dir) / "chinese_sp.vocab"
        vocab_list = []
        with open(vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    token = parts[0]
                    score = float(parts[1])
                    vocab_list.append((token, score))
        
        print_rank0(f"   📊 从 vocab 文件读取: {len(vocab_list)} tokens")
        
        # 构建 tokenizers Unigram 模型
        # 注意：SentencePiece vocab 中 unk_id=1 (pad=0, unk=1, bos=2, eos=3)
        tokenizer_obj = Tokenizer(Unigram(vocab_list, unk_id=1))
        tokenizer_obj.decoder = decoders.Metaspace()
        tokenizer_obj.pre_tokenizer = pre_tokenizers.Metaspace()
        
        # 保存为 tokenizer.json
        tokenizer_obj.save(str(tokenizer_dir / "tokenizer.json"))
        
        # 创建配置文件
        tokenizer_config = {
            "bos_token": "<s>",
            "eos_token": "</s>", 
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "add_bos_token": False,
            "add_eos_token": True,
            "model_max_length": 1024,
            "tokenizer_class": "PreTrainedTokenizerFast"
        }
        with open(tokenizer_dir / "tokenizer_config.json", "w") as f:
            json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
        
        special_tokens_map = {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>", 
            "pad_token": "<pad>"
        }
        with open(tokenizer_dir / "special_tokens_map.json", "w") as f:
            json.dump(special_tokens_map, f, indent=2, ensure_ascii=False)
        
        # 使用 PreTrainedTokenizerFast 加载
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_dir / "tokenizer.json"),
            bos_token="<s>", eos_token="</s>",
            unk_token="<unk>", pad_token="<pad>",
        )
        tokenizer.save_pretrained(str(tokenizer_dir))
        
        actual_vocab_size = len(tokenizer)
        print_rank0(f"   📊 HuggingFace 词表: {actual_vocab_size} tokens")
        
        if actual_vocab_size < 1000:
            raise RuntimeError(f"❌ 分词器词表异常: {actual_vocab_size} tokens，请检查环境")
        
        # 上传到 Hub
        if args.push_to_hub and args.hf_token:
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=args.hf_token)
                user = api.whoami()["name"]
                tokenizer.push_to_hub(f"{user}/chinese-sp-32k", token=args.hf_token)
                print_rank0("🚀 分词器已上传至 HuggingFace Hub")
            except Exception as e:
                print_rank0(f"⚠️ 上传失败: {e}")
        
        print_rank0(f"✅ 分词器训练完成，词表大小: {actual_vocab_size}")
    
    # 等待主进程
    if dist.is_initialized():
        dist.barrier()
    
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    return tokenizer


# ============================================================
# 数据预处理 [FIX-9: 降低 batch_size]
# ============================================================

def prepare_dataset(args, dataset, tokenizer):
    """Tokenize 和 Packing"""
    from datasets import load_from_disk
    
    lm_dataset_path = Path(args.work_dir) / "lm_dataset"
    
    # 缓存检测
    if lm_dataset_path.exists():
        print_rank0("✅ 检测到已处理的数据集，从缓存加载")
        lm_dataset = load_from_disk(str(lm_dataset_path))
        print_rank0(f"   样本数: {len(lm_dataset)}")
        return lm_dataset
    
    # 只在主进程处理
    if is_main_process():
        print_rank0("🔄 Tokenize 数据...")
        
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=False,
            )
        
        num_proc = min(8, multiprocessing.cpu_count())
        
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=2000,  # [FIX-9] 降低 batch_size
            remove_columns=dataset.column_names,
            num_proc=num_proc,
            desc="Tokenizing",
        )
        
        print_rank0("📦 Packing 数据...")
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
            batch_size=2000,  # [FIX-9] 降低 batch_size
            num_proc=num_proc,
            desc="Packing",
        )
        
        lm_dataset.save_to_disk(str(lm_dataset_path))
        print_rank0(f"✅ 数据处理完成: {len(lm_dataset)} 样本")
    
    # 等待主进程
    if dist.is_initialized():
        dist.barrier()
    
    lm_dataset = load_from_disk(str(lm_dataset_path))
    return lm_dataset


# ============================================================
# 模型创建 [FIX-1,2,3: 修复 compile 和 DDP 冲突]
# ============================================================

def create_model(args, tokenizer, flash_attn_available):
    """创建模型（不包含 compile - 由 Trainer 处理）"""
    from transformers import GPT2Config, GPT2LMHeadModel
    
    print_rank0("🏗️ 创建模型...")
    
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
    
    # [FIX-13] Flash Attention 检测基于 GPU 架构
    if args.use_flash_attn and flash_attn_available:
        config._attn_implementation = "flash_attention_2"
        print_rank0("   ✅ Flash Attention 2 已启用")
    else:
        config._attn_implementation = "sdpa"
        print_rank0("   ℹ️ 使用 SDPA 注意力")
    
    model = GPT2LMHeadModel(config)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print_rank0(f"   ✅ 模型创建完成: {param_count:.1f}M 参数")
    
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
            print_rank0(f"   ✅ Liger LayerNorm: 替换了 {count} 个")
        except ImportError:
            print_rank0("   ⚠️ liger-kernel 未安装，跳过")
    
    # [FIX-1] 不在这里启用 gradient_checkpointing
    # 让 TrainingArguments 统一管理，避免与 torch.compile 冲突
    
    # [FIX-2,3] 不在这里 torch.compile 或 DDP
    # 让 Trainer 自动处理
    
    return model


# ============================================================
# 检测 bitsandbytes 可用性 [FIX-4]
# ============================================================

def get_optimizer_name(use_8bit_adam):
    """[FIX-4] 安全检测 bitsandbytes 并选择优化器"""
    if not use_8bit_adam:
        return "adamw_torch_fused"
    
    try:
        import bitsandbytes as bnb
        # 实际测试是否支持 GPU
        test_param = torch.zeros(1, device='cuda', requires_grad=True)
        bnb.optim.Adam8bit([test_param])
        print_rank0("   ✅ 8-bit AdamW 可用")
        return "adamw_bnb_8bit"
    except Exception as e:
        print_rank0(f"   ⚠️ 8-bit AdamW 不可用: {e}")
        print_rank0("   → 回退到 fused AdamW")
        return "adamw_torch_fused"


# ============================================================
# 训练 [FIX-1,2,3,5,6: 全面修复]
# ============================================================

def train(args):
    """主训练函数"""
    local_rank = get_local_rank()
    world_size = get_world_size()
    
    # [FIX-12] 设置完整随机种子
    set_seed(args.seed)
    
    # [FIX-15] 更新缓存路径
    os.environ["HF_HOME"] = args.cache_dir
    os.environ["TRANSFORMERS_CACHE"] = args.cache_dir
    
    # 环境检查 ([FIX-13] 返回 flash_attn 可用性)
    flash_attn_available = check_environment(local_rank)
    
    # A100 优化 ([FIX-10] 传入 local_rank)
    if args.use_tf32:
        enable_a100_optimizations(local_rank)
    
    print_rank0("=" * 60)
    print_rank0("🚀 GPT-2 Chinese Pretraining (A100 v2 - FIXED)")
    print_rank0("=" * 60)
    print_rank0(f"   GPU: {torch.cuda.get_device_name(local_rank)}")
    print_rank0(f"   World Size: {world_size}")
    print_rank0(f"   工作目录: {args.work_dir}")
    
    # 创建目录
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
    dataset = load_datasets(args)
    
    # 分词器
    tokenizer = train_or_load_tokenizer(args, dataset)
    
    # 数据预处理
    lm_dataset = prepare_dataset(args, dataset, tokenizer)
    
    # 创建模型 ([FIX-1,2,3] 不在这里 compile 或 DDP)
    model = create_model(args, tokenizer, flash_attn_available or False)
    
    # [FIX-4] 检测优化器
    optim_name = get_optimizer_name(args.use_8bit_adam)
    
    # 训练参数
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    
    effective_batch = args.batch_size * args.gradient_accumulation * world_size
    print_rank0(f"\n📊 训练配置:")
    print_rank0(f"   Batch/GPU: {args.batch_size}")
    print_rank0(f"   有效 Batch: {effective_batch}")
    print_rank0(f"   Epochs: {args.num_epochs}")
    print_rank0(f"   优化器: {optim_name}")
    
    # [FIX-5,6] 计算最佳 num_workers
    num_workers = min(8, multiprocessing.cpu_count())
    
    training_args = TrainingArguments(
        output_dir=str(Path(args.work_dir) / "checkpoints"),
        
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        
        optim=optim_name,  # [FIX-4] 动态选择
        weight_decay=0.1,
        max_grad_norm=1.0,
        
        # BF16/FP16
        fp16=not args.use_bf16 and not torch.cuda.is_bf16_supported(),
        bf16=args.use_bf16 and torch.cuda.is_bf16_supported(),
        tf32=args.use_tf32,
        
        # [FIX-1] Gradient Checkpointing 由 Trainer 管理
        # 在使用 torch.compile 时禁用，避免冲突
        gradient_checkpointing=not args.use_compile,
        
        # [FIX-2,3] torch.compile 由 Trainer 管理
        torch_compile=args.use_compile,
        torch_compile_mode=args.compile_mode,
        
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        # prediction_loss_only=False,  # 移除以显示 eval_loss
        
        logging_steps=10,
        logging_first_step=True,
        report_to="none",
        
        push_to_hub=args.push_to_hub and is_main_process(),
        hub_model_id=args.hub_model_id,
        hub_strategy="checkpoint",
        
        # [FIX-5] 增加 num_workers
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=True,
        # [FIX-6] 启用 persistent_workers
        dataloader_persistent_workers=True if num_workers > 0 else False,
        
        seed=args.seed,
        
        # DDP 设置 - [FIX-3] 让 Trainer 自动处理
        ddp_find_unused_parameters=False,
    )
    
    # Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Callbacks
    from transformers import TrainerCallback
    
    class GenerationCallback(TrainerCallback):
        """评估时测试生成质量"""
        def __init__(self, tokenizer, prompts=None):
            self.tokenizer = tokenizer
            self.prompts = prompts or ["中国的历史", "人工智能是", "今天天气"]
        
        def on_evaluate(self, args, state, control, model, **kwargs):
            if not is_main_process():
                return
            
            print("\n" + "=" * 50)
            print(f"📝 Step {state.global_step} - 生成测试:")
            print("=" * 50)
            
            eval_model = model.module if hasattr(model, 'module') else model
            # 处理 torch.compile 包装
            if hasattr(eval_model, '_orig_mod'):
                eval_model = eval_model._orig_mod
            
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
    
    class DetailedLoggingCallback(TrainerCallback):
        """详细日志"""
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not is_main_process() or not logs:
                return
            
            step = state.global_step
            train_loss = logs.get("loss")
            eval_loss = logs.get("eval_loss")
            lr = logs.get("learning_rate", 0)
            
            # 显示训练 loss
            if train_loss is not None and (step <= 10 or step % 100 == 0):
                print(f"📊 Step {step}: train_loss={train_loss:.4f}, lr={lr:.2e}")
            
            # 显示验证 loss (eval 时触发)
            if eval_loss is not None:
                print(f"� Step {step}: eval_loss={eval_loss:.4f}")
    
    callbacks = [DetailedLoggingCallback()]
    if is_main_process():
        callbacks.append(GenerationCallback(tokenizer))
    
    # [FIX-14] 随机采样 eval_dataset
    eval_indices = random.sample(range(len(lm_dataset)), min(1000, len(lm_dataset)))
    eval_dataset = lm_dataset.select(eval_indices)
    
    # 创建 Trainer
    # [FIX-2,3] 直接传入 model，让 Trainer 处理 DDP 和 compile
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,  # transformers 5.0: tokenizer -> processing_class
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    # [FIX] 检查 resume 时是否有 checkpoint
    resume_path = None
    if args.resume:
        checkpoint_dir = Path(args.work_dir) / "checkpoints"
        checkpoints = list(checkpoint_dir.glob("checkpoint-*")) if checkpoint_dir.exists() else []
        if checkpoints:
            resume_path = True  # Trainer 会自动找最新的
            print_rank0(f"✅ 检测到 {len(checkpoints)} 个 checkpoint，继续训练")
        else:
            print_rank0("⚠️ 未找到 checkpoint，从头开始训练")
    
    # 开始训练
    print_rank0("\n" + "=" * 60)
    print_rank0("🎯 开始训练...")
    print_rank0("=" * 60)
    
    try:
        trainer.train(resume_from_checkpoint=resume_path)
        print_rank0("\n✅ 训练完成!")
        
        # 保存最终模型
        if is_main_process():
            final_path = Path(args.work_dir) / "final_model"
            trainer.save_model(str(final_path))
            tokenizer.save_pretrained(str(final_path))
            
            # 上传到 Hub
            if args.push_to_hub and args.hf_token:
                print_rank0("📤 上传到 HuggingFace Hub...")
                from huggingface_hub import HfApi
                api = HfApi(token=args.hf_token)
                user = api.whoami()["name"]
                repo_id = args.hub_model_id or f"{user}/gpt2-chinese-mini"
                
                # [FIX] 先创建仓库（如果不存在）
                api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
                
                api.upload_folder(
                    folder_path=str(final_path),
                    repo_id=repo_id,
                    commit_message="Training complete (A100 v2)",
                )
                print_rank0(f"🎉 模型已上传至: https://huggingface.co/{repo_id}")
                
    except KeyboardInterrupt:
        print_rank0("\n⚠️ 训练被中断，保存 checkpoint...")
        trainer.save_model()


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    args = parse_args()
    train(args)
