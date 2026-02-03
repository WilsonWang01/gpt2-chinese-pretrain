#!/usr/bin/env python3
"""
上传训练好的模型到 HuggingFace Hub

使用方法:
    python upload_to_hf.py --hf_token YOUR_TOKEN

可选参数:
    --model_path: 模型路径 (默认: /root/autodl-tmp/gpt2-chinese/final_model)
    --repo_id: HuggingFace 仓库 ID (默认: YOUR_USERNAME/gpt2-chinese-mini)
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="上传模型到 HuggingFace Hub")
    parser.add_argument("--hf_token", type=str, required=True, help="HuggingFace Token")
    parser.add_argument("--model_path", type=str, 
                        default="/root/autodl-tmp/gpt2-chinese/final_model",
                        help="模型路径")
    parser.add_argument("--repo_id", type=str, default=None,
                        help="HuggingFace 仓库 ID (默认: YOUR_USERNAME/gpt2-chinese-mini)")
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    
    # 检查模型文件
    if not model_path.exists():
        # 尝试 checkpoints 目录
        checkpoint_dir = Path("/root/autodl-tmp/gpt2-chinese/checkpoints")
        if checkpoint_dir.exists():
            checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"), 
                               key=lambda x: int(x.name.split("-")[1]))
            if checkpoints:
                model_path = checkpoints[-1]
                print(f"⚠️ final_model 不存在，使用最新 checkpoint: {model_path}")
            else:
                print("❌ 未找到任何模型文件!")
                return
        else:
            print("❌ 未找到模型目录!")
            return
    
    print(f"📂 模型路径: {model_path}")
    print(f"📄 文件列表:")
    for f in model_path.iterdir():
        size = f.stat().st_size / 1e6
        print(f"   - {f.name}: {size:.1f} MB")
    
    # 上传
    from huggingface_hub import HfApi, login
    
    login(token=args.hf_token)
    api = HfApi(token=args.hf_token)
    
    # 获取用户名
    user = api.whoami()["name"]
    repo_id = args.repo_id or f"{user}/gpt2-chinese-mini"
    
    print(f"\n🚀 上传到: https://huggingface.co/{repo_id}")
    
    # 创建仓库
    api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
    
    # 上传
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=repo_id,
        commit_message="Upload GPT-2 Chinese model (82M params, trained on Wiki+Zhihu)",
    )
    
    print(f"\n✅ 上传完成!")
    print(f"🔗 模型地址: https://huggingface.co/{repo_id}")
    print(f"🎮 Demo: https://huggingface.co/spaces/{user}/gpt2-chinese-demo")


if __name__ == "__main__":
    main()
