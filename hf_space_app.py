# HuggingFace Space: app.py
# ================================
# 部署步骤:
# 1. 在 huggingface.co/spaces 创建新 Space
# 2. SDK 选择 Gradio, Hardware 选择 CPU basic
# 3. 上传此文件和 requirements.txt
# ================================

import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ============================================
MODEL_ID = "Wilsonwin/gpt2-chinese-mini"
# ============================================

print(f"正在加载模型: {MODEL_ID}")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.eval()
    LOAD_SUCCESS = True
    print("✅ 模型加载成功!")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    tokenizer = None
    model = None
    LOAD_SUCCESS = False


def generate_text(
    prompt: str,
    max_length: int = 150,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
) -> str:
    """生成中文文本"""
    
    if not LOAD_SUCCESS:
        return "❌ 模型未加载成功，请检查模型 ID 是否正确"
    
    if not prompt or not prompt.strip():
        return "请输入提示文本"
    
    prompt = prompt.strip()
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=min(max_length, 300),
                temperature=max(temperature, 0.1),
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
                repetition_penalty=1.1,
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated
    
    except Exception as e:
        return f"生成出错: {str(e)}"


# === CSS 样式 ===
custom_css = """
.gradio-container {
    max-width: 900px !important;
}
.output-text {
    font-size: 16px;
    line-height: 1.8;
}
"""

# === Gradio 界面 ===
with gr.Blocks(
    title="中文 GPT-2 Mini",
    theme=gr.themes.Soft(),
    css=custom_css
) as demo:
    
    gr.Markdown("""
    # 🇨🇳 中文 GPT-2 Mini - 从零预训练演示
    
    这是一个在 **A100 GPU** 上从随机权重开始训练的中文语言模型。
    
    ### 📊 模型信息
    | 属性 | 值 |
    |---|---|
    | 架构 | GPT-2 (8 层, 768 维) |
    | 参数量 | 82M |
    | 训练数据 | 中文维基百科 + 知乎问答 |
    | 词表大小 | 32,000 (SentencePiece) |
    | 训练时长 | ~1.4 小时 (11,838 步) |
    
    > ⚠️ **注意**: 这是一个教学演示模型，生成质量有限，可能产生不准确或无意义的内容。
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(
                label="📝 输入提示词",
                placeholder="例如：人工智能的未来发展...",
                lines=3,
                max_lines=5,
            )
            
            with gr.Accordion("⚙️ 生成参数", open=False):
                max_length = gr.Slider(
                    minimum=50, maximum=300, value=150, step=10,
                    label="最大长度",
                    info="生成文本的最大 token 数"
                )
                temperature = gr.Slider(
                    minimum=0.1, maximum=1.5, value=0.8, step=0.1,
                    label="温度 (Temperature)",
                    info="越高越随机，越低越确定"
                )
                top_k = gr.Slider(
                    minimum=10, maximum=100, value=50, step=5,
                    label="Top-K",
                    info="从概率最高的 K 个 token 中采样"
                )
                top_p = gr.Slider(
                    minimum=0.5, maximum=1.0, value=0.95, step=0.05,
                    label="Top-P (Nucleus)",
                    info="从累积概率达到 P 的 token 中采样"
                )
            
            generate_btn = gr.Button("🚀 生成文本", variant="primary", size="lg")
        
        with gr.Column(scale=3):
            output = gr.Textbox(
                label="📖 生成结果",
                lines=10,
                max_lines=15,
                show_copy_button=True,
                elem_classes=["output-text"],
            )
    
    # 绑定事件
    generate_btn.click(
        fn=generate_text,
        inputs=[prompt_input, max_length, temperature, top_k, top_p],
        outputs=output,
    )
    
    prompt_input.submit(
        fn=generate_text,
        inputs=[prompt_input, max_length, temperature, top_k, top_p],
        outputs=output,
    )
    
    # 示例
    gr.Examples(
        examples=[
            ["中国的历史可以追溯到"],
            ["人工智能是一种"],
            ["在科学研究中，"],
            ["教育的重要性在于"],
            ["未来的城市将会"],
        ],
        inputs=prompt_input,
        label="💡 示例提示词"
    )
    
    gr.Markdown("""
    ---
    ### 🔗 相关链接
    - [模型仓库](https://huggingface.co/Wilsonwin/gpt2-chinese-mini) - 下载模型权重
    - [训练数据](https://huggingface.co/datasets/Wilsonwin/chinese-wiki-zhihu-corpus) - 7.56M 条中文语料
    
    ### 📚 技术细节
    - **分词器**: SentencePiece Unigram (32K 词表，专为中文优化)
    - **训练框架**: Hugging Face Transformers + Flash Attention 2 + 8-bit AdamW
    - **训练硬件**: AutoDL A100 40GB
    """)


# 启动
if __name__ == "__main__":
    demo.launch()
