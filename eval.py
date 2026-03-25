import argparse
import torch
from cs336_basics.transformer import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.utils import softmax

def parse_args():
    parser = argparse.ArgumentParser(description="TransformerLM Inference Script")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="训练好的模型权重路径 (.pt)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="运行设备")
    
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--theta", type=int, default=100000)
    
    parser.add_argument("--vocab_path", type=str, default="data/vocab.json")
    parser.add_argument("--merges_path", type=str, default="data/merges.txt")
    parser.add_argument("--special_tokens", type=str, nargs="*", default=["<|endoftext|>"])
    
    parser.add_argument("--prompt_file", type=str, default="", help="从文件读取提示词")
    parser.add_argument("--max_gen_len", type=int, default=200, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.8, help="温度系数，越高越随机")
    parser.add_argument("--repeat_penalty", type=float, default=1.2, help="重复惩罚，越大越不容易重复")
    parser.add_argument("--repeat_window", type=int, default=15, help="重复惩罚窗口大小")
    
    args = parser.parse_args()
    return args

def load_model(args):
    """加载训练好的 TransformerLM 模型"""

    dtype = torch.float32
    
    # 初始化模型结构（必须与训练一致）

    model = TransformerLM(
        vocab_size=args.vocab_size,
        max_seq_length=args.max_seq_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.theta,
        device=args.device,
        dtype=dtype
    )
    
    # 加载权重
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    # 你的保存格式是 "model" "optimizer" "iteration"
    model.load_state_dict(checkpoint["model"])
    
    model.eval()
    print(f"✅ model weight loaded from: {args.checkpoint_path}")
    return model

def load_tokenizer(args):
    return Tokenizer.from_files(
        args.vocab_path,
        args.merges_path,
        special_tokens=args.special_tokens
    )

def generate_text(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_gen_len: int = 200,
    temperature: float = 0.8,
    repeat_penalty: float = 1.2,
    repeat_window: int = 15,
    max_seq_length: int = 512
):
    """流式生成文本"""
    encoded = tokenizer.encode(prompt)
    print("\n" + "="*50)
    print(f"Prompt: \n{prompt}")
    print("="*50)
    print("results: ")
    
    recent_tokens = []
    with torch.no_grad():
        for _ in range(max_gen_len):
            if len(encoded) > max_seq_length:
                encoded = encoded[-max_seq_length:]
            
            input_tensor = torch.tensor([encoded], dtype=torch.long, device=device)
            logits = model(input_tensor)
            last_logits = logits[0, -1, :]
            
            for tok in recent_tokens[-repeat_window:]:
                last_logits[tok] /= repeat_penalty
            
            last_logits = last_logits / temperature
            probs = softmax(last_logits, dim=-1)
            
            token = torch.multinomial(probs, num_samples=1).item()
            encoded.append(token)
            recent_tokens.append(token)
            
            token_str = tokenizer.decode([token])
            print(token_str, end="", flush=True)
    
    print("\n" + "="*50 + "\n")
    full_text = tokenizer.decode(encoded)
    return full_text

def main():
    args = parse_args()
    
    model = load_model(args)
    tokenizer = load_tokenizer(args)
    
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
    else:
        prompt = args.prompt
    
    if not prompt:
        print("⚠️  No prompt, defult prompt")
        prompt = "Once upon a time"
    
    generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=args.device,
        max_gen_len=args.max_gen_len,
        temperature=args.temperature,
        repeat_penalty=args.repeat_penalty,
        repeat_window=args.repeat_window,
        max_seq_length=args.max_seq_length
    )

if __name__ == "__main__":
    main()
