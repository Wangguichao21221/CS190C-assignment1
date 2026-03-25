import argparse
import json
from xml.parsers.expat import model

import torch
import time
from cs336_basics.transformer import TransformerLM
from cs336_basics.utils import *
from cs336_basics.tokenizer import Tokenizer

def train_loop(batch_sampler,batch_size,seq_len,dataset_len,device,transformer_lm,token_positions,loss_func,optimizer,grad_clipper,should_output,vocab_path,merges_path,special_tokens):
    tensor_input = batch_sampler.get_batch_mmap(batch_size, seq_len, dataset_len, device=device)
    token_ids, labels = tensor_input

    if should_output:
        tokenizer = Tokenizer.from_files(
            vocab_path,
            merges_path,
            special_tokens=special_tokens
        )
        sample_ids = token_ids[0].cpu().numpy()
        sample_text = tokenizer.decode(sample_ids)
        print(f"sample text:\n{sample_text}\n")

    output = transformer_lm(token_ids)

    if should_output:
        sample_output = output[0].detach().cpu().numpy()
        sample_pred_ids = np.argmax(sample_output, axis=-1).tolist()
        sample_pred_text = tokenizer.decode(sample_pred_ids)
        print(f"sample prediction:\n{sample_pred_text}\n")

    loss = loss_func(output, labels)
    optimizer.zero_grad()  
    loss.backward()        
    grad_clipper(transformer_lm.parameters())  
    optimizer.step()       

    return loss.item()  

def decode_func(transformer_lm,tokenizer,config,device):
    prompt_path = "data/prompt.txt"
    with open(prompt_path, encoding='utf-8') as f:
        encoded_list = tokenizer.encode(f.read())
    decode_list = tokenizer.decode(encoded_list)
    print("-------------------------------")
    print(f"Text to decode:\n{decode_list}\n")
    print("Generated text:")
    input("Press Enter to start decoding...")
    it = 0
    max_gen_len = 100
    temp = 0.8
    repeat_penalty = 1.2
    recent_tokens = []

    while it < max_gen_len:
        if len(encoded_list) < config["max_seq_length"]:
            input_ids = encoded_list
        else:
            input_ids = encoded_list[-config["max_seq_length"]:]

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            output_scores = transformer_lm(input_tensor)
            last_token_scores = output_scores[0, -1, :]

            for token_id in recent_tokens[-10:]:
                last_token_scores[token_id] /= repeat_penalty

            last_token_scores = last_token_scores / temp
            last_token_weights = softmax(last_token_scores, dim=-1)
            sampled_id = torch.multinomial(last_token_weights, num_samples=1).item()

        encoded_list.append(sampled_id)
        sampled_token = tokenizer.decode([sampled_id])
        print(sampled_token, end="")

        recent_tokens.append(sampled_id)
        if len(recent_tokens) > 15:
            recent_tokens.pop(0)

        it += 1
def train_manage(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="data/TinyStoriesV2-GPT4.json", help="Path to the config file")
    args = parser.parse_args()
    config_path = args.config_path
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    d_model = config["d_model"]
    num_heads = config["num_heads"]
    d_ff = config["d_ff"]
    vocab_size = config["vocab_size"]
    num_layers = config["num_layers"]
    max_seq_length = config["max_seq_length"]
    seq_length = config["seq_length"]
    batch_size = config["batch_size"]
    theta = config["theta"]
    device = config["device"]
    num_epochs = config["num_epochs"]
    lr_max= config["lr"]
    lr_min = config["lr_min"]
    warmup_ratio = config["warmup_ratio"]
    warmfix_ratio = config["warmfix_ratio"]
    chunk_size = config["chunk_size"]
    vocab_path = config["vocab_path"]
    merges_path = config["merges_path"]
    special_tokens = config["special_tokens"]
    log_interval = config["log_interval"]
    save_interval = config["save_interval"]
    weight_decay = config["weight_decay"]
    betas = tuple(config["betas"])
    eps = config["eps"]
    max_norm = config["max_norm"]
    dtype = torch.float32
    dataset_len = config["dataset_len"]
    token_positions = torch.arange(seq_length,device=device)
    corpuse_path = config["corpus_path"]
    save_path = config["save_path"]
    transformer_lm = TransformerLM(
        vocab_size=vocab_size,
        max_seq_length=max_seq_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads, 
        d_ff=d_ff,
        rope_theta=theta,
        device=device,
        dtype=dtype
    )
    total_params = sum(p.numel() for p in transformer_lm.parameters())

    # 计算 可训练参数量
    trainable_params = sum(p.numel() for p in transformer_lm.parameters() if p.requires_grad)

    print(f"Model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    loss_func = CrossEntropyLoss()
    optimizer = AdamW_Optimizer(transformer_lm.parameters(), lr=lr_max, weight_decay=weight_decay, betas=betas, eps=eps)
    gradient_clipper = grad_clipper(max_norm)
    mmap = Mmap(corpuse_path,vocab_path,merges_path,special_tokens,chunk_size)
    batch_sampler = Batch_Random_Sampler(mmap)
    check_point_manager = Checkpoint_Manager()
    lr_scheduler = lr_cosine_scheduler()
    tokens_per_tensor = batch_size*seq_length
    print(f"tokens_per_tensor:{tokens_per_tensor}, dataset_len:{dataset_len}")
    total_iters = dataset_len//tokens_per_tensor * int(num_epochs)
    warmup_iters = int(total_iters*warmup_ratio)
    warmfix_iters = int(total_iters*warmfix_ratio)
    print(f'total_iters:{total_iters}, warmup_iters:{warmup_iters}, warmfix_iters:{warmfix_iters}')

    it = 0
    try:
        it= check_point_manager.load(save_path,transformer_lm,optimizer)
        print(f'Checkpoint loaded, starting from iteration {it}')
    except:
        print('No checkpoint found, starting from scratch.')
    last_time = time.time()
    last_it = it
    while it < total_iters:
        lr = lr_scheduler.get_lr(it, lr_max, lr_min, warmup_iters, warmfix_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if it>=int(total_iters*0.025) and it+1 %1000 == 0:
            should_output = True
        else:
            should_output = False
        loss = train_loop(batch_sampler,batch_size,seq_length,dataset_len,device,transformer_lm,token_positions,loss_func,optimizer,gradient_clipper,should_output,vocab_path,merges_path,special_tokens)
        if (it + 1) % log_interval == 0:
            current_time = time.time()
            time_spent = current_time - last_time
            ites_spent = it - last_it
            time_remaining = time_spent / ites_spent * (total_iters - it)
            hours = int(time_remaining) // 3600
            minutes = (int(time_remaining) % 3600) // 60
            seconds = int(time_remaining) % 60
            print(f"iteration/total:{it+1}/{total_iters}, loss:{loss}, lr:{lr}. Time remaining:{hours}h{minutes}m{seconds}s")

        if (it + 1) % save_interval == 0:
            try:
                check_point_manager.save(transformer_lm, optimizer, it+1, save_path)
                print(f"model saved at iteration {it+1} to {save_path}")
            except:
                print("model saving failed")

        it += 1
    try:
        check_point_manager.save(transformer_lm, optimizer, it, save_path)
        print(f"The model is trained for {num_epochs} epochs, final model saved to {save_path}")
    except:
        print("model saving failed")

    config_dict = {
        "max_seq_length": max_seq_length,
        "d_model": d_model,
        "num_heads": num_heads,
        "d_ff": d_ff,
        "num_layers": num_layers,
        "theta": theta
    }

    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
    decode_func(transformer_lm, tokenizer, config_dict, device)

if __name__ == "__main__":
    train_manage()