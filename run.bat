@echo off
uv run .\run.py ^
  --d_model 512 ^
  --num_heads 8 ^
  --d_ff 1344 ^
  --vocab_size 5000 ^
  --num_layers 8 ^
  --max_seq_length 256 ^
  --seq_length 256 ^
  --batch_size 48 ^
  --theta 100000 ^
  --device cpu ^
  --num_epochs 5.5 ^
  --lr 1e-4 ^
  --lr_min 1e-5 ^
  --warmup_ratio 0.05 ^
  --warmfix_ratio 0.9 ^
  --chunk_size 10000 ^
  --vocab_path ./data/vocab.json ^
  --merges_path ./data/merges.txt ^
  --special_tokens "<|endoftext|>" ^
  --log_interval 20 ^
  --save_interval 500 ^
  --weight_decay 0.01 ^
  --betas 0.9 0.95 ^
  --eps 1e-8 ^
  --max_norm 1.0 ^
  --corpus_path ./data/corpus.en ^
  --dataset_len 31278
pause