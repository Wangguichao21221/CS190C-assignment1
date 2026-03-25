uv run run_mmap_chunks.py \
  --corpus_path ./data/TinyStoriesV2-GPT4-train.txt \
  --vocab_path ./tests/fixtures/gpt2_vocab.json \
  --merges_path ./tests/fixtures/gpt2_merges.txt \
  --special_tokens "<|endoftext|>" \
  --chunk_size 1000000 \
