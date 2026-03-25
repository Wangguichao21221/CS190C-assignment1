from cs336_basics.utils import *
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.bpe import *
import argparse
import time
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_path_out", type=str, default="data/vocab.json")
    parser.add_argument("--merges_path_out", type=str, default="data/merges.txt")
    parser.add_argument("--special_tokens", type=str, nargs="*", default=["<|endoftext|>"])
    parser.add_argument("--corpus_path", type=str, default="data/corpus.txt")
    parser.add_argument("--num_process", type=int, default=4)
    parser.add_argument("--vocab_size", type=int, default=5000)
    args = parser.parse_args()
    return args
def BPE():
  args = parser()
  vocab_path_out = args.vocab_path_out
  merges_path_out = args.merges_path_out
  special_tokens = args.special_tokens
  corpus_path = args.corpus_path

  print('preprocessing the corpus and building the BPE vocabulary...')
  start_time = time.time()
  vocab,merges = train_bpe(
    input_path=corpus_path,
    vocab_size=args.vocab_size,
    special_tokens=special_tokens,
    num_process=args.num_process,
    vocab_output_path=vocab_path_out,
    merges_output_path=merges_path_out,
    save=True
    )
  end_time = time.time()
  print(f"preprocessing completed in {end_time - start_time:.2f} seconds, vocab size: {len(vocab)}, merges size: {len(merges)}")
  print()
if __name__ == "__main__":
  BPE()