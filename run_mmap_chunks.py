from cs336_basics.utils import *
from cs336_basics.tokenizer import Tokenizer
import argparse
import time
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", type=str, default="data/corpus.txt")
    parser.add_argument("--vocab_path", type=str, default="data/vocab.json")
    parser.add_argument("--merges_path", type=str, default="data/merges.txt")
    parser.add_argument("--special_tokens", type=str, nargs="*", default=["<|endoftext|>"])
    parser.add_argument("--chunk_size", type=int, default=10000)
    args = parser.parse_args()
    return args
def chunk():
  args = parser()
  corpus_path = args.corpus_path
  vocab_path = args.vocab_path
  merges_path = args.merges_path
  special_tokens = args.special_tokens
  chunk_size = args.chunk_size
  print('preprocessing the corpus and chunking...')
  start_time = time.time()
  mmap = Mmap(corpus_path,vocab_path, merges_path,special_tokens,chunk_size)
  mmap.save_as_memmap()
  end_time = time.time()
  print(f"preprocessing completed in {end_time - start_time:.2f} seconds.")
if __name__ == "__main__":
  chunk()