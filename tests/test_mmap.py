import pathlib
FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent) / "fixtures"
import cs336_basics.utils
CORPUS_PATH =FIXTURES_PATH / "corpus.en"
VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"
def test_mmap():
  mmap = cs336_basics.utils.Mmap(CORPUS_PATH,VOCAB_PATH,MERGES_PATH,special_tokens=["<|endoftext|>"],chunk_size=10)
  mmap.save_as_memmap()
test_mmap()