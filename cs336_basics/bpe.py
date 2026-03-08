import regex as re
import os
import multiprocessing
# PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# dic = {}
# for match in re.finditer(PAT,"some text that i'll pre-tokenize"):
#   token = match.group()
#   if token in dic:
#     dic[token]+=1
#   else:
#     dic[token] = 1
# print(dic)、


def find_chunk_boundaries(
    file,
    desired_num_chunks,
    split_special_token,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def tuple_to_bytes(tuple):
  bytes = b''
  for byte in tuple:
    bytes+=byte
  return bytes
# def merge_bytes_to_tuple(bytes1,bytes2):
   
def pretoken_dict(pretokens: list):
  pretoken_freq = {}
  for pretoken in pretokens:
     if pretoken in pretoken_freq:
        pretoken_freq[pretoken] += 1
     else:
        pretoken_freq[pretoken] = 1
  return pretoken_freq
def chunk_text(text: str, boundaries: list[int],special_tokens) -> list[str]:
    """根据边界切分文本为chunk"""
    chunks = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        chunk = text[start:end]
        striped_chunk = split_and_strip_special_tokens(chunk,special_tokens)
        chunks.append(striped_chunk)
    return chunks
def split_and_strip_special_tokens(text: bytes, special_tokens: list[str]) -> bytes:
    """
    按特殊标记拆分文本，并剥离特殊标记
    Args:
        text: 字节类型的文本（chunk/corpus）
        special_tokens: 特殊标记列表（如 ["<|endoftext|>"]）
    Returns:
        list[bytes]: 拆分后的纯文本片段列表（无特殊标记）
    """

    escaped_special_tokens = [re.escape(token) for token in special_tokens]

    delimiter_pattern = rb"|".join([token.encode("utf-8") for token in escaped_special_tokens])

    split_parts = re.split(delimiter_pattern, text)
    striped_chunk =bytes()
    for part in split_parts:
       striped_chunk +=part
    return striped_chunk
def pretokenize_chunk(args):
    chunk, chunk_id = args
    try:
        tokens = pretokenize(chunk)
        # print(f"process complete：Chunk {chunk_id}, generate {len(tokens)} 个token")
        return tokens
    except Exception as e:
        print(f"Chunk {chunk_id} error：{e}")
        return []
def pretokenize_parallel(chunks,num_processes):
  final_tokens = []
  with multiprocessing.Pool(processes=num_processes) as pool:
      task_args = [(chunk, i) for i, chunk in enumerate(chunks)]
      chunk_results = pool.map(pretokenize_chunk, task_args)
  for result in chunk_results:
      final_tokens.extend(result)
  # print(final_tokens)
  return final_tokens
def pretokenize(corpus):
  PAT = rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
  pretokens = []
  for match in re.finditer(PAT,corpus):
    token = match.group()
    pretokens.append(token)
  return pretokens
def train_bpe(
  input_path: str | os.PathLike,
  vocab_size: int,
  special_tokens: list[str],
  num_process = 8,
  **kwargs,
)-> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
  with open(input_path,mode='rb') as f:
    corpus = f.read()
    boundaries = find_chunk_boundaries(f,num_process,b"<|endoftext|>")
    chunks = chunk_text(corpus,boundaries,special_tokens)

  pretokens = pretokenize_parallel(chunks,num_process)
  vocab,merges = bpe(pretokens,vocab_size)
  return  vocab,merges
def bpe(pretokens,vocab_size):
  merges = []
  #initial vocab
  vocab = {i:b"" for i in range(vocab_size)}
  vocab[0] = b"<|endoftext|>"
  for i in range(256):
    vocab[i + 1] = bytes([i])


  freq_dict = pretoken_dict(pretokens)
  # print(freq_dict)
  pair_dict = {token:[] for token in freq_dict}
  # find all the pairs in each pretoken
  for token in freq_dict:
    for i in range(len(token)-1):
      pair = (bytes([token[i]]),bytes([token[i+1]]))
      pair_dict[token].append(pair)
  vocab_index = 257
  # iterate the pretokens to find the most freq pair

  pairs_counts = {}
  max_num = 0
  max_pair = None
  for token in pair_dict:
    pairs = pair_dict[token]
    for pair in pairs:
      if pair in pairs_counts:
        pairs_counts[pair] += freq_dict[token]
      else:
        pairs_counts[pair] = freq_dict[token]
  # print(sorted(pairs_counts.items(),key=lambda x:x[1],reverse=True)[:10])
  max_pair,max_num = max(pairs_counts.items(), key=lambda x: (x[1], x[0]))
  merges.append(max_pair)
  vocab[vocab_index] = tuple_to_bytes(max_pair)
  vocab_index+=1
  # print(pair_dict)
  pairs_counts.pop(max_pair)
  # print(max_num,max_pair)
  # print(pairs_counts)
  # merge loop
  last_pair = max_pair
  while vocab_index<vocab_size:
    # print(f"Last pair:{last_pair}")
    for pretoken in pair_dict:
       pairs = pair_dict[pretoken]
       while last_pair in pairs:
          # print(f"processing:{pretoken}")
          index = pairs.index(last_pair)
          # front pair
          if index !=0:
            front = pairs[index-1]
            if front in pairs_counts:
              pairs_counts[front]-=freq_dict[pretoken]
            new_pair = (front[0],tuple_to_bytes(last_pair))
            pairs[index-1] = (front[0],front[1]+last_pair[1])
            if new_pair in pairs_counts:
               pairs_counts[new_pair]+=freq_dict[pretoken]
            else:
               pairs_counts[new_pair]=freq_dict[pretoken]
          # behind pair
          if index != len(pairs)-1:
             behind = pairs[index+1]
            #  print("pairs counts ",pairs_counts)
             if behind in pairs_counts:
              pairs_counts[behind]-=freq_dict[pretoken]
             new_pair = (tuple_to_bytes(last_pair),behind[1])
             pairs[index+1] = (last_pair[0]+behind[0],behind[1])
             if new_pair in pairs_counts:
               pairs_counts[new_pair]+=freq_dict[pretoken]
             else:
               pairs_counts[new_pair]=freq_dict[pretoken]
          pairs.pop(index)
    max_pair,max_num = max(pairs_counts.items(), key=lambda x: (x[1], x[0]))
    last_pair = max_pair
    vocab[vocab_index] = tuple_to_bytes(max_pair)
    merges.append(max_pair)
    pairs_counts.pop(max_pair)
    vocab_index+=1
  # print("==================================")
  # print(vocab)
  # print(merges)
  return  vocab,merges

if __name__ == "__main__":
  train_bpe("./tests/fixtures/corpus.en",500,["<|endoftext|>"])