import json
import regex as re

def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d
class Tokenizer:
  def __init__(self,vocab: dict[int:bytes] ,merges :list[tuple[bytes,bytes]],special_tokens:list[str]=None):
    self.vocab = vocab
    self.merges = merges
    if special_tokens:
      self.special_tokens = [token.encode(encoding='utf-8') for token in special_tokens]
    else:
      self.special_tokens = []
    special_tokens_to_append = []
    for bytes in self.special_tokens:
      if bytes not in self.vocab.values():
        special_tokens_to_append.append(bytes)
    length = len(vocab)
    for token in special_tokens_to_append:
       self.vocab[length] = token
       length+=1
    self.bytes2index = {token:k for k,token in self.vocab.items()}
    self.merge_rank = {pair: rank for rank, pair in enumerate(self.merges)}
    # i = 0
    # for k,v in self.vocab.items():
    #   print(f'k:{k},v:{v}')
    #   i+=1
    #   if i>10:
    #     break
    # i = 0
    # for k,v in self.bytes2index.items():
    #   print(f'k:{k},v:{v}')
    #   i+=1
    #   if i>10:
    #     break
  @classmethod
  def from_files(cls, vocab_filepath,merges_filepath,special_tokens=None):
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_filepath,encoding='utf-8') as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    gpt2_bpe_merges = []
    with open(merges_filepath,encoding='utf-8') as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
    # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
    # just return the original bytes, so we don't force students to use
    # any particular encoding scheme.
    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }
    # If any of the special tokens don't exist in the vocab, append them to the vocab.
    if special_tokens:
        for special_token in special_tokens:
            byte_encoded_special_token = special_token.encode("utf-8")
            if byte_encoded_special_token not in set(vocab.values()):
                vocab[len(vocab)] = byte_encoded_special_token

    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    return cls(vocab,merges,special_tokens)
  def pretokenize(self,text:str) -> list[bytes]:
    text = text.encode(encoding='utf-8')
    PAT = rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    if not self.special_tokens:
      pretokens = []
      for match in re.finditer(PAT, text):
        pretokens.append(match.group())
      return pretokens

    pretokens = []
    escaped_tokens = [re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)]
    special_pattern = rb'(?:' + b'|'.join(escaped_tokens) + rb')'

    # First carve out exact special-token spans, then pretokenize the rest.
    last_end = 0
    for special_match in re.finditer(special_pattern, text):
      non_special_chunk = text[last_end:special_match.start()]
      for match in re.finditer(PAT, non_special_chunk):
        pretokens.append(match.group())
      pretokens.append(special_match.group())
      last_end = special_match.end()

    tail_chunk = text[last_end:]
    for match in re.finditer(PAT, tail_chunk):
      pretokens.append(match.group())

    return pretokens
  def merge(self, token: bytes):
    # print(f"Merging:{token}")
    parts = [bytes([bt]) for bt in token]
    while len(parts) > 1:
      best_rank = float('inf')
      best_idx = -1
      for i in range(len(parts) - 1):
        rank = self.merge_rank.get((parts[i], parts[i+1]), float('inf'))
        if rank < best_rank:
          best_rank = rank
          best_idx = i
      if best_idx == -1:
        break
      parts[best_idx] = parts[best_idx] + parts[best_idx + 1]
      parts.pop(best_idx + 1)
    # print(f"final tokens:{parts}")
    return parts

  def encode(self, text: str)-> list[int]:
  
    pretokens = self.pretokenize(text)
    # print(f"pretokens:{pretokens}")
    final = []
    list_of_bytes = []
    list_of_indexes = []
    for pretoken in pretokens:
      if pretoken in self.special_tokens:
        list_of_bytes.append(pretoken)
      else:
        list_of_bytes.extend(self.merge(pretoken))
    # print(list_of_bytes)
    for bts in list_of_bytes:
      if bts in self.bytes2index:
        list_of_indexes.append(self.bytes2index[bts])
      else:
         print("Error! not found in vocab!")
        #  print(pretokens)
        #  print(bts)
        #  print(list_of_bytes)
    return list_of_indexes
  def encode_iterable(self, iterable):
    for chunk in iterable:
      for token_id in self.encode(chunk):
        yield token_id
  def decode(self, token_ids: list[int]) -> str:
      if not token_ids:
          return ""
        
      byte_pieces = []
      for id in token_ids:
          if id in self.vocab:
              byte_pieces.append(self.vocab[id])
          else:
              byte_pieces.append(b' ')
        
      full_bytes = b''.join(byte_pieces)
      
      text = full_bytes.decode('utf-8', errors='replace')
        
      return text