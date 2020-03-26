"""Byte pair encoding utilities"""

import os
import json
import regex as re
from functools import lru_cache

@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    """Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class Encoder:
    def __init__(self, encoder, bpe_merges, errors='replace'):
        self.encoder = encoder
        self.decoder = {v:k for k,v in self.encoder.items()}
        self.errors = errors # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        while len(self.cache) > 1000:
          self.cache.popitem()
        return word

    def encode(self, text):
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text

try:
  from tokenizers import Tokenizer, models, pre_tokenizers, decoders
  use_high_speed_tokenizer = True
  print('Using high-speed tokenizer')
except:
  use_high_speed_tokenizer = False

class HighSpeedTokenizer(object):
  def __init__(self, vocab_path, bpe_merges_path):
    tokenizer = Tokenizer(models.BPE.from_files(vocab_path, bpe_merges_path))
    # Use the byte level
    add_prefix_spaces = False # Whether to automatically prefix the sequences with a space if none found
    tokenizer.with_pre_tokenizer(pre_tokenizers.ByteLevel.new(add_prefix_spaces))
    tokenizer.with_decoder(decoders.ByteLevel.new())
    # Setup truncation if needed
    truncate = False
    max_length = 1024
    if truncate:
      stride = 0
      strategy = 'longest_first' # Can also be `only_first` or `only_second`
      tokenizer.with_truncation(max_length, stride, strategy)
    # Setup padding if needed
    padding = False
    # Whether to always pad to max_length. If this is false, we will pad to the
    # longest sequence in the batch.
    pad_to_max_length = False
    padding_side = "right" # Can also be "left"
    pad_token_id = 0
    pad_token_type_id = 0
    pad_token = "[PAD]"
    if padding:
      tokenizer.with_padding(
        max_length if pad_to_max_length else None,
        padding_side,
        pad_token_id,
        pad_token_type_id,
        pad_token
      )
    self.tokenizer = tokenizer

  def encode(self, text):
    tokens = []
    lines = text.splitlines()
    c = '\n'
    n = len(lines) - 1
    for i, line in enumerate(lines):
      if i >= n:
        c = ''
      encoding = self.tokenizer.encode(line + c)
      tokens.extend(encoding.ids)
    if text.endswith('\n'):
      tokens.extend(self.tokenizer.encode('\n').ids)
    return tokens

  def decode(self, tokens):
    text = self.tokenizer.decode(tokens, False)
    return text

def read_bucket(path, mode='rb'):
  if os.path.isfile(path):
    with open(path, mode) as f:
      return f.read()
  else:
    import tensorflow as tf
    with tf.io.gfile.GFile(path, mode=mode) as f:
      return f.read()

import tempfile
from contextlib import contextmanager

@contextmanager
def bucket_file(path):
  if os.path.isfile(path):
    with open(path, "rb") as f:
      data = f.read()
    yield path, data
  else:
    data = read_bucket(path)
    with tempfile.NamedTemporaryFile() as tmp:
      tmp.write(data)
      tmp.seek(0)
      yield tmp.name, data

def bucket_path(path, *parts):
  if len(parts) <= 0:
    return path
  if path.startswith('gs://'):
    sep = '/'
  else:
    sep = os.sep
  if not path.endswith(sep):
    path = path + sep
  path = path + parts[0]
  return bucket_path(path, *parts[1:])

def get_encoder(model_path=None):
  if model_path is None:
    #model_path = 'gs://gpt-2/models/117M/'
    model_path = os.path.dirname(__file__)
  with bucket_file(bucket_path(model_path, 'encoder.json')) as (vocab_path, vocab_data):
    with bucket_file(bucket_path(model_path, 'vocab.bpe')) as (bpe_merges_path, bpe_data):
      encoder = json.loads(vocab_data.decode('utf8'))
      if use_high_speed_tokenizer:
        tokenizer = HighSpeedTokenizer(vocab_path=vocab_path, bpe_merges_path=bpe_merges_path)
        tokenizer.encoder = encoder
        return tokenizer
      bpe_data = bpe_data.decode('utf8')
      bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
      return Encoder(
          encoder=encoder,
          bpe_merges=bpe_merges,
      )

