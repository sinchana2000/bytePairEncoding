# Byte-Pair Encoding Tokenizer

A small, dependency-free implementation of a byte-level Byte-Pair Encoding (BPE) tokenizer in pure Python. It learns merge rules from a text corpus, then encodes text into token ids and decodes ids back into text.

## How it works

BPE starts from raw UTF-8 bytes (ids 0 to 255) and repeatedly merges the most frequent adjacent pair of ids into a new id (256, 257, ...). Because the base vocabulary is all 256 bytes, any string can be encoded, including non-English scripts and emoji, with no unknown tokens.

- **Training** counts adjacent pairs, merges the most frequent one, records the merge, and repeats until the vocabulary reaches the requested size.
- **Encoding** converts text to bytes, then repeatedly applies the learned merge with the lowest id (the earliest learned) until none apply.
- **Decoding** rebuilds each token's byte string from the merge table, concatenates the bytes, and decodes them as UTF-8.

## Files

| File | Purpose |
|------|---------|
| `BPEimpl.py` | The tokenizer plus a built-in set of self-checks |

## API

```python
Pair = tuple[int, int]

get_pair_counts(ids: list[int]) -> dict[Pair, int]
merge(ids: list[int], pair: Pair, new_id: int) -> list[int]

class BPETokenizer:
    merges: dict[Pair, int]                      # learned merges, in training order
    train(text: str, vocab_size: int) -> None    # learns vocab_size - 256 merges
    encode(text: str) -> list[int]
    decode(ids: list[int]) -> str
```

- `get_pair_counts` returns how often each adjacent pair occurs. Empty and single-item lists give `{}`.
- `merge` replaces non-overlapping occurrences of `pair` with `new_id`, scanning left to right. For example, `merge([1, 1, 1], (1, 1), 9)` gives `[9, 1]`.
- `decode` uses `errors="replace"`, so invalid byte sequences become the replacement character instead of raising.

## Usage

```python
from bpe_tokenizer import BPETokenizer

tok = BPETokenizer()
tok.train("aaabdaaabac", vocab_size=259)   # 3 merges: ids 256, 257, 258

ids = tok.encode("aaabdaaabac")
print(ids)               # shorter than the 11 input bytes
print(tok.decode(ids))   # "aaabdaaabac"
```

Round trips are lossless for any input, including multi-byte text:

```python
s = "ಕನ್ನಡ 🙂"
assert tok.decode(tok.encode(s)) == s
```

## Running the checks

The bottom of the file contains a small test harness. Run it directly:

```bash
python BPEimpl.py
```
## Requirements

Python 3.9 or newer. There are no third-party dependencies.
