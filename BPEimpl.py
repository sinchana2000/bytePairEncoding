#byte-pair-encoding

Pair = tuple[int, int]


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_pair_counts(ids: list[int]) -> dict[Pair, int]:

    p: dict[Pair, int] = {}
    for pair in zip(ids, ids[1:]):
        p[pair] = p.get(pair,0)+1
    return p

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def merge(ids: list[int], pair: Pair, new_id: int) -> list[int]:

    ans = []
    i=0
    while i in range(len(ids)):
        if(i<len(ids)-1 and (ids[i], ids[i+1])==pair):
            ans.append(new_id)
            i+=2
        else:
            ans.append(ids[i])
            i+=1
    return ans


class BPETokenizer:
    def __init__(self):

        self.merges: dict[Pair, int] = {}

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    def train(self, text: str, vocab_size: int) -> None:

        i=0
        btext = list(text.encode("utf-8"))
        maxbyte = 256
        while i<vocab_size-256:
            pair_count_map = get_pair_counts(btext)
            maxpair, maxfreq = max(pair_count_map.items(), key=lambda x:x[1])
            btext = merge(btext, maxpair, maxbyte)
            self.merges[maxpair] = maxbyte
            maxbyte+=1
            i+=1

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    def decode(self, ids: list[int]) -> str:

        vocabs = {}
        i=0
        while i < 256 :
            vocabs[i] = bytes([i])
            i+=1
        for pair,new_id in self.merges.items():
            vocabs[new_id] = vocabs[pair[0]]+vocabs[pair[1]]
        ans = b""
        for id in ids:
            ans+=vocabs[id]
        return ans.decode("utf-8", errors="replace")

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    def encode(self, text: str) -> list[int]:

        ids = list(text.encode("utf-8"))
        while len(ids)>=2:
            pair_counts = get_pair_counts(ids)
            pair = min(pair_counts, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids
# ===========================================================================
# Checks. Don't edit below this line.
# ===========================================================================
def _check(name, fn):
    try:
        fn()
    except NotImplementedError:
        print(f"[ ] {name}: not written yet. Start here.")
        raise SystemExit(0)
    except AssertionError as e:
        print(f"[x] {name}: WRONG. {e}")
        raise SystemExit(1)
    print(f"[✓] {name}")


def _t1():
    got = get_pair_counts([1, 2, 3, 1, 2])
    want = {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    assert got == want, f"got {got}, want {want}"
    assert get_pair_counts([]) == {}, "empty list should give {}"
    assert get_pair_counts([7]) == {}, "one item has no pairs"


def _t2():
    got = merge([1, 2, 3, 1, 2], (1, 2), 99)
    assert got == [99, 3, 99], f"got {got}, want [99, 3, 99]"
    got = merge([1, 1, 1], (1, 1), 9)
    assert got == [9, 1], f"overlap case: got {got}, want [9, 1]"
    got = merge([5, 6], (1, 2), 99)
    assert got == [5, 6], f"pair absent: got {got}, want [5, 6]"
    assert merge([], (1, 2), 99) == [], "empty list should stay empty"


def _t3():
    tok = BPETokenizer()
    tok.train("aaabdaaabac", vocab_size=259)
    assert len(tok.merges) == 3, f"want 3 merges, got {len(tok.merges)}"
    first = list(tok.merges.items())[0]
    a = ord("a")
    assert first == ((a, a), 256), f"first merge should be ('a','a') -> 256, got {first}"
    assert sorted(tok.merges.values()) == [256, 257, 258], "new ids should be 256, 257, 258"


def _t4():
    tok = BPETokenizer()
    assert tok.decode([104, 105]) == "hi", "with no merges, ids are just bytes"
    tok.train("aaabdaaabac", vocab_size=259)
    assert tok.decode([256]) == "aa", f"token 256 should decode to 'aa', got {tok.decode([256])!r}"
    for tid in tok.merges.values():
        tok.decode([tid])

def _t5():
    tok = BPETokenizer()
    tok.train("aaabdaaabac", vocab_size=259)
    ids = tok.encode("aaabdaaabac")
    assert len(ids) < 11, f"encode didn't compress anything: {ids}"
    assert all(i < 259 for i in ids), "encode produced an id that was never learned"
    assert tok.encode("") == [], "empty string should give []"
    assert tok.encode("z") == [ord("z")], "single char should be its byte"


def _t6():
    corpus = open(__file__, encoding="utf-8").read() * 3
    tok = BPETokenizer()
    tok.train(corpus, vocab_size=300)
    for s in ["def merge(ids): return ids", "hello world", "ಕನ್ನಡ 🙂", ""]:
        ids = tok.encode(s)
        assert tok.decode(ids) == s, f"round trip failed for {s!r}"
    s = "def merge(ids): return ids"
    print(f"    {len(s.encode())} bytes -> {len(tok.encode(s))} tokens")


if __name__ == "__main__":
    _check("TODO 1 get_pair_counts", _t1)
    _check("TODO 2 merge", _t2)
    _check("TODO 3 train", _t3)
    _check("TODO 4 decode", _t4)
    _check("TODO 5 encode", _t5)
    _check("Round trip (incl. Kannada + emoji)", _t6)
    print("\nAll done. Now compare with bpe.py and see what you did differently.")