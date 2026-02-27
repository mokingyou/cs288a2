"""
BPE (Byte Pair Encoding) training implementation.

This module implements the BPE algorithm for learning a tokenizer vocabulary
from a text corpus, compatible with GPT-2 style tokenization.
"""
from __future__ import annotations
import regex as re
from pathlib import Path
from typing import Iterator
import heapq
from collections import Counter, defaultdict

# GPT-2 pre-tokenization pattern
GPT2_PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    re.UNICODE
)


def get_pairs(word: tuple[bytes, ...]) -> set[tuple[bytes, bytes]]:
    """Get all adjacent pairs in a word (tuple of byte tokens)."""
    pairs = set()
    for i in range(len(word) - 1):
        pairs.add((word[i], word[i + 1]))
    return pairs


def merge_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """Merge all occurrences of a pair in a word."""
    first, second = pair
    new_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
            new_word.append(first + second)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


def pre_tokenize(text: str, special_tokens: list[str] | None = None) -> Iterator[str]:
    """
    Pre-tokenize text using GPT-2 pattern, preserving special tokens.
    
    Special tokens are yielded as-is (not split by the regex pattern).
    """
    special_tokens = special_tokens or []
    
    if not special_tokens:
        # No special tokens, just use the pattern
        for match in GPT2_PAT.finditer(text):
            yield match.group()
        return
    
    # Sort special tokens by length (longest first) for greedy matching
    sorted_specials = sorted(special_tokens, key=len, reverse=True)
    
    # Build a pattern that matches special tokens
    import re as std_re
    special_pattern = "|".join(std_re.escape(s) for s in sorted_specials)
    split_pattern = f"({special_pattern})"
    
    # Split text by special tokens
    parts = std_re.split(split_pattern, text)
    
    for part in parts:
        if part in special_tokens:
            # Special token - yield as-is, but it won't be BPE-encoded
            # (we skip special tokens in the word frequency counting)
            continue
        elif part:
            # Regular text - apply GPT-2 pre-tokenization
            for match in GPT2_PAT.finditer(part):
                yield match.group()

def build_initial_vocab(special_tokens: list[str]
                        ) -> dict[int, str]:
    """ 1. VOCABULARY INITIALIZATION
       The initial vocabulary is built in this exact order:
       - First: Add special tokens (in the order provided)
       - Then: Add all 256 single-byte values (0x00 to 0xFF)
       
       Example with special_tokens=["<|endoftext|>"]:
         vocab = {
             0: b"<|endoftext|>",   # Special token first
             1: b"\\x00",           # Byte 0
             2: b"\\x01",           # Byte 1
             ...
             256: b"\\xff",         # Byte 255
         }
       
       So the initial vocab size = len(special_tokens) + 256 """
    
    vocab = dict()
    
    for i, special_token in enumerate(special_tokens):
        vocab[i] = special_token.encode("utf-8")

    for i, byte_element in enumerate(list(range(256))):
        vocab[i+len(special_tokens)] = bytes([i])

    return vocab


def update_pair_counts(word_freqs: Counter[tuple[bytes, ...], int],
                       ) -> Counter[tuple[bytes, bytes]]:
    """ 3. PAIR FREQUENCY COUNTING  
       Count how often each adjacent pair appears across ALL words, weighted by
       word frequency.
       
       Example: If word (b'h', b'e', b'l', b'l', b'o') appears 10 times:
         - pair (b'h', b'e') gets +10
         - pair (b'e', b'l') gets +10
         - pair (b'l', b'l') gets +10
         - pair (b'l', b'o') gets +10 """
    pair_counts = Counter()

    for word, freq in word_freqs.items():
        pairs = get_pairs(word)
        for pair in pairs:
            pair_counts[pair] += freq
    
    return pair_counts

import heapq
from collections import Counter, defaultdict

class RevPair:
    __slots__ = ("p",)
    def __init__(self, p: tuple[bytes, bytes]):
        self.p = p
    def __lt__(self, other: "RevPair") -> bool:
        # Reverse the *entire tuple* ordering.
        # This makes the heap prefer lexicographically *largest* pair on ties.
        return self.p > other.p

def heap_push(heap, pair_counts: Counter, p: tuple[bytes, bytes]):
    c = pair_counts.get(p, 0)
    if c > 0:
        # heap item ordered by (-count, RevPair(pair)) == max(count, pair)
        heapq.heappush(heap, (-c, RevPair(p), p))

def heap_pop_best(heap, pair_counts: Counter) -> tuple[bytes, bytes] | None:
    while heap:
        negc, _rp, p = heap[0]
        c = -negc
        # lazy deletion: stale count or dead pair => discard
        if pair_counts.get(p, 0) != c or c <= 0:
            heapq.heappop(heap)
            continue
        return p
    return None

def iter_adjacent_pairs(word: tuple[bytes, ...]):
    for i in range(len(word) - 1):
        yield (word[i], word[i + 1])

def contains_pair(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> bool:
    a, b = pair
    for i in range(len(word) - 1):
        if word[i] == a and word[i + 1] == b:
            return True
    return False

def iter_adjacent_pairs(word: tuple[bytes, ...]) -> Iterator[tuple[bytes, bytes]]:
    for i in range(len(word) - 1):
        yield (word[i], word[i + 1])

def merge_word_once(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    # same as your merge_word; keeps “merge all occurrences” semantics
    first, second = pair
    out = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
            out.append(first + second)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)



def train_bpe(
    input_path: Path,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    special_tokens = special_tokens or []

    text = input_path.read_text(encoding="utf-8")
    vocab = build_initial_vocab(special_tokens)  # your function; returns id->bytes
    merges: list[tuple[bytes, bytes]] = []

    # Forbidden substrings (prefixes of specials), like you had — but do it as bytes list
    forbidden: list[bytes] = []
    for s in special_tokens:
        sb = s.encode("utf-8")
        forbidden.extend(sb[:i] for i in range(2, len(sb) + 1))

    # 1) Build word_freqs as Counter[word_tuple] -> freq (streaming)
    word_freqs: Counter[tuple[bytes, ...]] = Counter()
    for tok in pre_tokenize(text, special_tokens=special_tokens):
        b = tok.encode("utf-8")
        if forbidden and any(f in b for f in forbidden):
            continue
        word = tuple(b[i:i+1] for i in range(len(b)))  # tuple of single bytes
        if word:
            word_freqs[word] += 1

    # 2) Freeze unique words into arrays (CS336 pattern)
    #    We'll keep an indexed list of current word representations and their frequencies.
    words = list(word_freqs.keys())
    freqs = [word_freqs[w] for w in words]
    
    # 3) pair_counts + pair_to_words (inverted index)
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)

    def add_word_pairs(wid: int, w: tuple[bytes, ...], f: int):
        for p in iter_adjacent_pairs(w):
            pair_counts[p] += f
            pair_to_words[p].add(wid)

    def remove_word_pairs(wid: int, w: tuple[bytes, ...], f: int):
        for p in iter_adjacent_pairs(w):
            pair_counts[p] -= f
            pair_to_words[p].discard(wid)

    for wid, w in enumerate(words):
        add_word_pairs(wid, w, freqs[wid])

    # 1) Build heap once initially over pair_counts
    heap = []
    for p, c in pair_counts.items():
        if c > 0:
            heap_push(heap, pair_counts, p)


    num_merges = vocab_size - len(vocab)
    for _ in range(num_merges):
        best_pair = heap_pop_best(heap, pair_counts)
        if best_pair is None:
            break

        # If you need the EXACT autograder semantics, this equals:
        # best_pair == max(pair_counts, key=lambda p: (pair_counts[p], p))

        # record merge + vocab add
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

        affected = list(pair_to_words.get(best_pair, ()))
        if not affected:
            pair_counts[best_pair] = 0
            continue

        touched = set()

        for wid in affected:
            old_w = words[wid]
            f = freqs[wid]

            # membership can be stale because we use lazy structures
            if not contains_pair(old_w, best_pair):
                pair_to_words[best_pair].discard(wid)
                continue

            # remove old contributions (and track touched pairs)
            for p in iter_adjacent_pairs(old_w):
                touched.add(p)
            remove_word_pairs(wid, old_w, f)

            # merge
            new_w = merge_word(old_w, best_pair)
            words[wid] = new_w

            # add new contributions (and track touched pairs)
            for p in iter_adjacent_pairs(new_w):
                touched.add(p)
            add_word_pairs(wid, new_w, f)

        pair_to_words[best_pair].clear()

        # 3) Push touched pairs back into heap (lazy deletion handles staleness)
        for p in touched:
            if pair_counts.get(p, 0) > 0:
                heap_push(heap, pair_counts, p)

    return vocab, merges


