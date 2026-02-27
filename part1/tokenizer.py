"""
BPE Tokenizer implementation compatible with GPT-2 / tiktoken.
"""

from __future__ import annotations

import regex as re
from typing import Iterator
import codecs

class Tokenizer:
    """
    A BPE (Byte Pair Encoding) tokenizer compatible with GPT-2.
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        Initialize the tokenizer.

        Args:
            vocab: Mapping from token ID to bytes
            merges: List of BPE merge pairs (bytes, bytes)
            special_tokens: List of special token strings
        """
        self.vocab = vocab  # id -> bytes
        self.inverse_vocab = {v: k for k, v in vocab.items()}  # bytes -> id (also used as rank)
        self.merges = merges
        # Note: We use inverse_vocab for BPE ranking, not the merges list.
        # In GPT-2/tiktoken, the token ID serves as the rank - lower ID = higher priority.
        # This is different from naive BPE which uses merge order.
        
        # Handle special tokens
        self.special_tokens = special_tokens or []
        # Sort special tokens by length (descending) for longest-match-first
        self.special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)
        if self.special_tokens_sorted:
            escaped = [re.escape(s) for s in self.special_tokens_sorted]
            self._special_split_re = re.compile(f"({'|'.join(escaped)})")
            self._special_set = set(self.special_tokens_sorted)
        else:
            self._special_split_re = None
            self._special_set = set()

        # Build special token to ID mapping
        self.special_token_ids = {}
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes in self.inverse_vocab:
                self.special_token_ids[token] = self.inverse_vocab[token_bytes]
        
        # GPT-2 regex pattern for pre-tokenization
        # This splits text into chunks that are tokenized independently
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            re.UNICODE
        )
        

    def _get_pairs(self, tokens: list[bytes]) -> set[tuple[bytes, bytes]]:
        """Get all adjacent pairs of tokens."""
        pairs = set()
        for i in range(len(tokens) - 1):
            pairs.add((tokens[i], tokens[i + 1]))
        return pairs

    def merge_word(self, word: list[bytes, ...], pair: tuple[bytes, bytes]) -> list[bytes, ...]:
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
        return new_word

    def _bpe(self, token_bytes: bytes) -> list[bytes]:
        """
        Apply BPE to a single token (sequence of bytes).
        Returns a list of merged byte sequences.
        
        Uses vocab ranks (token IDs) to determine merge priority.
        Lower token ID = higher priority (more common/earlier merge).
        
        Algorithm:
            1. Start with individual bytes as tokens
            2. While there are pairs that can be merged:
               a. Find the pair whose merged result has the lowest vocab rank
               b. Merge all occurrences of that pair
            3. Return final token list
        """
        # Start with individual bytes
        tokens = [bytes([b]) for b in token_bytes]
        
        if len(tokens) <= 1:
            return tokens

        # TODO: Implement BPE algorithm
        # "token_bytes" = encoding of "hello" but in bytes: "31713"
        # merge byte pairs as long as there is a pair in hello that is in the inverse vocab, but pick the highest priority pair
        # so lowest rank

        while True:
            possible_pairs = self._get_pairs(tokens)
            if not possible_pairs:
                break
            
            min_rank = float('inf')
            best_pair = None

            for pair in possible_pairs:
                merged = pair[0] + pair[1]
                if merged in self.inverse_vocab:
                    rank = self.inverse_vocab[merged]
                    if rank < min_rank:
                        min_rank = rank
                        best_pair = pair

            if best_pair is None:
                break
            else:
                tokens = self.merge_word(tokens, best_pair)
        
        return tokens

    def _split_with_special_tokens(self, text: str) -> list[tuple[str, bool]]:
        if not self._special_split_re:
            return [(text, False)] if text else []
        parts = [p for p in self._special_split_re.split(text) if p != ""]
        return [(p, p in self._special_set) for p in parts]

    def _encode_chunk(self, text: str) -> list[int]:
        """
        Encode a text chunk (without special tokens) to token IDs.
        
        Algorithm:
            1. Use regex pattern (self.pat) to split text into pre-tokens
            2. For each pre-token:
               a. Convert to bytes
               b. Apply BPE to get list of byte sequences
               c. Convert each byte sequence to token ID using inverse_vocab
               d. Handle unknown tokens by falling back to individual bytes
        """
        if not text:
            return []
        
        ids = []
        
        # TODO: Implement encoding
        for pre_token in self.pat.findall(text):
            bt = pre_token.encode("utf-8")
            # each self._bpe returns [b'he', b'll', b'o'] or something
            for piece in self._bpe(bt):
                tid = self.inverse_vocab[piece]
                if tid is not None:
                    ids.append(tid)
                else:
                    ids.extend(self.inverse_vocab[bytes([b])] for b in piece)
        return ids


    def encode(self, text: str) -> list[int]:
        """
        Encode a string to a list of token IDs.
        
        Args:
            text: Input string to encode
            
        Returns:
            List of token IDs
        """
        if not text:
            return []
        
        ids = []
        
        # Split by special tokens first
        parts = self._split_with_special_tokens(text)
        
        for part, is_special in parts:
            if is_special:
                # Add special token ID
                ids.append(self.special_token_ids[part])
            else:
                # Encode regular text
                ids.extend(self._encode_chunk(part))
        
        return ids

    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of token IDs to a string.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded string
        
        Algorithm:
            1. For each token_id, look up corresponding bytes in self.vocab
            2. Concatenate all byte chunks
            3. Decode as UTF-8 with errors="replace"
        """
        if not ids:
            return ""
        
        byte_list = [self.vocab[id] if id in self.vocab else id for id in ids]

        decoder = codecs.getincrementaldecoder('utf-8')('replace')

        decoded_string = ""

        for chunk in byte_list:
            decoded_string += decoder.decode(chunk, final=False)

        decoded_string += decoder.decode(b'', final=True)

        return decoded_string

    def encode_iterable(self, iterable: Iterator[str]) -> Iterator[int]:
        """
        Memory-efficient encoding of an iterable of strings.
        Yields token IDs one at a time without loading entire input into memory.
        
        Args:
            iterable: An iterable of strings (e.g., file handle)
            
        Yields:
            Token IDs one at a time
        """
        # Buffer for handling text that spans multiple lines
        buffer = ""
        
        for chunk in iterable:
            buffer += chunk
            
            # Process complete portions, keeping potential partial special tokens
            # Find the last safe split point
            safe_end = self._find_safe_split_point(buffer)
            
            if safe_end > 0:
                to_process = buffer[:safe_end]
                buffer = buffer[safe_end:]
                
                for token_id in self.encode(to_process):
                    yield token_id
        
        # Process remaining buffer
        if buffer:
            for token_id in self.encode(buffer):
                yield token_id

    def _find_safe_split_point(self, text: str) -> int:
        """
        Find a safe point to split text for streaming encoding.
        We need to be careful not to split in the middle of:
        1. A potential special token
        2. A whitespace sequence (to preserve tokens like '\\n\\n')
        """
        if not text:
            return 0
        
        # Check if any special token could be starting at the end
        max_special_len = max((len(s) for s in self.special_tokens), default=0)
        
        # We need to keep at least max_special_len - 1 characters in buffer
        # to avoid splitting a special token
        min_keep = max_special_len - 1 if max_special_len > 0 else 0
        
        if len(text) <= min_keep:
            return 0
        
        safe_end = len(text)
        
        # Check for partial special token matches at the end
        for special in self.special_tokens:
            # Check if any prefix of special token matches end of text
            for prefix_len in range(1, len(special)):
                prefix = special[:prefix_len]
                if text.endswith(prefix):
                    safe_end = min(safe_end, len(text) - prefix_len)
        
        # Don't split in the middle of trailing whitespace
        # This prevents breaking up tokens like '\n\n'
        if safe_end > 0:
            # Find the last non-whitespace character
            last_non_ws = safe_end - 1
            while last_non_ws >= 0 and text[last_non_ws].isspace():
                last_non_ws -= 1
            
            # If there's trailing whitespace, don't include it in this chunk
            # unless the entire text is whitespace
            if last_non_ws >= 0 and last_non_ws < safe_end - 1:
                safe_end = last_non_ws + 1
        
        return safe_end


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Tokenizer:
    """
    Create a tokenizer from vocabulary and merge rules.
    
    Args:
        vocab: Mapping from token ID to bytes
        merges: List of BPE merge pairs
        special_tokens: Optional list of special token strings
        
    Returns:
        Tokenizer instance
    """
    return Tokenizer(vocab, merges, special_tokens)