"""BPE tokenizer wrapper around the HuggingFace `tokenizers` library.

Trains a byte-level BPE tokenizer (GPT-2 style) on a text corpus. The byte-level
pre-tokenizer means any unicode string round-trips losslessly through the
vocabulary — no out-of-vocab tokens, no `<unk>` needed in practice (but kept for
safety).
"""
from __future__ import annotations

import os
from collections.abc import Iterable

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDec
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPre
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


class BPETokenizer:
    """Byte-level BPE tokenizer with a clean train / save / load API."""

    def __init__(self, vocab_size: int = 8192):
        self._target_vocab_size = vocab_size
        self._tok: Tokenizer | None = None

    def train(
        self,
        files: Iterable[str] | str,
        save_path: str | None = None,
        min_frequency: int = 2,
        limit_alphabet: int = 1000,
    ) -> "BPETokenizer":
        if isinstance(files, str):
            files = [files]

        tok = Tokenizer(BPE(unk_token="<unk>"))
        tok.pre_tokenizer = ByteLevelPre(add_prefix_space=False)
        tok.decoder = ByteLevelDec()

        trainer = BpeTrainer(
            vocab_size=self._target_vocab_size,
            special_tokens=SPECIAL_TOKENS,
            min_frequency=min_frequency,
            initial_alphabet=ByteLevelPre.alphabet(),
            limit_alphabet=limit_alphabet,
            show_progress=True,
        )
        tok.train(list(files), trainer)
        self._tok = tok
        if save_path is not None:
            self.save(save_path)
        return self

    def train_from_iterator(
        self,
        iterator: Iterable[str],
        save_path: str | None = None,
        min_frequency: int = 2,
        limit_alphabet: int = 1000,
    ) -> "BPETokenizer":
        tok = Tokenizer(BPE(unk_token="<unk>"))
        tok.pre_tokenizer = ByteLevelPre(add_prefix_space=False)
        tok.decoder = ByteLevelDec()

        trainer = BpeTrainer(
            vocab_size=self._target_vocab_size,
            special_tokens=SPECIAL_TOKENS,
            min_frequency=min_frequency,
            initial_alphabet=ByteLevelPre.alphabet(),
            limit_alphabet=limit_alphabet,
            show_progress=True,
        )
        tok.train_from_iterator(iterator, trainer)
        self._tok = tok
        if save_path is not None:
            self.save(save_path)
        return self

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        assert self._tok is not None, "tokenizer not trained/loaded"
        self._tok.save(path)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        inst = cls.__new__(cls)
        inst._target_vocab_size = None
        inst._tok = Tokenizer.from_file(path)
        return inst

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [enc.ids for enc in self._tok.encode_batch(texts)]

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)

    @property
    def vocab(self) -> dict[str, int]:
        return self._tok.get_vocab()

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def pad_id(self) -> int:
        return self._tok.token_to_id("<pad>")

    @property
    def bos_id(self) -> int:
        return self._tok.token_to_id("<bos>")

    @property
    def eos_id(self) -> int:
        return self._tok.token_to_id("<eos>")

    @property
    def unk_id(self) -> int:
        return self._tok.token_to_id("<unk>")

    def token_to_id(self, token: str) -> int:
        return self._tok.token_to_id(token)

    def id_to_token(self, idx: int) -> str:
        return self._tok.id_to_token(idx)