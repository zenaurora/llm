"""Smoke tests for norm / ffn / tokenizer."""
from __future__ import annotations

import os
import tempfile

import torch

from llm.data import BPETokenizer
from llm.model import FFN, LayerNorm, RMSNorm, SwiGLU
from llm.utils import get_device


def test_norm():
    dev = get_device()
    x = torch.randn(2, 4, 16, device=dev)

    ln = LayerNorm(16).to(dev)
    y = ln(x)
    assert y.shape == x.shape
    assert torch.allclose(y.mean(dim=-1), torch.zeros(2, 4, device=dev), atol=1e-5)
    assert torch.allclose(y.std(dim=-1, unbiased=False), torch.ones(2, 4, device=dev), atol=1e-4)

    rms = RMSNorm(16).to(dev)
    y2 = rms(x)
    assert y2.shape == x.shape
    assert torch.allclose(y2.pow(2).mean(dim=-1).sqrt(), torch.ones(2, 4, device=dev), atol=1e-4)

    rms_bias = RMSNorm(16, bias=True).to(dev)
    y3 = rms_bias(x)
    assert y3.shape == x.shape

    print("[ok] LayerNorm & RMSNorm")


def test_ffn():
    dev = get_device()
    x = torch.randn(2, 4, 32, device=dev)

    ffn = FFN(32, hidden_dim=128, activation="gelu").to(dev)
    y = ffn(x)
    assert y.shape == x.shape

    swiglu = SwiGLU(32).to(dev)
    z = swiglu(x)
    assert z.shape == x.shape

    print("[ok] FFN & SwiGLU")


def test_tokenizer(tmp_corpus: str):
    tok = BPETokenizer(vocab_size=512)
    tok.train(tmp_corpus)

    text = "Hello, world! 你好，世界。"
    ids = tok.encode(text)
    roundtrip = tok.decode(ids)

    assert len(ids) > 0
    assert roundtrip == text, f"round-trip failed:\n  in: {text}\n  out: {roundtrip}"

    print(f"[ok] BPETokenizer  vocab={tok.vocab_size}  ids={ids[:10]}...")

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "tok.json")
        tok.save(path)
        reloaded = BPETokenizer.load(path)
        assert reloaded.encode(text) == ids
    print("[ok] BPETokenizer save/load")


def make_corpus(path: str) -> None:
    lines = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello, world! This is a tiny English/中文 mixed corpus.",
        "PyTorch on Apple Silicon with MPS acceleration works well.",
        "Tokenization splits text into subword units for neural models.",
        "Byte-level BPE can represent any unicode string losslessly.",
        "Transformer models use self-attention and feed-forward layers.",
        "RMSNorm is faster than LayerNorm with similar quality.",
        "SwiGLU activations outperform plain ReLU in modern LLMs.",
    ] * 50
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    with tempfile.TemporaryDirectory() as td:
        corpus = os.path.join(td, "corpus.txt")
        make_corpus(corpus)

        test_norm()
        test_ffn()
        test_tokenizer(corpus)

    print("\nall good. device:", get_device())


if __name__ == "__main__":
    main()