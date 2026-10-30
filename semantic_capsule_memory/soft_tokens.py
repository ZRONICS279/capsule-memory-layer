"""Trainable semantic capsule tokens.

This module is the real target architecture. It requires PyTorch and, for the
helper training script, Hugging Face Transformers. The rest of the package can
run without those dependencies, but this path is what turns a large context into
N learned vectors that can be passed through `inputs_embeds` instead of text.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SoftCapsuleConfig:
    vocab_size: int
    token_dim: int = 768
    capsule_tokens: int = 32
    encoder_layers: int = 4
    encoder_heads: int = 8
    ff_multiplier: int = 4
    dropout: float = 0.0
    max_context_tokens: int = 8192


def require_torch():
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Soft token capsules require PyTorch. Install torch to train or run this module."
        ) from exc
    return torch, nn


class SoftCapsuleEncoder:
    """Factory wrapper so importing the package does not require torch."""

    @staticmethod
    def build(config: SoftCapsuleConfig):
        torch, nn = require_torch()

        class _Encoder(nn.Module):
            """Compress token IDs into learned memory vectors.

            The learned queries are the capsule slots. Cross-attention lets each
            slot pull a different slice of meaning out of the encoded long
            context. The output is shaped `[batch, capsule_tokens, token_dim]`
            and can be prepended to an LLM's question embeddings.
            """

            def __init__(self):
                super().__init__()
                self.config = config
                self.embedding = nn.Embedding(config.vocab_size, config.token_dim)
                layer = nn.TransformerEncoderLayer(
                    d_model=config.token_dim,
                    nhead=config.encoder_heads,
                    dim_feedforward=config.token_dim * config.ff_multiplier,
                    dropout=config.dropout,
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=config.encoder_layers)
                self.queries = nn.Parameter(torch.randn(config.capsule_tokens, config.token_dim) * 0.02)
                self.cross_attention = nn.MultiheadAttention(
                    embed_dim=config.token_dim,
                    num_heads=config.encoder_heads,
                    dropout=config.dropout,
                    batch_first=True,
                )
                self.norm = nn.LayerNorm(config.token_dim)

            def forward(self, context_ids, context_mask=None):
                if context_ids.shape[1] > config.max_context_tokens:
                    context_ids = context_ids[:, -config.max_context_tokens :]
                    if context_mask is not None:
                        context_mask = context_mask[:, -config.max_context_tokens :]

                hidden = self.embedding(context_ids)
                padding_mask = None if context_mask is None else context_mask == 0
                hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)

                batch = context_ids.shape[0]
                queries = self.queries.unsqueeze(0).expand(batch, -1, -1)
                capsule, _ = self.cross_attention(
                    query=queries,
                    key=hidden,
                    value=hidden,
                    key_padding_mask=padding_mask,
                    need_weights=False,
                )
                return self.norm(capsule)

        return _Encoder()


class CapsuleProjector:
    """Project capsule vectors into a target LLM embedding space."""

    @staticmethod
    def build(capsule_dim: int, llm_dim: int):
        torch, nn = require_torch()

        class _Projector(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.LayerNorm(capsule_dim),
                    nn.Linear(capsule_dim, llm_dim),
                    nn.GELU(),
                    nn.Linear(llm_dim, llm_dim),
                )

            def forward(self, capsule_vectors):
                return self.net(capsule_vectors)

        return _Projector()


def build_inputs_embeds(llm, capsule_embeds, question_ids):
    """Prepend capsule embeddings to question embeddings for a causal LLM.

    `llm` must expose `get_input_embeddings()`, which Hugging Face causal LMs do.
    """

    question_embeds = llm.get_input_embeddings()(question_ids)
    return require_torch()[0].cat([capsule_embeds, question_embeds], dim=1)


def build_causal_labels(question_ids, answer_ids, capsule_tokens: int, ignore_index: int = -100):
    """Create labels that train only on answer tokens.

    Capsule and question positions are masked so the loss rewards answering from
    the compressed memory, not reproducing the input.
    """

    torch, _ = require_torch()
    batch = question_ids.shape[0]
    masked_prefix = torch.full(
        (batch, capsule_tokens + question_ids.shape[1]),
        ignore_index,
        dtype=answer_ids.dtype,
        device=answer_ids.device,
    )
    return torch.cat([masked_prefix, answer_ids], dim=1)

