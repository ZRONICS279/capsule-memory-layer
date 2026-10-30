"""Train soft semantic capsules against a frozen causal LLM.

Input JSONL schema:

```json
{"context": "...long prior context...", "question": "...", "answer": "..."}
```

This is intentionally written as the direct research path:

1. Encode long context into 16-128 soft capsule tokens.
2. Project capsules into the frozen LLM embedding space.
3. Feed `[capsule_embeds, question_embeds, answer_embeds]` with labels masked
   so only answer tokens train.
4. Optimize the capsule encoder/projector, not the base LLM.

The current execution environment does not include torch/transformers, so this
script is provided as runnable project code once those dependencies are present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from semantic_capsule_memory.soft_tokens import (
    CapsuleProjector,
    SoftCapsuleConfig,
    SoftCapsuleEncoder,
    build_causal_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face causal LM name or local path")
    parser.add_argument("--data", required=True, type=Path, help="JSONL training records")
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    parser.add_argument("--capsule-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-context-tokens", type=int, default=8192)
    parser.add_argument("--max-question-tokens", type=int, default=512)
    parser.add_argument("--max-answer-tokens", type=int, default=512)
    return parser.parse_args()


def iter_records(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> int:
    args = parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit("Install torch and transformers to run soft-capsule training.") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(args.model)
    llm.eval()
    for parameter in llm.parameters():
        parameter.requires_grad_(False)

    llm_dim = llm.get_input_embeddings().embedding_dim
    config = SoftCapsuleConfig(
        vocab_size=len(tokenizer),
        token_dim=llm_dim,
        capsule_tokens=args.capsule_tokens,
        max_context_tokens=args.max_context_tokens,
    )
    encoder = SoftCapsuleEncoder.build(config)
    projector = CapsuleProjector.build(config.token_dim, llm_dim)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=args.lr,
    )

    records = list(iter_records(args.data))
    args.output.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        total_loss = 0.0
        for record in records:
            context = tokenizer(
                record["context"],
                return_tensors="pt",
                truncation=True,
                max_length=args.max_context_tokens,
            )
            question = tokenizer(
                record["question"],
                return_tensors="pt",
                truncation=True,
                max_length=args.max_question_tokens,
            )
            answer = tokenizer(
                record["answer"] + tokenizer.eos_token,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_answer_tokens,
            )

            capsule = encoder(context["input_ids"], context.get("attention_mask"))
            capsule = projector(capsule)
            question_embeds = llm.get_input_embeddings()(question["input_ids"])
            answer_embeds = llm.get_input_embeddings()(answer["input_ids"])
            inputs_embeds = torch.cat([capsule, question_embeds, answer_embeds], dim=1)

            labels = build_causal_labels(
                question["input_ids"],
                answer["input_ids"],
                capsule_tokens=args.capsule_tokens,
            )
            outputs = llm(inputs_embeds=inputs_embeds, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach())

        average = total_loss / max(len(records), 1)
        print(json.dumps({"epoch": epoch + 1, "loss": average}))

    torch.save(
        {
            "config": config,
            "encoder": encoder.state_dict(),
            "projector": projector.state_dict(),
        },
        args.output / "soft_capsule.pt",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

