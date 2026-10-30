#!/usr/bin/env python3
"""Render a compact methods graphic explaining the benchmark setup."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Pillow is required. Install with: python3 -m pip install pillow") from exc


ROOT = Path(__file__).resolve().parents[1]


def _font(size: int, bold: bool = False):
    candidates = []
    if bold:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/Library/Fonts/Arial Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            ]
        )
    candidates.extend(
        [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    max_width: int,
    font,
    fill: tuple[int, int, int],
    line_gap: int = 8,
) -> int:
    for line in _wrap(draw, text, font, max_width):
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + line_gap
    return y


def _card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    body: list[str],
    *,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    title_font,
    body_font,
    text: tuple[int, int, int],
    muted: tuple[int, int, int],
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=18, fill=fill, outline=outline, width=2)
    draw.text((x0 + 28, y0 + 24), title, font=title_font, fill=text)
    y = y0 + 78
    for item in body:
        draw.ellipse((x0 + 30, y + 9, x0 + 39, y + 18), fill=outline)
        y = _draw_wrapped(draw, item, x0 + 54, y, x1 - x0 - 84, body_font, muted, line_gap=6)
        y += 12


def render(out: Path) -> None:
    width, height = 1800, 1800
    bg = (250, 249, 246)
    text = (31, 31, 31)
    muted = (91, 91, 91)
    line = (224, 220, 214)
    peach = (247, 219, 207)
    peach_line = (214, 139, 112)
    soft = (244, 242, 236)

    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)

    title_font = _font(62, True)
    subtitle_font = _font(28)
    card_title = _font(34, True)
    body_font = _font(24)
    formula_font = _font(30, True)
    foot_font = _font(22)

    title = "How The Agentic Memory Benchmark Works"
    tw = draw.textbbox((0, 0), title, font=title_font)[2]
    draw.text(((width - tw) / 2, 75), title, font=title_font, fill=text)
    subtitle = "Multi-session coding tasks with changing decisions, stale facts, and token pressure"
    sw = draw.textbbox((0, 0), subtitle, font=subtitle_font)[2]
    draw.text(((width - sw) / 2, 150), subtitle, font=subtitle_font, fill=muted)

    cards = [
        (
            (90, 250, 850, 650),
            "1. Simulated Project History",
            [
                "Each case starts with several past work sessions, like a real repo worked on over days or weeks.",
                "The history includes auth migrations, route changes, config updates, feature flags, ownership changes, dependency pins, and long-horizon project notes.",
                "Some earlier decisions are intentionally superseded by newer final decisions.",
            ],
            soft,
            line,
        ),
        (
            (950, 250, 1710, 650),
            "2. Memory System Retrieval",
            [
                "Each memory layer ingests the same sessions, then receives the same live task.",
                "It must retrieve the final current facts the agent needs, while avoiding stale facts from earlier sessions.",
                "Adapters tested include Capsule, mem0 async, BM25 RAG, Vector RAG, LangGraph, and no-memory.",
            ],
            peach,
            peach_line,
        ),
        (
            (90, 730, 850, 1130),
            "3. Agent Coding Task",
            [
                "For agentic rows, the retrieved memory is given to a coding agent using gpt-4.1-mini.",
                "The agent edits a temporary repo fixture and runs the target test command.",
                "Tasks are only considered memory-grounded when the agent solves the task and the memory context was clean.",
            ],
            peach,
            peach_line,
        ),
        (
            (950, 730, 1710, 1130),
            "4. What Was Scored",
            [
                "Recall: did the memory include the required current facts?",
                "Stale leakage: did old rejected facts show up as active context?",
                "CtxOK: all required current facts are present and no stale facts leaked.",
                "MemOK: the agent solved the task and CtxOK was true.",
            ],
            soft,
            line,
        ),
    ]

    for box, title_text, body, fill, outline in cards:
        _card(
            draw,
            box,
            title_text,
            body,
            fill=fill,
            outline=outline,
            title_font=card_title,
            body_font=body_font,
            text=text,
            muted=muted,
        )

    formula_box = (210, 1225, 1590, 1515)
    draw.rounded_rectangle(formula_box, radius=22, fill=(255, 255, 253), outline=peach_line, width=3)
    fx0, fy0, fx1, _ = formula_box
    draw.text((fx0 + 42, fy0 + 34), "Headline Score", font=card_title, fill=text)
    formula = "Score = Recall x (1 - Stale Leakage)"
    fw = draw.textbbox((0, 0), formula, font=formula_font)[2]
    draw.text((fx0 + (fx1 - fx0 - fw) / 2, fy0 + 105), formula, font=formula_font, fill=text)
    explanation = (
        "A 100% score means the memory retrieved every required current fact and leaked 0% stale facts. "
        "Lower scores mean missing facts, stale facts, or both."
    )
    _draw_wrapped(draw, explanation, fx0 + 42, fy0 + 170, fx1 - fx0 - 84, body_font, muted)

    foot = (
        "The benchmark separates raw task success from memory quality because an agent can sometimes pass tests "
        "even when memory was missing or stale. MemOK is the stricter agentic-memory metric."
    )
    y = _draw_wrapped(draw, foot, 120, 1600, width - 240, foot_font, muted)
    draw.text((120, y + 22), "Generated from the local benchmark suite.", font=foot_font, fill=muted)

    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="benchtestResults/run-20260620-130940/researchpaper/benchmark-methods-graphic.png",
    )
    args = parser.parse_args()
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    render(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
