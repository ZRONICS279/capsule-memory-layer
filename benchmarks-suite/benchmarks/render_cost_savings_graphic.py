#!/usr/bin/env python3
"""Render a cost-savings graphic from benchmark reports."""

from __future__ import annotations

import argparse
import json
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


def _load(report_dir: Path, name: str) -> dict:
    with (report_dir / name).open() as fh:
        return json.load(fh)


def _score(report: dict, adapter: str) -> dict:
    return report["summary"]["adapter_scores"][adapter]


def _cost_tokens_per_clean_success(score: dict) -> float | None:
    success = score.get("memory_success_rate")
    tokens = score.get("avg_total_tokens")
    if not success or not tokens:
        return None
    return tokens / success


def _money(tokens: float, blended_price_per_million: float) -> float:
    return tokens / 1_000_000 * blended_price_per_million


def render(report_dir: Path, out: Path, *, blended_price_per_million: float) -> None:
    full = _load(report_dir, "reports-all-agentic.json")
    adapter_order = [
        ("capsule", "Capsule"),
        ("mem0_real_async", "mem0 async"),
        ("langgraph_real", "LangGraph"),
        ("bm25_rag", "BM25 RAG"),
        ("vector_rag", "Vector RAG"),
    ]
    metrics = []
    adapter_scores = full.get("summary", {}).get("adapter_scores", {})
    for adapter, label in adapter_order:
        if adapter not in adapter_scores:
            continue
        score = _score(full, adapter)
        cost_tokens = _cost_tokens_per_clean_success(score)
        total_tokens = float(score.get("avg_total_tokens", 0) or 0)
        metrics.append(
            {
                "adapter": adapter,
                "label": label,
                "cost_tokens": cost_tokens,
                "tokens": total_tokens,
                "memok": score.get("memory_success_rate") or 0.0,
                "success": score.get("success_rate") or 0.0,
                "score": score.get("avg_cross_session_score", 0.0),
            }
        )

    width, height = 1800, 1450
    bg = (250, 249, 246)
    text = (31, 31, 31)
    muted = (95, 95, 95)
    line = (224, 220, 214)
    peach = (247, 219, 207)
    peach_line = (214, 139, 112)
    gray = (225, 222, 216)
    dark_gray = (184, 181, 176)

    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)
    title_font = _font(62, True)
    subtitle_font = _font(27)
    label_font = _font(28, True)
    value_font = _font(25, True)
    small_font = _font(21)
    callout_font = _font(34, True)

    title = "Estimated Cost Per Clean Memory-Grounded Success"
    tw = draw.textbbox((0, 0), title, font=title_font)[2]
    draw.text(((width - tw) / 2, 70), title, font=title_font, fill=text)
    subtitle = "Full agentic coding benchmark: average agent tokens divided by MemOK"
    sw = draw.textbbox((0, 0), subtitle, font=subtitle_font)[2]
    draw.text(((width - sw) / 2, 148), subtitle, font=subtitle_font, fill=muted)

    chart_left, chart_right = 180, 1620
    axis_y = 830
    chart_top = 280
    chart_h = axis_y - chart_top
    bar_w = 170
    gap = 95
    finite_values = [item["cost_tokens"] for item in metrics if item["cost_tokens"] is not None]
    if not finite_values:
        raise SystemExit("No adapters with memory_success_rate were found in reports-all-agentic.json")
    max_value = max(finite_values) * 1.25

    # y-axis grid
    tick_step = 30_000 if max_value > 50_000 else 2_000
    max_tick = int(((max_value + tick_step - 1) // tick_step) * tick_step)
    for tick in range(0, max_tick + 1, tick_step):
        y = axis_y - int(chart_h * tick / max_value)
        draw.line((chart_left - 35, y, chart_right + 20, y), fill=line, width=1)
        label = "0" if tick == 0 else f"{tick // 1000}k"
        draw.text((chart_left - 95, y - 12), label, font=small_font, fill=muted)
    draw.line((chart_left - 35, axis_y, chart_right + 20, axis_y), fill=(160, 160, 160), width=2)

    x = chart_left
    for item in metrics:
        is_capsule = item["adapter"] == "capsule"
        fill = peach if is_capsule else gray
        outline = peach_line if is_capsule else dark_gray
        value = item["cost_tokens"]
        if value is not None:
            bar_h = max(4, int(chart_h * value / max_value))
            draw.rounded_rectangle(
                (x, axis_y - bar_h, x + bar_w, axis_y),
                radius=12,
                fill=fill,
                outline=outline,
                width=3 if is_capsule else 2,
            )
            token_text = f"{value / 1000:.1f}k"
            dollar_text = f"${_money(value, blended_price_per_million):.4f}"
            tw = draw.textbbox((0, 0), token_text, font=value_font)[2]
            draw.text((x + (bar_w - tw) / 2, axis_y - bar_h - 64), token_text, font=value_font, fill=text)
            dw = draw.textbbox((0, 0), dollar_text, font=small_font)[2]
            draw.text((x + (bar_w - dw) / 2, axis_y - bar_h - 34), dollar_text, font=small_font, fill=muted)
        else:
            draw.rounded_rectangle(
                (x, axis_y - 35, x + bar_w, axis_y),
                radius=12,
                fill=(240, 238, 233),
                outline=dark_gray,
                width=2,
            )
            no_text = "No clean"
            no_text_2 = "success"
            nw = draw.textbbox((0, 0), no_text, font=value_font)[2]
            nw2 = draw.textbbox((0, 0), no_text_2, font=value_font)[2]
            draw.text((x + (bar_w - nw) / 2, axis_y - 122), no_text, font=value_font, fill=text)
            draw.text((x + (bar_w - nw2) / 2, axis_y - 92), no_text_2, font=value_font, fill=text)

        label_w = draw.textbbox((0, 0), item["label"], font=label_font)[2]
        draw.text((x + (bar_w - label_w) / 2, axis_y + 24), item["label"], font=label_font, fill=text)
        memok = f"MemOK {item['memok']:.1%}"
        success = f"Success {item['success']:.1%}"
        score = f"Score {item['score']:.1%}"
        mw = draw.textbbox((0, 0), memok, font=small_font)[2]
        suw = draw.textbbox((0, 0), success, font=small_font)[2]
        sw = draw.textbbox((0, 0), score, font=small_font)[2]
        draw.text((x + (bar_w - suw) / 2, axis_y + 60), success, font=small_font, fill=muted)
        draw.text((x + (bar_w - mw) / 2, axis_y + 82), memok, font=small_font, fill=muted)
        draw.text((x + (bar_w - sw) / 2, axis_y + 104), score, font=small_font, fill=muted)
        x += bar_w + gap

    capsule_cost = metrics[0]["cost_tokens"] or 0
    mem0_cost = metrics[1]["cost_tokens"] or 0
    relative = 1 - capsule_cost / mem0_cost if capsule_cost and mem0_cost else 0
    callout = f"Capsule used ~{relative:.1%} fewer tokens per clean success than mem0"
    cw = draw.textbbox((0, 0), callout, font=callout_font)[2]
    draw.text(((width - cw) / 2, 1000), callout, font=callout_font, fill=peach_line)

    box = (160, 1090, 1640, 1255)
    draw.rounded_rectangle(box, radius=18, fill=(255, 255, 253), outline=peach_line, width=3)
    draw.text((205, 1125), "How to read this", font=label_font, fill=text)
    draw.text(
        (205, 1177),
        "Bars show estimated cost per clean memory-grounded success: average agent tokens / MemOK.",
        font=small_font,
        fill=muted,
    )
    draw.text(
        (205, 1217),
        "Adapters with 0% MemOK had no clean memory-grounded successes in this run, so cost per clean success is undefined.",
        font=small_font,
        fill=muted,
    )

    foot = (
        "Conservative token-only estimate from benchmark reports. It does not include engineering time saved, fewer failed PR attempts, "
        "or provider-specific storage/embedding fees."
    )
    draw.text((120, 1370), foot, font=small_font, fill=muted)

    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", default="benchtestResults/run-20260620-130940")
    parser.add_argument("--out", default="benchtestResults/run-20260620-130940/researchpaper/cost-savings-graphic.png")
    parser.add_argument("--blended-price-per-million", type=float, default=2.0)
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    out = Path(args.out)
    if not report_dir.is_absolute():
        report_dir = ROOT / report_dir
    if not out.is_absolute():
        out = ROOT / out
    render(report_dir, out, blended_price_per_million=args.blended_price_per_million)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
