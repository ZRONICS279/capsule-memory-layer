#!/usr/bin/env python3
"""Render a compact benchmark scorecard from benchmark JSON reports."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Pillow is required to render the scorecard. Install with: python3 -m pip install pillow"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RowSpec:
    label: str
    subtitle: str
    report: str
    note: str


ROWS = [
    RowSpec(
        "Long-horizon retrieval",
        "100+ event project-memory stress",
        "reports-long-horizon-retrieval.json",
        "stale facts + token pressure",
    ),
    RowSpec(
        "Long-horizon agentic",
        "end-to-end coding solve",
        "reports-long-horizon-agentic.json",
        "memory used by an agent",
    ),
    RowSpec(
        "Full retrieval",
        "all software-memory families",
        "reports-all-retrieval.json",
        "broad retrieval-only eval",
    ),
    RowSpec(
        "Full retrieval + baselines",
        "mem0, BM25, LangGraph, vector RAG",
        "reports-all-retrieval-expanded.json",
        "expanded baseline sweep",
    ),
    RowSpec(
        "Full agentic coding",
        "all families, repeated solves",
        "reports-all-agentic.json",
        "task success + stale handling",
    ),
    RowSpec(
        "Persistent two-phase",
        "ingest once, retrieve later",
        "reports-persistent-retrieval.json",
        "real product lifecycle",
    ),
    RowSpec(
        "General chatbot memory",
        "preferences, deletion, privacy",
        "reports-chat-memory.json",
        "not a coding-only workload",
    ),
]


COLUMNS = [
    ("capsule", "Capsule"),
    ("mem0_real_async", "mem0 async"),
    ("bm25_rag", "BM25 RAG"),
    ("langgraph_real", "LangGraph"),
    ("vector_rag", "Vector RAG"),
    ("no_memory", "No memory"),
]


def _load_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as fh:
        return json.load(fh)


def _adapter_scores(report: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not report:
        return {}
    return report.get("summary", {}).get("adapter_scores", {}) or {}


def _score(adapter_data: dict[str, Any] | None) -> float | None:
    if not adapter_data:
        return None
    value = adapter_data.get("avg_cross_session_score")
    return float(value) if value is not None else None


def _success(adapter_data: dict[str, Any] | None) -> float | None:
    if not adapter_data:
        return None
    value = adapter_data.get("success_rate")
    return float(value) if value is not None else None


def _memory_success(adapter_data: dict[str, Any] | None) -> float | None:
    if not adapter_data:
        return None
    value = adapter_data.get("memory_success_rate")
    return float(value) if value is not None else None


def _ctxok(adapter_data: dict[str, Any] | None) -> float | None:
    if not adapter_data:
        return None
    value = adapter_data.get("context_success_rate")
    return float(value) if value is not None else None


def _recall(adapter_data: dict[str, Any] | None) -> float | None:
    if not adapter_data:
        return None
    value = adapter_data.get("avg_recall")
    return float(value) if value is not None else None


def _stale(adapter_data: dict[str, Any] | None) -> float | None:
    if not adapter_data:
        return None
    value = adapter_data.get("avg_stale_penalty")
    return float(value) if value is not None else None


def _tokens(adapter_data: dict[str, Any] | None) -> float | None:
    if not adapter_data:
        return None
    value = adapter_data.get("avg_token_savings")
    return float(value) if value is not None else None


def _prepared_tokens(adapter_data: dict[str, Any] | None) -> int | None:
    if not adapter_data:
        return None
    value = adapter_data.get("avg_prepared_context_tokens")
    return int(round(float(value))) if value is not None else None


def _raw_tokens(adapter_data: dict[str, Any] | None) -> int | None:
    if not adapter_data:
        return None
    value = adapter_data.get("avg_raw_context_tokens")
    return int(round(float(value))) if value is not None else None


def _ingest_seconds(adapter_data: dict[str, Any] | None) -> float | None:
    if not adapter_data:
        return None
    value = adapter_data.get("avg_ingest_duration_seconds")
    return float(value) if value is not None else None


def _pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def _seconds(value: float | None) -> str:
    if value is None:
        return "-"
    if value < 0.05:
        return "<0.1s"
    if value < 10:
        return f"{value:.1f}s"
    return f"{value:.0f}s"


def _token_count(value: int | None) -> str:
    if value is None:
        return "-"
    if value >= 1000:
        compact = value / 1000
        return f"{compact:.1f}k".replace(".0k", "k")
    return str(value)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf",
                "/Library/Fonts/Arial Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            ]
        )
    candidates.extend(
        [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttf",
            "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _center_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    x0, y0, x1, y1 = xy
    w, h = _text_size(draw, text, font)
    draw.text((x0 + (x1 - x0 - w) / 2, y0 + (y1 - y0 - h) / 2), text, font=font, fill=fill)


def build_matrix(report_dir: Path) -> list[dict[str, Any]]:
    matrix = []
    for row in ROWS:
        scores = _adapter_scores(_load_report(report_dir / row.report))
        values = {key: scores.get(key) for key, _ in COLUMNS}
        numeric_scores = {key: _score(data) for key, data in values.items()}
        best_score = max((v for v in numeric_scores.values() if v is not None), default=None)
        matrix.append(
            {
                "label": row.label,
                "subtitle": row.subtitle,
                "note": row.note,
                "report": row.report,
                "values": values,
                "scores": numeric_scores,
                "best_score": best_score,
            }
        )
    return matrix


def render_scorecard(report_dir: Path, out: Path, *, model_label: str = "") -> None:
    matrix = build_matrix(report_dir)

    width = 2600
    left = 105
    right = 70
    top = 90
    title_h = 165
    header_h = 88
    row_h = 278
    foot_h = 215
    label_w = 470
    col_w = (width - left - right - label_w) // len(COLUMNS)
    height = top + title_h + header_h + row_h * len(matrix) + foot_h

    colors = {
        "bg": (250, 249, 246),
        "text": (31, 31, 31),
        "muted": (104, 104, 104),
        "line": (224, 220, 214),
        "capsule_fill": (247, 219, 207),
        "capsule_border": (214, 139, 112),
        "best_fill": (238, 236, 229),
        "soft": (245, 243, 238),
    }

    image = Image.new("RGB", (width, height), colors["bg"])
    draw = ImageDraw.Draw(image)

    title_font = _font(64, bold=True)
    subtitle_font = _font(25)
    model_font = _font(21)
    header_font = _font(30, bold=True)
    row_font = _font(29, bold=True)
    small_font = _font(22)
    value_font = _font(32, bold=True)
    subvalue_font = _font(17)
    foot_font = _font(20)

    title = "Agentic Memory Benchmark Results"
    tw, th = _text_size(draw, title, title_font)
    draw.text(((width - tw) / 2, top), title, font=title_font, fill=colors["text"])
    subtitle = "Cross-session coding memory, stale fact handling, retrieval precision, and token efficiency"
    sw, _ = _text_size(draw, subtitle, subtitle_font)
    draw.text(((width - sw) / 2, top + th + 18), subtitle, font=subtitle_font, fill=colors["muted"])
    model_text = model_label or "Agent model: gpt-4.1-mini for agentic coding rows; retrieval-only rows use memory retrieval only"
    mw, _ = _text_size(draw, model_text, model_font)
    draw.text(((width - mw) / 2, top + th + 55), model_text, font=model_font, fill=colors["muted"])

    table_top = top + title_h
    header_top = table_top
    first_col_x = left + label_w
    capsule_x0 = first_col_x
    capsule_x1 = first_col_x + col_w
    table_bottom = table_top + header_h + row_h * len(matrix)

    draw.rounded_rectangle(
        (capsule_x0 + 5, header_top + 2, capsule_x1 - 5, table_bottom - 2),
        radius=22,
        fill=colors["capsule_fill"],
        outline=colors["capsule_border"],
        width=4,
    )

    draw.line((left, header_top + header_h, width - right, header_top + header_h), fill=colors["line"], width=2)
    for idx, (_, title_text) in enumerate(COLUMNS):
        x0 = first_col_x + idx * col_w
        x1 = x0 + col_w
        _center_text(draw, (x0, header_top, x1, header_top + header_h), title_text, header_font, colors["text"])

    for r, row in enumerate(matrix):
        y0 = header_top + header_h + r * row_h
        y1 = y0 + row_h
        draw.line((left, y1, width - right, y1), fill=colors["line"], width=2)

        draw.text((left + 12, y0 + 55), row["label"], font=row_font, fill=colors["text"])
        draw.text((left + 12, y0 + 98), row["subtitle"], font=small_font, fill=colors["muted"])
        draw.text((left + 12, y0 + 130), row["note"], font=small_font, fill=colors["muted"])

        for idx, (key, _) in enumerate(COLUMNS):
            x0 = first_col_x + idx * col_w
            x1 = x0 + col_w
            data = row["values"][key]
            score = row["scores"][key]
            is_best = score is not None and row["best_score"] is not None and abs(score - row["best_score"]) < 1e-9
            if is_best and key != "capsule":
                draw.rectangle((x0 + 6, y0 + 2, x1 - 6, y1 - 2), fill=colors["best_fill"])

            fill = colors["text"] if score is not None else colors["muted"]
            value = _pct(score)
            vw, vh = _text_size(draw, value, value_font)
            draw.text((x0 + (col_w - vw) / 2, y0 + 35), value, font=value_font, fill=fill)

            subparts = []
            success = _success(data)
            memory_success = _memory_success(data)
            ctxok = _ctxok(data)
            recall = _recall(data)
            stale = _stale(data)
            token_savings = _tokens(data)
            prepared_tokens = _prepared_tokens(data)
            ingest_seconds = _ingest_seconds(data)
            if recall is not None:
                subparts.append(f"Recall {_pct(recall)}")
            if ctxok is not None:
                subparts.append(f"CtxOK {_pct(ctxok)}")
            if success is not None:
                subparts.append(f"Success {_pct(success)}")
            if memory_success is not None:
                subparts.append(f"MemOK {_pct(memory_success)}")
            if stale is not None:
                subparts.append(f"Stale {_pct(stale)}")
            if token_savings is not None:
                subparts.append(f"Save {_pct(token_savings)}")
            if prepared_tokens is not None:
                subparts.append(f"CtxTok {_token_count(prepared_tokens)}")
            if ingest_seconds is not None:
                subparts.append(f"Ingest {_seconds(ingest_seconds)}")
            for offset, subline in enumerate(subparts[:8]):
                suw, _ = _text_size(draw, subline, subvalue_font)
                draw.text(
                    (x0 + (col_w - suw) / 2, y0 + 77 + offset * 21),
                    subline,
                    font=subvalue_font,
                    fill=colors["muted"],
                )

    foot_y = table_bottom + 36
    foot_1 = "Score is computed from required-fact recall after leakage penalties. In coding runs: Score = Recall x (1 - Stale leakage)."
    foot_2 = "CtxOK means all required current facts appeared in context and no stale facts leaked. MemOK means task success plus CtxOK."
    foot_3 = "A 100% score means every required current fact was retrieved with 0% stale leakage; lower scores mean missing facts and/or stale leakage."
    foot_4 = "Subtext shows Recall, CtxOK, task Success, MemOK when available, Stale leakage, token Save, CtxTok, and Ingest speed. Missing baselines are shown as '-'."
    draw.text((left, foot_y), foot_1, font=foot_font, fill=colors["muted"])
    draw.text((left, foot_y + 34), foot_2, font=foot_font, fill=colors["muted"])
    draw.text((left, foot_y + 68), foot_3, font=foot_font, fill=colors["muted"])
    draw.text((left, foot_y + 102), foot_4, font=foot_font, fill=colors["muted"])

    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)


def write_markdown(report_dir: Path, out: Path, *, model_label: str = "") -> None:
    matrix = build_matrix(report_dir)
    lines = [
        "# Agentic Memory Benchmark Scorecard",
        "",
        model_label or "Agent model: gpt-4.1-mini for agentic coding rows; retrieval-only rows use memory retrieval only.",
        "",
        "Each cell shows score, recall, CtxOK, stale leakage, token savings, prepared context tokens, ingest speed, and task success when available.",
        "",
        "Score is computed from required-fact recall after leakage penalties. In coding runs: `Score = Recall x (1 - Stale leakage)`. A 100% score means every required current fact was retrieved with 0% stale leakage.",
        "",
        "CtxOK means all required current facts appeared in context and no stale facts leaked. MemOK means task success plus CtxOK.",
        "",
        "| Benchmark | " + " | ".join(title for _, title in COLUMNS) + " |",
        "|---|" + "|".join("---" for _ in COLUMNS) + "|",
    ]
    for row in matrix:
        vals = []
        for key, _ in COLUMNS:
            data = row["values"][key]
            if not data:
                vals.append("-")
                continue
            parts = [
                f"Score {_pct(_score(data))}",
                f"Recall {_pct(_recall(data))}",
                f"CtxOK {_pct(_ctxok(data))}",
                f"Stale {_pct(_stale(data))}",
                f"Save {_pct(_tokens(data))}",
                f"CtxTok {_token_count(_prepared_tokens(data))}",
                f"RawTok {_token_count(_raw_tokens(data))}",
                f"Ingest {_seconds(_ingest_seconds(data))}",
            ]
            success = _success(data)
            if success is not None:
                parts.append(f"Success {_pct(success)}")
            memory_success = _memory_success(data)
            if memory_success is not None:
                parts.append(f"MemOK {_pct(memory_success)}")
            vals.append("<br>".join(parts))
        lines.append(f"| {row['label']} | " + " | ".join(vals) + " |")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", default=".", help="Directory containing report JSON files.")
    parser.add_argument("--out", default="researchpaper/benchmark-scorecard.png", help="Output PNG path.")
    parser.add_argument(
        "--markdown-out",
        default="researchpaper/benchmark-scorecard.md",
        help="Output Markdown table path.",
    )
    parser.add_argument(
        "--model-label",
        default="",
        help="Text to show below the title describing the agent/model used.",
    )
    args = parser.parse_args()

    report_dir = (ROOT / args.report_dir).resolve() if not Path(args.report_dir).is_absolute() else Path(args.report_dir)
    out = (ROOT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    md_out = (
        (ROOT / args.markdown_out).resolve()
        if not Path(args.markdown_out).is_absolute()
        else Path(args.markdown_out)
    )
    render_scorecard(report_dir, out, model_label=args.model_label)
    write_markdown(report_dir, md_out, model_label=args.model_label)
    print(f"Wrote {out}")
    print(f"Wrote {md_out}")


if __name__ == "__main__":
    main()
