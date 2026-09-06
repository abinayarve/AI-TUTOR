"""
frame_renderer.py
==================

Renders the actual pixel frames used to build a real MP4 teaching video:

- An avatar frame (simple, friendly cartoon face with mouth open/closed
  states for a lightweight "talking" animation synced to narration audio).
- A subject-aware visual panel, driven by the SAME structured
  ``visual_data`` the lesson generator produces, so charts/flowcharts/
  timelines/tables are specific to the actual concept being taught
  instead of a generic placeholder shape.
- A composed scene frame that combines both plus a caption/title bar.

This module is intentionally independent of Streamlit, Gemini, and the
rest of the app so it can be unit-tested and reused by both:
  1. video_engine.py (to build the real MP4 file), and
  2. app.py (to keep the on-screen visual and the video visual consistent).

No API key or network access is required.
"""

from __future__ import annotations

import io
import textwrap
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


# ============================================================
# THEME
# ============================================================

BG_LEFT = (24, 28, 46)
BG_RIGHT = (255, 255, 255)
ACCENT = (99, 102, 241)
ACCENT_SOFT = (199, 210, 254)
TEXT_LIGHT = (240, 240, 250)
TEXT_DARK = (30, 30, 40)
SKIN_TONE = (247, 200, 160)
HAPPY = (16, 185, 129)
CONCERNED = (244, 114, 182)


# ============================================================
# FONT HELPERS
# ============================================================

def _get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Best-effort font loader that always returns something usable."""

    candidates = (
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        if bold
        else [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )

    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue

    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def _wrap(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> List[str]:
    """Wrap text to fit max_width, measured with the actual font."""

    text = str(text or "").strip()
    if not text:
        return []

    words = text.split()
    lines: List[str] = []
    current = ""

    for word in words:
        trial = f"{current} {word}".strip()
        width = draw.textlength(trial, font=font)
        if width <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines


# ============================================================
# AVATAR FRAME
# ============================================================

EXPRESSION_ACCENT = {
    "happy": HAPPY,
    "encouraging": HAPPY,
    "concerned": CONCERNED,
    "questioning": ACCENT,
    "thinking": ACCENT_SOFT,
    "explaining": ACCENT,
    "neutral": ACCENT_SOFT,
}


def render_avatar_frame(
    width: int,
    height: int,
    expression: str = "explaining",
    mouth_open: bool = True,
    teacher_name: str = "Prof. AI",
    language: str = "",
    motion_t: float = 0.0,
    blink: bool = False,
    gesture_phase: float = 0.0,
    mouth_state: int = 1,
) -> Image.Image:
    """
    Draw a lightweight, animated illustrated avatar frame.

    This is intentionally an illustrated face (not a photoreal/
    generative avatar) so it renders instantly and deterministically
    without any external model or GPU — but it is NOT a single static
    image with a flapping mouth. Every frame reflects continuous idle
    motion so a sequence of frames reads as genuinely "alive":

    - ``motion_t``: seconds of continuous phase used to drive a slow
      head bob + sway and shoulder sway (independent of clip length).
    - ``blink``: draw closed/squinting eyes for this frame.
    - ``gesture_phase``: 0..1 progress used to animate a simple
      speaking hand gesture that rises and falls.
    - ``mouth_state``: 0=closed, 1=half-open, 2=wide-open — gives a
      3-shape lip-sync flap instead of a binary on/off mouth.
    """

    import math

    accent = EXPRESSION_ACCENT.get(expression, ACCENT)

    img = Image.new("RGB", (width, height), BG_LEFT)
    draw = ImageDraw.Draw(img)

    # Soft radial-ish backdrop using concentric rectangles.
    for i in range(6):
        shade = 24 + i * 4
        draw.rectangle(
            [i * 6, i * 6, width - i * 6, height - i * 6],
            outline=(shade, shade, shade + 20),
        )

    # Continuous idle motion: slow head bob + gentle side-to-side sway.
    bob = int(round(4 * math.sin(2 * math.pi * 0.28 * motion_t)))
    sway = int(round(3 * math.sin(2 * math.pi * 0.18 * motion_t + 1.2)))

    base_cx, base_cy = width // 2, int(height * 0.42)
    cx, cy = base_cx + sway, base_cy + bob
    head_r = int(min(width, height) * 0.24)

    # Shoulders / torso (sways with the same phase for a whole-body feel).
    shoulder_w = int(width * 0.62)
    shoulder_top = cy + int(head_r * 1.15)
    draw.rounded_rectangle(
        [
            base_cx - shoulder_w // 2 + sway // 2,
            shoulder_top,
            base_cx + shoulder_w // 2 + sway // 2,
            height,
        ],
        radius=28,
        fill=(45, 50, 74),
        outline=accent,
        width=3,
    )

    # Speaking hand gesture: a small hand that rises near the shoulder
    # while explaining, following gesture_phase (0..1 sawtooth per beat).
    if expression not in ("thinking",):
        gesture_lift = int(round(18 * math.sin(math.pi * gesture_phase)))
        hand_x = base_cx + int(shoulder_w * 0.34) + sway // 2
        hand_y = shoulder_top - 6 - gesture_lift
        hand_r = max(6, int(head_r * 0.16))
        # forearm
        draw.line(
            [base_cx + int(shoulder_w * 0.30), shoulder_top + 6, hand_x, hand_y],
            fill=SKIN_TONE,
            width=max(6, int(head_r * 0.22)),
        )
        # hand
        draw.ellipse(
            [hand_x - hand_r, hand_y - hand_r, hand_x + hand_r, hand_y + hand_r],
            fill=SKIN_TONE,
            outline=accent,
            width=2,
        )

    # Head
    draw.ellipse(
        [cx - head_r, cy - head_r, cx + head_r, cy + head_r],
        fill=SKIN_TONE,
        outline=accent,
        width=4,
    )

    # Simple hair/hairline arc for a slightly more human silhouette.
    draw.arc(
        [cx - head_r, cy - head_r - int(head_r * 0.25), cx + head_r, cy + int(head_r * 0.35)],
        start=200,
        end=340,
        fill=(60, 45, 40),
        width=max(4, int(head_r * 0.22)),
    )

    # Eyes (closed to a thin line when blinking).
    eye_dy = -int(head_r * 0.15)
    eye_dx = int(head_r * 0.38)
    eye_r = max(4, int(head_r * 0.09))
    for sign in (-1, 1):
        ex = cx + sign * eye_dx
        ey = cy + eye_dy
        if blink:
            draw.line([ex - eye_r, ey, ex + eye_r, ey], fill=TEXT_DARK, width=3)
        else:
            draw.ellipse([ex - eye_r, ey - eye_r, ex + eye_r, ey + eye_r], fill=TEXT_DARK)

    # Eyebrows (expression-dependent, with a tiny bob-linked twitch).
    brow_dy = eye_dy - eye_r - 6 + (1 if blink else 0)
    brow_tilt = 4 if expression in ("concerned", "questioning") else -2
    for sign in (-1, 1):
        ex = cx + sign * eye_dx
        ey = cy + brow_dy
        draw.line(
            [ex - eye_r - 2, ey + sign * brow_tilt, ex + eye_r + 2, ey - sign * brow_tilt],
            fill=TEXT_DARK,
            width=3,
        )

    # Mouth — three shapes (closed / half / open) for a smoother flap
    # than a binary on-off toggle.
    mouth_w = int(head_r * 0.6)
    mouth_y = cy + int(head_r * 0.42)
    if mouth_state <= 0:
        draw.line(
            [cx - mouth_w // 2, mouth_y, cx + mouth_w // 2, mouth_y],
            fill=TEXT_DARK,
            width=4,
        )
    else:
        mouth_h = int(head_r * (0.16 if mouth_state == 1 else 0.32))
        draw.ellipse(
            [cx - mouth_w // 2, mouth_y - mouth_h // 2, cx + mouth_w // 2, mouth_y + mouth_h // 2],
            fill=(120, 40, 50),
            outline=TEXT_DARK,
            width=2,
        )

    # Name badge
    font_name = _get_font(int(height * 0.045), bold=True)
    font_sub = _get_font(int(height * 0.03))

    badge_y = base_cy + head_r + int(height * 0.14)
    draw.text((base_cx, badge_y), teacher_name, font=font_name, fill=TEXT_LIGHT, anchor="mm")

    if language:
        draw.text(
            (base_cx, badge_y + int(height * 0.05)),
            f"Teaching in {language}",
            font=font_sub,
            fill=ACCENT_SOFT,
            anchor="mm",
        )

    # Speaking indicator dot (pulses with the mouth state).
    dot_color = HAPPY if mouth_state > 0 else (90, 90, 110)
    draw.ellipse(
        [width - 28, height - 28, width - 12, height - 12],
        fill=dot_color,
    )

    return img


def split_caption_chunks(script: str, max_chunks: int = 10, max_chars: int = 140) -> List[str]:
    """Split a spoken_script into short caption chunks so on-screen text
    progresses over the clip instead of showing one frozen slice for
    the whole module (which reads as "static" even when the avatar
    moves)."""

    text = " ".join(str(script or "").split())
    if not text:
        return [""]

    # Prefer sentence boundaries; fall back to fixed-size slices.
    raw_sentences = [s.strip() for s in text.replace("!", ".").replace("?", "?.").split(".") if s.strip()]
    chunks: List[str] = []
    current = ""
    for sentence in raw_sentences:
        trial = f"{current} {sentence}.".strip()
        if len(trial) <= max_chars or not current:
            current = trial
        else:
            chunks.append(current)
            current = f"{sentence}."
    if current:
        chunks.append(current)

    if not chunks:
        chunks = [text[i:i + max_chars] for i in range(0, len(text), max_chars)] or [text]

    if len(chunks) > max_chunks:
        # Merge down to max_chunks roughly evenly.
        step = len(chunks) / max_chunks
        merged = []
        for i in range(max_chunks):
            part = chunks[int(i * step):int((i + 1) * step)] or [chunks[-1]]
            merged.append(" ".join(part))
        chunks = merged

    return chunks


# ============================================================
# VISUAL PANEL (topic-aware, data-driven)
# ============================================================

def _fig_to_image(fig, width: int, height: int) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    img = img.resize((width, height))
    return img


def _blank_panel(width: int, height: int) -> Tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (width, height), BG_RIGHT)
    draw = ImageDraw.Draw(img)
    return img, draw


def _derive_flow_steps(concept: str, visual_content: str) -> List[str]:
    """Best-effort, concept-specific fallback when the LLM didn't supply
    flow_steps. Pulls short phrases out of visual_content instead of using
    a fixed generic label set."""

    text = (visual_content or concept or "").strip()
    parts = [p.strip(" .") for p in text.replace(";", ",").split(",") if p.strip()]
    parts = [p for p in parts if 2 <= len(p) <= 40]

    if len(parts) >= 3:
        return parts[:6]

    if concept:
        return [f"Start: {concept}", "Key mechanism", f"Result of {concept}"]

    return ["Step 1", "Step 2", "Step 3"]


def _derive_timeline_events(concept: str, visual_content: str) -> List[str]:
    text = (visual_content or concept or "").strip()
    sentences = [s.strip() for s in text.replace("\n", ". ").split(".") if s.strip()]
    if len(sentences) >= 3:
        return sentences[:6]
    if concept:
        return [f"Introduction to {concept}", f"Core idea of {concept}", f"Why {concept} matters"]
    return ["Beginning", "Development", "Outcome"]


def render_visual_panel(
    width: int,
    height: int,
    visual_type: str,
    visual_content: str,
    visual_data: Optional[Dict[str, Any]],
    topic: str,
    concept: str,
) -> Image.Image:
    """
    Build the topic-aware visual panel image.

    Uses ``visual_data`` (produced by the LLM per module) when present;
    otherwise derives a concept-specific fallback from ``visual_content``
    instead of a fixed generic placeholder.
    """

    visual_type = str(visual_type or "TEXT").upper()
    visual_data = visual_data or {}
    label = concept or topic or "This concept"

    font_title = _get_font(int(height * 0.05), bold=True)
    font_body = _get_font(int(height * 0.038))

    # ---------------- GRAPH ----------------
    if visual_type == "GRAPH" and plt is not None:
        points = visual_data.get("chart_points") or []
        if points and len(points) >= 2:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
        else:
            # Concept-specific-looking fallback: derive a simple curve
            # from the concept name's length so different concepts at
            # least don't all render an identical parabola.
            seed = max(1, len(label) % 9 + 1)
            xs = list(range(1, 11))
            ys = [seed * (11 - x) for x in xs]

        fig, ax = plt.subplots(figsize=(width / 110, height / 110))
        ax.plot(xs, ys, marker="o", color="#6366F1", linewidth=2.5)
        ax.set_xlabel(visual_data.get("x_label") or "")
        ax.set_ylabel(visual_data.get("y_label") or "")
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return _fig_to_image(fig, width, height)

    # ---------------- EQUATION / LATEX ----------------
    if visual_type in {"EQUATION", "LATEX"} and plt is not None and visual_content:
        try:
            fig = plt.figure(figsize=(width / 110, height / 110))
            fig.text(0.5, 0.5, f"${visual_content}$", fontsize=22, ha="center", va="center")
            return _fig_to_image(fig, width, height)
        except Exception:
            pass  # fall through to text card

    # ---------------- FLOWCHART / PROCESS ----------------
    if visual_type in {"FLOWCHART", "PROCESS"}:
        steps = visual_data.get("flow_steps") or _derive_flow_steps(concept, visual_content)
        img, draw = _blank_panel(width, height)
        draw.text((width // 2, int(height * 0.08)), label, font=font_title, fill=TEXT_DARK, anchor="mm")

        n = max(1, len(steps))
        box_w = int(width * 0.8)
        box_h = int(height * 0.55 / n) - 10
        x0 = int(width * 0.1)
        y = int(height * 0.18)

        for i, step in enumerate(steps):
            draw.rounded_rectangle(
                [x0, y, x0 + box_w, y + box_h],
                radius=10,
                outline=ACCENT,
                width=3,
                fill=(238, 240, 255),
            )
            lines = _wrap(draw, f"{i + 1}. {step}", font_body, box_w - 24)
            ty = y + box_h // 2 - (len(lines) * 14) // 2
            for line in lines:
                draw.text((x0 + box_w // 2, ty), line, font=font_body, fill=TEXT_DARK, anchor="mm")
                ty += 20
            if i < len(steps) - 1:
                arrow_y = y + box_h
                draw.line([x0 + box_w // 2, arrow_y, x0 + box_w // 2, arrow_y + 10], fill=ACCENT, width=3)
            y += box_h + 10

        return img

    # ---------------- TIMELINE ----------------
    if visual_type == "TIMELINE":
        events = visual_data.get("timeline_events") or _derive_timeline_events(concept, visual_content)
        img, draw = _blank_panel(width, height)
        draw.text((width // 2, int(height * 0.08)), label, font=font_title, fill=TEXT_DARK, anchor="mm")

        n = max(1, len(events))
        line_x = int(width * 0.12)
        draw.line([line_x, int(height * 0.18), line_x, int(height * 0.9)], fill=ACCENT, width=3)

        step_y = (int(height * 0.9) - int(height * 0.18)) / n
        y = int(height * 0.2)
        for event in events:
            draw.ellipse([line_x - 8, y - 8, line_x + 8, y + 8], fill=ACCENT)
            lines = _wrap(draw, str(event), font_body, int(width * 0.72))
            ty = y - (len(lines) * 9)
            for line in lines:
                draw.text((line_x + 24, ty), line, font=font_body, fill=TEXT_DARK, anchor="lm")
                ty += 20
            y += step_y

        return img

    # ---------------- TABLE ----------------
    if visual_type == "TABLE":
        rows = visual_data.get("table_rows") or [["Concept", label], ["Key idea", visual_content or "See explanation"]]
        img, draw = _blank_panel(width, height)
        draw.text((width // 2, int(height * 0.08)), label, font=font_title, fill=TEXT_DARK, anchor="mm")

        n = max(1, len(rows))
        row_h = int(height * 0.7 / n)
        y = int(height * 0.16)
        col_x = int(width * 0.06)
        col_w = int(width * 0.88)

        for row in rows:
            draw.rectangle([col_x, y, col_x + col_w, y + row_h - 6], outline=ACCENT, width=2)
            text = " — ".join(str(c) for c in row)
            lines = _wrap(draw, text, font_body, col_w - 20)
            ty = y + row_h // 2 - (len(lines) * 10)
            for line in lines:
                draw.text((col_x + 12, ty), line, font=font_body, fill=TEXT_DARK, anchor="lm")
                ty += 20
            y += row_h

        return img

    # ---------------- CODE ----------------
    if visual_type == "CODE":
        img, draw = _blank_panel(width, height)
        draw.rectangle([0, 0, width, height], fill=(30, 32, 44))
        code_font = _get_font(int(height * 0.042))
        text = visual_content or "# code example"
        y = int(height * 0.08)
        for raw_line in text.splitlines()[:14]:
            draw.text((int(width * 0.06), y), raw_line, font=code_font, fill=(180, 230, 180))
            y += int(height * 0.06)
        return img

    # ---------------- DEFAULT: TEXT / DIAGRAM / MAP / SIMULATION / IMAGE ----------------
    img, draw = _blank_panel(width, height)
    draw.text((width // 2, int(height * 0.1)), label, font=font_title, fill=TEXT_DARK, anchor="mm")
    body_text = visual_content or f"The AI Teacher is explaining {label}."
    lines = _wrap(draw, body_text, font_body, int(width * 0.82))
    y = int(height * 0.24)
    for line in lines[:12]:
        draw.text((width // 2, y), line, font=font_body, fill=TEXT_DARK, anchor="mm")
        y += 26
    return img


# ============================================================
# SCENE COMPOSITION
# ============================================================

def compose_scene_frame(
    width: int,
    height: int,
    avatar_img: Image.Image,
    visual_img: Image.Image,
    topic: str,
    module_title: str,
    caption_text: str = "",
) -> Image.Image:
    """Combine avatar + visual panel + header/caption into one frame."""

    frame = Image.new("RGB", (width, height), BG_LEFT)

    avatar_w = int(width * 0.38)
    visual_w = width - avatar_w

    avatar_resized = avatar_img.resize((avatar_w, height))
    visual_resized = visual_img.resize((visual_w, int(height * 0.86)))

    frame.paste(avatar_resized, (0, 0))
    frame.paste(visual_resized, (avatar_w, int(height * 0.0)))

    draw = ImageDraw.Draw(frame)

    # Top title bar (over the visual half only, avatar has its own name badge)
    title_font = _get_font(int(height * 0.045), bold=True)
    draw.rectangle([avatar_w, int(height * 0.86), width, height], fill=(20, 22, 36))
    draw.text(
        (avatar_w + 16, int(height * 0.93)),
        module_title or topic,
        font=title_font,
        fill=TEXT_LIGHT,
        anchor="lm",
    )

    if caption_text:
        cap_font = _get_font(int(height * 0.032))
        wrapped = textwrap.shorten(caption_text, width=90, placeholder="…")
        draw.text(
            (avatar_w + 16, int(height * 0.965)),
            wrapped,
            font=cap_font,
            fill=ACCENT_SOFT,
            anchor="lm",
        )

    return frame
