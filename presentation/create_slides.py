"""Create the editable 16:9 presentation deck (requires python-pptx)."""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "slides.pptx"

NAVY = RGBColor(8, 24, 37)
PANEL = RGBColor(16, 39, 57)
PANEL_2 = RGBColor(21, 49, 69)
WHITE = RGBColor(237, 244, 250)
MUTED = RGBColor(166, 184, 199)
TEAL = RGBColor(51, 211, 194)
ORANGE = RGBColor(255, 157, 81)
PINK = RGBColor(255, 104, 151)
BLUE = RGBColor(94, 176, 255)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank = prs.slide_layouts[6]


def _background(slide) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = NAVY


def _text(
    slide,
    text: str,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    size: float = 18,
    color=WHITE,
    bold: bool = False,
    font: str = "Aptos",
    align=PP_ALIGN.LEFT,
    valign=MSO_ANCHOR.TOP,
) -> None:
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(0.02)
    frame.margin_right = Inches(0.02)
    frame.margin_top = Inches(0.01)
    frame.margin_bottom = Inches(0.01)
    frame.vertical_anchor = valign
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    paragraph.space_after = Pt(0)
    run = paragraph.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return box


def _card(slide, x: float, y: float, w: float, h: float, fill=PANEL, line=None) -> None:
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(x),
        Inches(y),
        Inches(w),
        Inches(h),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line or fill
    shape.adjustments[0] = 0.08
    return shape


def _line(
    slide, x1: float, y1: float, x2: float, y2: float, color=TEAL, width: float = 1.5
) -> None:
    shape = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT,
        Inches(x1),
        Inches(y1),
        Inches(x2),
        Inches(y2),
    )
    shape.line.color.rgb = color
    shape.line.width = Pt(width)
    return shape


def _header(slide, number: int, title: str, kicker: str | None = None) -> None:
    _background(slide)
    if kicker:
        _text(slide, kicker.upper(), 0.7, 0.34, 6.0, 0.28, size=10, color=TEAL, bold=True)
    _text(slide, title, 0.7, 0.72, 11.9, 0.72, size=30, bold=True)
    _line(slide, 0.7, 1.57, 12.62, 1.57, color=PANEL_2, width=1.2)
    _text(slide, f"C12  /  {number:02d}", 0.7, 7.12, 2.0, 0.2, size=9, color=MUTED)
    _text(
        slide,
        "MEASURE  ·  INFER  ·  TUNE",
        9.4,
        7.12,
        3.2,
        0.2,
        size=9,
        color=MUTED,
        align=PP_ALIGN.RIGHT,
    )


def _bullet(slide, text: str, x: float, y: float, w: float, *, color=WHITE, size=16) -> None:
    _text(slide, "•", x, y, 0.25, 0.36, size=size + 1, color=TEAL, bold=True)
    _text(slide, text, x + 0.28, y, w - 0.28, 0.52, size=size, color=color)


def _image_fit(slide, path: Path, x: float, y: float, w: float, h: float) -> None:
    picture = slide.shapes.add_picture(str(path), Inches(x), Inches(y))
    image_ratio = picture.width / picture.height
    box_ratio = w / h
    if image_ratio > box_ratio:
        picture.width = Inches(w)
        picture.height = int(picture.width / image_ratio)
        picture.left = Inches(x)
        picture.top = Inches(y + (h - picture.height / 914400) / 2)
    else:
        picture.height = Inches(h)
        picture.width = int(picture.height * image_ratio)
        picture.left = Inches(x + (w - picture.width / 914400) / 2)
        picture.top = Inches(y)
    return picture


# 1 — Cover -----------------------------------------------------------------
slide = prs.slides.add_slide(blank)
_background(slide)
_card(slide, 8.72, 0.78, 3.83, 5.93, fill=PANEL)
_text(
    slide,
    "C12 HACKATHON  /  REFERENCE SOLUTION",
    0.82,
    0.72,
    6.6,
    0.35,
    size=11,
    color=TEAL,
    bold=True,
)
_text(slide, "AUTOMATE\nCSD CALIBRATION", 0.78, 1.43, 7.5, 2.05, size=38, color=WHITE, bold=True)
_text(
    slide,
    "Detect the sticks. Measure their contrast.\nRecenter as the barriers move the scene.",
    0.84,
    3.78,
    7.1,
    0.85,
    size=20,
    color=MUTED,
)
_text(
    slide,
    "A physics-guided detector + a drift-aware global optimizer",
    0.84,
    5.35,
    7.2,
    0.4,
    size=14,
    color=TEAL,
    bold=True,
)
for idx, (num, label, detail) in enumerate(
    [
        ("01", "DETECT", "short dark sticks"),
        ("02", "QUANTIFY", "local contrast"),
        ("03", "RECENTER", "plunger plane"),
    ]
):
    yy = 1.28 + idx * 1.62
    _text(slide, num, 9.12, yy, 0.55, 0.44, size=22, color=TEAL, bold=True)
    _text(slide, label, 9.87, yy, 2.1, 0.36, size=15, color=WHITE, bold=True)
    _text(slide, detail, 9.87, yy + 0.44, 2.15, 0.35, size=13, color=MUTED)
    if idx < 2:
        _line(slide, 9.16, yy + 0.68, 12.08, yy + 0.68, color=PANEL_2, width=1.0)
_text(
    slide, "10 MIN  ·  ENGLISH  ·  REPRODUCIBLE SEEDS", 0.82, 7.08, 6.7, 0.22, size=9, color=MUTED
)

# 2 — What the detector sees -----------------------------------------------
slide = prs.slides.add_slide(blank)
_header(slide, 2, "A sparse target inside a noisy map", "01 / Detection")
_card(slide, 0.72, 1.85, 3.12, 4.74)
_text(slide, "WHAT COUNTS AS A STICK?", 1.0, 2.16, 2.55, 0.32, size=12, color=TEAL, bold=True)
_bullet(slide, "Short and dark", 1.0, 2.82, 2.55, size=15)
_bullet(slide, "Near 45° orientation", 1.0, 3.56, 2.55, size=15)
_bullet(slide, "Only a few pixels wide", 1.0, 4.30, 2.55, size=15)
_line(slide, 1.0, 5.12, 3.55, 5.12, color=PANEL_2)
_text(
    slide,
    "Row stripes and long charging-line connectors are nuisance structure, not labels.",
    1.0,
    5.38,
    2.52,
    0.82,
    size=13,
    color=MUTED,
)
_card(slide, 4.08, 1.85, 8.48, 4.74, fill=WHITE)
_image_fit(slide, ROOT / "assets/detection_example.png", 4.17, 2.20, 8.30, 3.98)
_text(
    slide,
    "Raw CSD  →  ground truth  →  detector overlay",
    4.22,
    6.22,
    8.1,
    0.25,
    size=11,
    color=MUTED,
    align=PP_ALIGN.CENTER,
)

# 3 — Detector architecture -------------------------------------------------
slide = prs.slides.add_slide(blank)
_header(slide, 3, "Matched filters grounded in device geometry", "01 / Detection")
steps = [
    ("1", "ROW BASELINE", "Subtract each row's median to suppress shared horizontal noise."),
    (
        "2",
        "TEMPLATE BANK",
        "Blurred, zero-sum dark rectangles across physical length, width, and angle.",
    ),
    ("3", "MASK", "Noise-normalized response → non-maximum suppression → winning footprint."),
]
for idx, (num, title, detail) in enumerate(steps):
    x = 0.78 + idx * 4.15
    _card(slide, x, 2.05, 3.7, 2.48, fill=PANEL)
    _text(slide, num, x + 0.25, 2.28, 0.55, 0.55, size=28, color=TEAL, bold=True)
    _text(slide, title, x + 0.88, 2.36, 2.55, 0.36, size=13, bold=True)
    _text(slide, detail, x + 0.28, 3.13, 3.1, 1.04, size=15, color=MUTED)
    if idx < 2:
        _text(slide, "→", x + 3.76, 2.99, 0.35, 0.42, size=24, color=TEAL, bold=True)
_card(slide, 0.78, 4.93, 11.82, 1.28, fill=PANEL_2)
_text(slide, "WHY THIS MODEL?", 1.05, 5.17, 1.8, 0.28, size=11, color=TEAL, bold=True)
_text(
    slide,
    "Known geometry is a transferable prior; no training-set leakage, metadata input, or image normalization.",
    3.0,
    5.14,
    9.1,
    0.52,
    size=16,
    color=WHITE,
)
_text(
    slide,
    "Segmentation threshold fixed at 4.0 on validation seed 999.",
    1.05,
    5.73,
    10.4,
    0.25,
    size=11,
    color=MUTED,
)

# 4 — Detection metrics -----------------------------------------------------
slide = prs.slides.add_slide(blank)
_header(slide, 4, "Held-out detection: high recall, transparent trade-off", "01 / Validation")
_text(slide, "120", 0.92, 2.02, 2.2, 0.8, size=44, color=TEAL, bold=True)
_text(slide, "held-out images\nseed 2026", 0.98, 2.90, 2.5, 0.7, size=16, color=MUTED)
_card(slide, 3.45, 1.95, 4.05, 3.9, fill=PANEL)
_text(slide, "PIXEL MASK", 3.78, 2.28, 2.4, 0.35, size=13, color=TEAL, bold=True)
_text(slide, "0.746", 3.78, 2.82, 2.7, 0.75, size=42, bold=True)
_text(slide, "macro F1", 3.80, 3.55, 2.4, 0.32, size=15, color=MUTED)
_line(slide, 3.80, 4.08, 7.07, 4.08, color=PANEL_2)
_text(slide, "0.608", 3.80, 4.30, 1.55, 0.55, size=27, color=WHITE, bold=True)
_text(slide, "macro IoU", 5.28, 4.41, 1.7, 0.28, size=13, color=MUTED)
_card(slide, 7.83, 1.95, 4.05, 3.9, fill=PANEL)
_text(slide, "OBJECT-LEVEL", 8.16, 2.28, 2.5, 0.35, size=13, color=TEAL, bold=True)
_text(slide, "0.931", 8.16, 2.82, 2.7, 0.75, size=42, bold=True)
_text(slide, "stick recall", 8.18, 3.55, 2.4, 0.32, size=15, color=MUTED)
_line(slide, 8.18, 4.08, 11.45, 4.08, color=PANEL_2)
_text(slide, "0.718", 8.18, 4.30, 1.55, 0.55, size=27, color=WHITE, bold=True)
_text(slide, "object precision", 9.67, 4.41, 1.9, 0.28, size=13, color=MUTED)
_text(
    slide,
    "Validation: 100 images / seed 999  ·  Test threshold untouched  ·  8-connected overlap defines an object hit",
    0.95,
    6.25,
    11.6,
    0.4,
    size=12,
    color=MUTED,
)

# 5 — Challenge 2 objective -------------------------------------------------
slide = prs.slides.add_slide(blank)
_header(slide, 5, "Optimize the strongest interdot, not image standard deviation", "02 / Objective")
_card(slide, 0.78, 1.95, 5.62, 4.68, fill=PANEL)
_text(slide, "THE FIVE GATES", 1.12, 2.28, 2.1, 0.32, size=12, color=TEAL, bold=True)
_text(slide, "g1  ·  g3  ·  g5", 1.12, 2.82, 4.5, 0.5, size=27, bold=True)
_text(slide, "barriers: contrast + drift", 1.14, 3.34, 4.55, 0.38, size=15, color=MUTED)
_line(slide, 1.12, 4.05, 5.98, 4.05, color=PANEL_2)
_text(slide, "g2  ·  g4", 1.12, 4.42, 4.5, 0.5, size=27, color=WHITE, bold=True)
_text(slide, "plungers: recenter the scan", 1.14, 4.94, 4.55, 0.38, size=15, color=MUTED)
_card(slide, 6.75, 1.95, 5.83, 4.68, fill=PANEL_2)
_text(slide, "MEASUREMENT OBJECTIVE", 7.10, 2.28, 3.7, 0.32, size=12, color=TEAL, bold=True)
_text(slide, "strongest local dip", 7.10, 2.88, 4.9, 0.56, size=27, bold=True)
_text(slide, "matched response ÷ template gain", 7.12, 3.52, 4.95, 0.38, size=16, color=MUTED)
_text(
    slide,
    "Removes most pixel-area bias while avoiding scene-wide dilution.",
    7.12,
    4.35,
    4.7,
    0.72,
    size=17,
    color=WHITE,
)
_text(
    slide,
    "Objective uses measured images only.",
    7.12,
    5.72,
    4.5,
    0.32,
    size=13,
    color=TEAL,
    bold=True,
)

# 6 — Drift correction ------------------------------------------------------
slide = prs.slides.add_slide(blank)
_header(slide, 6, "Drift correction is part of the experiment", "02 / Tracking")
flow = [
    (0.86, "BARRIER\nSTEP", "gates change"),
    (3.95, "IMAGE\nSHIFT", "sticks move"),
    (7.04, "REGISTER\nEDGES", "estimate Δx, Δy"),
    (10.13, "PAN\nWINDOW", "keep sticks visible"),
]
for idx, (x, title, sub) in enumerate(flow):
    _card(slide, x, 2.16, 2.35, 1.72, fill=PANEL if idx != 2 else PANEL_2)
    _text(
        slide,
        title,
        x + 0.18,
        2.46,
        1.98,
        0.74,
        size=18,
        color=TEAL if idx == 2 else WHITE,
        bold=True,
        align=PP_ALIGN.CENTER,
    )
    _text(slide, sub, x + 0.18, 3.32, 1.98, 0.27, size=11, color=MUTED, align=PP_ALIGN.CENTER)
    if idx < 3:
        _text(slide, "→", x + 2.49, 2.75, 0.45, 0.42, size=25, color=TEAL, bold=True)
_card(slide, 0.86, 4.48, 11.62, 1.48, fill=PANEL_2)
_text(slide, "CALIBRATE", 1.18, 4.80, 1.35, 0.28, size=11, color=TEAL, bold=True)
_text(slide, "6 orthogonal probes", 1.18, 5.20, 2.2, 0.3, size=14)
_text(slide, "FIT", 4.18, 4.80, 0.8, 0.28, size=11, color=TEAL, bold=True)
_text(slide, "robust linear → quadratic drift", 4.18, 5.20, 3.3, 0.3, size=14)
_text(slide, "GUARD", 8.20, 4.80, 1.1, 0.28, size=11, color=TEAL, bold=True)
_text(slide, "ambiguous match → repeat + average", 8.20, 5.20, 3.8, 0.3, size=14)
_text(
    slide,
    "Relative registration only: no known drift matrix, reveal(), or simulator attributes.",
    1.1,
    6.30,
    11.2,
    0.3,
    size=12,
    color=MUTED,
)

# 7 — Search plan -----------------------------------------------------------
slide = prs.slides.add_slide(blank)
_header(slide, 7, "Global exploration → local pattern search", "02 / Search")
_card(slide, 0.84, 1.98, 5.55, 4.7, fill=PANEL)
_text(slide, "01  GLOBAL", 1.18, 2.30, 2.1, 0.35, size=13, color=TEAL, bold=True)
_text(slide, "48 scrambled Sobol points", 1.18, 2.88, 4.6, 0.42, size=21, bold=True)
_text(slide, "3-D barrier cube: [−0.5, 0.5]³", 1.18, 3.42, 4.7, 0.34, size=14, color=MUTED)
_line(slide, 1.18, 4.05, 5.98, 4.05, color=PANEL_2)
_text(slide, "Trackable route", 1.18, 4.38, 2.3, 0.35, size=17, bold=True)
_text(
    slide,
    "Nearest-neighbour order; each barrier move ≤ 0.08 V. Intermediate measurements still contribute scores.",
    1.18,
    4.85,
    4.65,
    0.93,
    size=14,
    color=MUTED,
)
_card(slide, 6.82, 1.98, 5.64, 4.7, fill=PANEL_2)
_text(slide, "02  LOCAL", 7.18, 2.30, 2.0, 0.35, size=13, color=TEAL, bold=True)
_text(slide, "Two separated starts", 7.18, 2.88, 4.6, 0.42, size=21, bold=True)
for idx, (val, label) in enumerate(
    [("0.08 V", "explore"), ("0.04 V", "refine"), ("0.02 V", "refine"), ("0.01 V", "confirm")]
):
    x = 7.18 + idx * 1.22
    _card(slide, x, 3.77, 1.02, 0.94, fill=PANEL)
    _text(
        slide,
        val,
        x + 0.04,
        3.96,
        0.94,
        0.25,
        size=13,
        color=WHITE,
        bold=True,
        align=PP_ALIGN.CENTER,
    )
    _text(slide, label, x + 0.04, 4.30, 0.94, 0.22, size=9, color=MUTED, align=PP_ALIGN.CENTER)
    if idx < 3:
        _text(slide, "→", x + 1.04, 4.04, 0.2, 0.28, size=15, color=TEAL, bold=True)
_text(
    slide,
    "Coordinate pattern search · 2 frames for local comparisons · 3-frame final score",
    7.18,
    5.20,
    4.65,
    0.78,
    size=14,
    color=MUTED,
)

# 8 — Results ---------------------------------------------------------------
slide = prs.slides.add_slide(blank)
_header(slide, 8, "Ten fresh devices: strong measured-score recovery", "02 / Results")
_image_fit(slide, ROOT / "assets/optimization_benchmark.png", 0.77, 1.84, 8.85, 4.95)
_card(slide, 9.88, 1.95, 2.62, 4.54, fill=PANEL)
_text(slide, "0.874", 10.17, 2.30, 2.1, 0.7, size=36, color=TEAL, bold=True, align=PP_ALIGN.CENTER)
_text(
    slide, "mean score ratio", 10.11, 3.03, 2.2, 0.32, size=13, color=MUTED, align=PP_ALIGN.CENTER
)
_line(slide, 10.20, 3.63, 12.18, 3.63, color=PANEL_2)
_text(
    slide, "0.857", 10.17, 3.88, 2.1, 0.58, size=29, color=WHITE, bold=True, align=PP_ALIGN.CENTER
)
_text(slide, "median ratio", 10.11, 4.46, 2.2, 0.30, size=13, color=MUTED, align=PP_ALIGN.CENTER)
_text(
    slide,
    "334.5 scans\n7.53M pixels\nmedian budget",
    10.12,
    5.12,
    2.14,
    0.90,
    size=14,
    color=WHITE,
    align=PP_ALIGN.CENTER,
)
_text(
    slide,
    "Oracle measurements are post-run self-checks and excluded from the optimizer budget. Ratios are noisy image-derived proxies.",
    0.92,
    6.68,
    11.5,
    0.28,
    size=10,
    color=MUTED,
)

# 9 — Limits and take-away --------------------------------------------------
slide = prs.slides.add_slide(blank)
_header(slide, 9, "What transfers — and what we would improve next", "03 / Outlook")
for x, title, body, color in [
    (
        0.86,
        "TRANSFER",
        "Physical template prior\nRow-noise suppression\nFixed-seed evaluation",
        TEAL,
    ),
    (
        4.93,
        "LIMITS",
        "Known stick geometry\nAmbiguous low-SNR drift\nNoisy strongest-stick estimate",
        ORANGE,
    ),
    (
        9.00,
        "NEXT",
        "Uncertainty-aware CNR\nMulti-resolution scans\nBayesian / trust-region search",
        BLUE,
    ),
]:
    _card(slide, x, 2.02, 3.48, 2.55, fill=PANEL)
    _text(slide, title, x + 0.28, 2.33, 2.9, 0.35, size=13, color=color, bold=True)
    _text(slide, body, x + 0.28, 2.98, 2.9, 1.26, size=15, color=WHITE)
_card(slide, 0.86, 5.13, 11.62, 1.20, fill=PANEL_2)
_text(slide, "TAKE-AWAY", 1.18, 5.48, 1.35, 0.3, size=12, color=TEAL, bold=True)
_text(
    slide,
    "Build the objective from physics. Track the scene. Keep the oracle outside the algorithm.",
    2.83,
    5.40,
    8.95,
    0.46,
    size=18,
    color=WHITE,
    bold=True,
)

prs.save(OUT)
print(f"wrote {OUT}")
