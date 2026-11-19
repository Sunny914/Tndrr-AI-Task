"""Build the four-page A4 TNDRR assignment report. Measured numbers only."""

from __future__ import annotations

import os
from pathlib import Path

from reportlab.lib.colors import Color, HexColor, white

WHITE = white
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT = Path(__file__).resolve().parent / "TNDRR_Assignment_Report.pdf"
FONT_DIR = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"

NAVY = HexColor("#17324F")
TEAL = HexColor("#087E80")
AMBER = HexColor("#C28A39")
INK = HexColor("#263445")
MUTED = HexColor("#667587")
BG = HexColor("#F4F7F9")
RULE = HexColor("#D5DDE5")
TEAL_SOFT = HexColor("#D7EEED")
AMBER_SOFT = HexColor("#F7EDDA")
NAVY_SOFT = HexColor("#E7EEF3")
PAGE_W, PAGE_H = A4
MARGIN = 42
CONTENT_W = PAGE_W - 2 * MARGIN
TOP_MARGIN = 38
BOTTOM_MARGIN = 36


def _register_fonts() -> None:
    mapping = {
        "SegoeUI": "segoeui.ttf",
        "SegoeUI-Bold": "segoeuib.ttf",
        "SegoeUI-Italic": "segoeuii.ttf",
        "SegoeUI-BoldItalic": "segoeuiz.ttf",
        "SegoeUI-Light": "segoeuil.ttf",
        "SegoeUI-Semilight": "segoeuisl.ttf",
        "Consolas": "consola.ttf",
        "Consolas-Bold": "consolab.ttf",
    }
    for name, fname in mapping.items():
        path = FONT_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"Required font not found: {path}")
        pdfmetrics.registerFont(TTFont(name, str(path)))
    pdfmetrics.registerFontFamily(
        "SegoeUI",
        normal="SegoeUI",
        bold="SegoeUI-Bold",
        italic="SegoeUI-Italic",
        boldItalic="SegoeUI-BoldItalic",
    )


_register_fonts()


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    parent = base["Normal"]
    s: dict[str, ParagraphStyle] = {}
    s["kicker"] = ParagraphStyle(
        "kicker", parent=parent, fontName="SegoeUI", fontSize=8,
        leading=11, textColor=TEAL, spaceAfter=1, tracking=0.6,
    )
    s["title"] = ParagraphStyle(
        "title", parent=parent, fontName="SegoeUI-Bold", fontSize=17.5,
        leading=21, textColor=NAVY, spaceAfter=3,
    )
    s["subtitle"] = ParagraphStyle(
        "subtitle", parent=parent, fontName="SegoeUI", fontSize=9.4,
        leading=12.6, textColor=MUTED,         spaceAfter=6,
    )
    s["h2"] = ParagraphStyle(
        "h2", parent=parent, fontName="SegoeUI-Bold", fontSize=11.5,
        leading=14.5, textColor=NAVY, spaceBefore=0, spaceAfter=0,
    )
    s["body"] = ParagraphStyle(
        "body", parent=parent, fontName="SegoeUI", fontSize=9.5,
        leading=12.8, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=4,
    )
    s["small"] = ParagraphStyle(
        "small", parent=parent, fontName="SegoeUI", fontSize=8.6,
        leading=11.5, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=3,
    )
    s["caption"] = ParagraphStyle(
        "caption", parent=parent, fontName="SegoeUI-Italic", fontSize=8,
        leading=10.6, textColor=MUTED, spaceAfter=2, spaceBefore=1,
    )
    s["cell"] = ParagraphStyle(
        "cell", parent=parent, fontName="SegoeUI", fontSize=8.6,
        leading=11.2, textColor=INK, alignment=TA_LEFT,
    )
    s["cellm"] = ParagraphStyle(
        "cellm", parent=parent, fontName="SegoeUI", fontSize=8.6,
        leading=11.2, textColor=MUTED, alignment=TA_LEFT,
    )
    s["cellc"] = ParagraphStyle(
        "cellc", parent=parent, fontName="SegoeUI", fontSize=8.6,
        leading=11.2, textColor=INK, alignment=TA_CENTER,
    )
    s["th"] = ParagraphStyle(
        "th", parent=parent, fontName="SegoeUI-Bold", fontSize=8.2,
        leading=11, textColor=NAVY, alignment=TA_LEFT,
    )
    s["hero_lbl"] = ParagraphStyle(
        "hero_lbl", parent=parent, fontName="SegoeUI", fontSize=7.4,
        leading=9.4, textColor=MUTED, alignment=TA_CENTER,
    )
    s["eq"] = ParagraphStyle(
        "eq", parent=parent, fontName="Consolas", fontSize=11,
        leading=14, textColor=NAVY, alignment=TA_CENTER, spaceBefore=2, spaceAfter=3,
    )
    s["badge"] = ParagraphStyle(
        "badge", parent=parent, fontName="SegoeUI-Bold", fontSize=7.2,
        leading=9, textColor=AMBER, alignment=TA_LEFT,
    )
    s["card_title"] = ParagraphStyle(
        "card_title", parent=parent, fontName="SegoeUI-Bold", fontSize=9,
        leading=11.5, textColor=NAVY,
    )
    s["card_body"] = ParagraphStyle(
        "card_body", parent=parent, fontName="SegoeUI", fontSize=8.3,
        leading=11, textColor=INK,
    )
    s["take_h"] = ParagraphStyle(
        "take_h", parent=parent, fontName="SegoeUI-Bold", fontSize=8.4,
        leading=11, textColor=NAVY,
    )
    s["take_b"] = ParagraphStyle(
        "take_b", parent=parent, fontName="SegoeUI", fontSize=8.2,
        leading=10.8, textColor=INK,
    )
    return s


S = styles()


def P(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def header_footer(canvas, doc) -> None:
    canvas.saveState()
    w, h = A4
    canvas.setFillColor(NAVY)
    canvas.setFont("SegoeUI", 7.6)
    canvas.drawString(MARGIN, h - 22, "TNDRR AI/ML Engineering Assignment")
    canvas.setFillColor(MUTED)
    canvas.setFont("SegoeUI", 7.6)
    canvas.drawRightString(w - MARGIN, h - 22, "Combined Technical Report")
    canvas.setStrokeColor(TEAL)
    canvas.setLineWidth(1.15)
    canvas.line(MARGIN, h - 27, w - MARGIN, h - 27)
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.6)
    canvas.line(MARGIN, 24, w - MARGIN, 24)
    canvas.setFillColor(MUTED)
    canvas.setFont("SegoeUI", 7.2)
    canvas.drawString(MARGIN, 13, "Measured  ·  Theoretical  ·  Proposed  ·  Provisional")
    canvas.setFillColor(NAVY)
    canvas.setFont("SegoeUI", 7.2)
    canvas.drawRightString(w - MARGIN, 13, f"Page {doc.page} of 4")
    canvas.restoreState()


def heading(text: str, space_before: float = 9) -> Table:
    mark = Table([[""]], colWidths=[3.2], rowHeights=[12])
    mark.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), TEAL),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    t = Table([[mark, P(text, "h2")]], colWidths=[7, CONTENT_W - 7])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (0, 0), 6),
        ("RIGHTPADDING", (1, 0), (1, 0), 0),
        ("TOPPADDING", (0, 0), (-1, -1), space_before),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def pill(text: str, fg: Color, bg: Color) -> Table:
    p = Paragraph(
        text,
        ParagraphStyle(
            "pill", parent=S["badge"], textColor=fg, alignment=TA_CENTER,
            fontName="SegoeUI-Bold", fontSize=7.0, leading=9,
        ),
    )
    t = Table([[p]], colWidths=[None])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROUNDEDCORNERS", (0, 0), (-1, -1), [3, 3, 3, 3]),
    ]))
    return t


def panel_table(data, col_widths, header=True) -> Table:
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    cmds = [
        ("FONTNAME", (0, 0), (-1, -1), "SegoeUI"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 3.4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.4),
        ("BACKGROUND", (0, 0), (-1, -1), BG),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
        ("ALIGN", (1, 0), (-1, -1), "LEFT"),
    ]
    if header:
        cmds += [
            ("FONTNAME", (0, 0), (-1, 0), "SegoeUI-Bold"),
            ("TEXTCOLOR", (0, 0), (-1, 0), NAVY),
            ("BACKGROUND", (0, 0), (-1, 0), NAVY_SOFT),
            ("LINEBELOW", (0, 0), (-1, 0), 1.0, TEAL),
        ]
    t.setStyle(TableStyle(cmds))
    return t


class RoundedCard(Flowable):
    """Filled rounded rectangle behind child flowables."""

    def __init__(self, inner, width, fill=BG, stroke=None, pad=8, radius=5):
        super().__init__()
        self.inner = inner
        self.card_width = width
        self.fill = fill
        self.stroke = stroke
        self.pad = pad
        self.radius = radius
        self._iw = 0
        self._ih = 0

    def wrap(self, aw, ah):
        iw, ih = self.inner.wrap(self.card_width - 2 * self.pad, ah)
        self._iw, self._ih = iw, ih
        self.width = self.card_width
        self.height = ih + 2 * self.pad
        return self.width, self.height

    def draw(self):
        c = self.canv
        c.saveState()
        c.setFillColor(self.fill)
        if self.stroke:
            c.setStrokeColor(self.stroke)
            c.setLineWidth(0.8)
            stroke = 1
        else:
            stroke = 0
        c.roundRect(0, 0, self.width, self.height, self.radius, fill=1, stroke=stroke)
        c.restoreState()
        self.inner.drawOn(c, self.pad, self.pad)


class HeroMetrics(Flowable):
    def __init__(self, width, height=78):
        super().__init__()
        self.width = width
        self.height = height

    def wrap(self, aw, ah):
        return self.width, self.height

    def draw(self):
        c = self.canv
        n = 4
        gap = 8
        w = (self.width - gap * (n - 1)) / n
        items = [
            ("BASE ACCURACY", "28%", "28 / 100  ·  measured", MUTED, NAVY_SOFT, NAVY),
            ("LoRA ACCURACY", "84%", "84 / 100  ·  measured", TEAL, TEAL_SOFT, TEAL),
            ("IMPROVEMENT", "+56 pp", "same 100 held-out articles", TEAL, TEAL_SOFT, TEAL),
            ("TRAINABLE PARAMS", "540,672", "0.1093% of 494,573,440", NAVY, NAVY_SOFT, NAVY),
        ]
        for i, (label, value, note, accent, fill, value_color) in enumerate(items):
            x = i * (w + gap)
            c.setFillColor(fill)
            c.roundRect(x, 0, w, self.height, 5, fill=1, stroke=0)
            c.setFillColor(accent)
            c.rect(x, self.height - 3.2, w, 3.2, fill=1, stroke=0)
            c.setFillColor(MUTED)
            c.setFont("SegoeUI", 6.8)
            c.drawCentredString(x + w / 2, self.height - 16, label)
            c.setFillColor(value_color)
            c.setFont("SegoeUI-Bold", 22)
            c.drawCentredString(x + w / 2, 28, value)
            c.setFillColor(MUTED)
            c.setFont("SegoeUI", 6.6)
            c.drawCentredString(x + w / 2, 10, note)


class HBarChart(Flowable):
    def __init__(self, width, rows, max_value, height=122, unit="GiB", bar_color=TEAL):
        super().__init__()
        self.width = width
        self.rows = rows  # list of (label, value, color)
        self.max_value = max_value
        self.height = height
        self.unit = unit
        self.bar_color = bar_color

    def wrap(self, aw, ah):
        return self.width, self.height

    def draw(self):
        c = self.canv
        label_w = 128
        value_w = 58
        left = label_w
        right = self.width - value_w
        plot_w = right - left
        n = len(self.rows)
        row_h = self.height / n
        for i, (label, value, color) in enumerate(self.rows):
            y = self.height - (i + 1) * row_h
            mid = y + row_h / 2
            c.setFillColor(INK)
            c.setFont("SegoeUI", 8.0)
            c.drawRightString(left - 8, mid - 3, label)
            bw = plot_w * (value / self.max_value)
            c.setFillColor(BG)
            c.roundRect(left, mid - 8, plot_w, 16, 3, fill=1, stroke=0)
            c.setFillColor(color)
            c.roundRect(left, mid - 8, max(bw, 2), 16, 3, fill=1, stroke=0)
            c.setFillColor(NAVY)
            c.setFont("SegoeUI-Bold", 8.2)
            c.drawString(right + 8, mid - 3, f"{value:.2f} {self.unit}")


class GroupedBars(Flowable):
    """Grouped bars for percentage metrics plus a separate MRR panel."""

    def __init__(self, width, height=156):
        super().__init__()
        self.width = width
        self.height = height

    def wrap(self, aw, ah):
        return self.width, self.height

    def draw(self):
        c = self.canv
        mrr_w = 118
        plot_w = self.width - mrr_w - 12
        pad_l, pad_b, pad_t = 28, 28, 18
        axis_y = pad_b
        axis_top = self.height - pad_t
        plot_h = axis_top - axis_y
        cats = [
            ("Hit@1", 53.3, 60.0),
            ("Hit@5", 93.3, 93.3),
            ("Quote Hit@5", 60.0, 80.0),
        ]
        n = len(cats)
        slot = (plot_w - pad_l) / n
        bar_w = 16
        gap = 5
        c.setStrokeColor(RULE)
        c.setLineWidth(0.6)
        c.line(pad_l, axis_y, plot_w, axis_y)
        for tick, lbl in [(0, "0"), (50, "50%"), (100, "100%")]:
            y = axis_y + plot_h * (tick / 100)
            c.setStrokeColor(RULE)
            c.setDash(1, 2)
            c.line(pad_l, y, plot_w, y)
            c.setDash()
            c.setFillColor(MUTED)
            c.setFont("SegoeUI", 6.6)
            c.drawRightString(pad_l - 4, y - 2, lbl)
        for i, (name, a, b) in enumerate(cats):
            cx = pad_l + slot * i + slot / 2
            xa = cx - gap / 2 - bar_w
            xb = cx + gap / 2
            ha = plot_h * (a / 100)
            hb = plot_h * (b / 100)
            c.setFillColor(HexColor("#8AA3B5"))
            c.roundRect(xa, axis_y, bar_w, ha, 2, fill=1, stroke=0)
            c.setFillColor(TEAL)
            c.roundRect(xb, axis_y, bar_w, hb, 2, fill=1, stroke=0)
            def _val(xbar, hbar, txt, fill):
                inside = hbar >= 26
                c.setFillColor(WHITE if inside else INK)
                c.setFont("SegoeUI-Bold" if inside else "SegoeUI", 6.4)
                yy = axis_y + hbar - 11 if inside else axis_y + hbar + 3
                c.drawCentredString(xbar + bar_w / 2, yy, txt)

            _val(xa, ha, f"{a:.1f}%", HexColor("#8AA3B5"))
            _val(xb, hb, f"{b:.1f}%", TEAL)
            c.setFillColor(NAVY)
            c.setFont("SegoeUI-Bold", 7.6)
            c.drawCentredString(cx, 8, name)
        # legend
        c.setFillColor(HexColor("#8AA3B5"))
        c.roundRect(pad_l, self.height - 12, 8, 8, 1, fill=1, stroke=0)
        c.setFillColor(MUTED)
        c.setFont("SegoeUI", 7.2)
        c.drawString(pad_l + 12, self.height - 11, "Config A")
        c.setFillColor(TEAL)
        c.roundRect(pad_l + 72, self.height - 12, 8, 8, 1, fill=1, stroke=0)
        c.setFillColor(MUTED)
        c.drawString(pad_l + 84, self.height - 11, "Config B")
        # MRR panel
        x0 = plot_w + 12
        c.setFillColor(BG)
        c.roundRect(x0, 0, mrr_w, self.height, 5, fill=1, stroke=0)
        c.setFillColor(NAVY)
        c.setFont("SegoeUI-Bold", 8)
        c.drawCentredString(x0 + mrr_w / 2, self.height - 16, "MRR")
        c.setFillColor(MUTED)
        c.setFont("SegoeUI", 6.6)
        c.drawCentredString(x0 + mrr_w / 2, self.height - 28, "unitless  ·  0–1 scale")
        mrr = [("A", 0.706, HexColor("#8AA3B5")), ("B", 0.739, TEAL)]
        bar_max = 70
        bx = x0 + 28
        for i, (lab, val, color) in enumerate(mrr):
            y = self.height - 58 - i * 42
            c.setFillColor(INK)
            c.setFont("SegoeUI-Bold", 8.4)
            c.drawString(bx, y + 18, f"{lab}  {val:.3f}")
            c.setFillColor(color)
            c.roundRect(bx, y, bar_max * val, 14, 2, fill=1, stroke=0)
        c.setFillColor(MUTED)
        c.setFont("SegoeUI", 6.2)
        c.drawString(bx, 10, "not a percentage")


class PipelineDiagram(Flowable):
    def __init__(self, width, height=188):
        super().__init__()
        self.width = width
        self.height = height

    def wrap(self, aw, ah):
        return self.width, self.height

    def _box(self, c, x, y, w, h, text, fill, fg=NAVY, dashed=False):
        c.saveState()
        c.setFillColor(fill)
        if dashed:
            c.setStrokeColor(AMBER)
            c.setDash(2, 2)
            c.setLineWidth(0.9)
            c.roundRect(x, y, w, h, 4, fill=1, stroke=1)
        else:
            c.setStrokeColor(RULE)
            c.setLineWidth(0.5)
            c.roundRect(x, y, w, h, 4, fill=1, stroke=1)
        c.setFillColor(fg)
        lines = text.split("\n")
        c.setFont("SegoeUI-Bold" if len(lines) == 1 else "SegoeUI", 7.2 if len(lines) > 1 else 7.6)
        total = len(lines) * 9
        ty = y + h / 2 + total / 2 - 8
        for line in lines:
            c.drawCentredString(x + w / 2, ty, line)
            ty -= 9
        c.restoreState()

    def _harrow(self, c, x1, x2, y):
        c.setStrokeColor(TEAL)
        c.setFillColor(TEAL)
        c.setLineWidth(1.05)
        c.line(x1, y, x2 - 5, y)
        p = c.beginPath()
        p.moveTo(x2, y)
        p.lineTo(x2 - 6, y + 3)
        p.lineTo(x2 - 6, y - 3)
        p.close()
        c.drawPath(p, fill=1, stroke=0)

    def _varrow(self, c, x, y1, y2):
        c.setStrokeColor(TEAL)
        c.setFillColor(TEAL)
        c.setLineWidth(1.05)
        c.line(x, y1, x, y2 + 5)
        p = c.beginPath()
        p.moveTo(x, y2)
        p.lineTo(x - 3, y2 + 6)
        p.lineTo(x + 3, y2 + 6)
        p.close()
        c.drawPath(p, fill=1, stroke=0)

    def draw(self):
        c = self.canv
        W, H = self.width, self.height
        c.setFillColor(BG)
        c.roundRect(0, 0, W, H, 6, fill=1, stroke=0)
        c.setFillColor(MUTED)
        c.setFont("SegoeUI", 7)
        c.drawString(14, H - 14, "Retrieval path (evaluated)     ·     dashed box is generation-only")

        bh = 28
        y_top = H - 48
        b1, b2, b3 = 102, 122, 136
        gap = 18
        x0 = 14
        x1 = x0 + b1 + gap
        x2 = x1 + b2 + gap
        self._box(c, x0, y_top, b1, bh, "NIST AI RMF PDF", WHITE, NAVY)
        self._harrow(c, x0 + b1, x1, y_top + bh / 2)
        self._box(c, x1, y_top, b2, bh, "PyMuPDF extraction", WHITE, NAVY)
        self._harrow(c, x1 + b2, x2, y_top + bh / 2)
        self._box(c, x2, y_top, b3, bh, "Page-bounded chunking", WHITE, NAVY)

        cfg_w, cfg_h = 172, 32
        y_mid = 90
        xa = 42
        xb = W - 42 - cfg_w
        hub_x = x2 + b3 / 2
        fork_y = y_top - 6
        c.setStrokeColor(TEAL)
        c.setFillColor(TEAL)
        c.setLineWidth(1.05)
        c.line(hub_x, y_top, hub_x, fork_y)
        c.line(xa + cfg_w / 2, fork_y, xb + cfg_w / 2, fork_y)
        self._varrow(c, xa + cfg_w / 2, fork_y, y_mid + cfg_h)
        self._varrow(c, xb + cfg_w / 2, fork_y, y_mid + cfg_h)
        self._box(c, xa, y_mid, cfg_w, cfg_h, "Config A   120 tok / 20 overlap\n213 chunks", WHITE, NAVY)
        self._box(c, xb, y_mid, cfg_w, cfg_h, "Config B   220 tok / 40 overlap\n126 chunks", WHITE, NAVY)

        y_bot, bot_h, bot_w = 16, 30, 112
        xs = [14, 146, 278]
        labels = ["MiniLM embeddings", "NumPy cosine search", "Top-5 evidence"]
        for x, lab in zip(xs, labels):
            self._box(c, x, y_bot, bot_w, bot_h, lab, WHITE, NAVY)
        self._harrow(c, xs[0] + bot_w, xs[1], y_bot + bot_h / 2)
        self._harrow(c, xs[1] + bot_w, xs[2], y_bot + bot_h / 2)

        join_y = (y_mid + y_bot + bot_h) / 2 - 2
        mini_cx = xs[0] + bot_w / 2
        ca_cx = xa + cfg_w / 2
        cb_cx = xb + cfg_w / 2
        c.setStrokeColor(TEAL)
        c.setLineWidth(1.05)
        c.line(ca_cx, y_mid, ca_cx, join_y)
        c.line(cb_cx, y_mid, cb_cx, join_y)
        c.line(ca_cx, join_y, cb_cx, join_y)
        c.line(mini_cx, join_y, ca_cx, join_y)
        self._varrow(c, mini_cx, join_y, y_bot + bot_h)

        gx, gw = 404, 94
        self._box(
            c, gx, y_bot, gw, bot_h,
            "Ollama Qwen\n(not in Hit@k)",
            AMBER_SOFT, AMBER, dashed=True,
        )
        c.setStrokeColor(AMBER)
        c.setDash(1.5, 1.5)
        c.setLineWidth(0.85)
        c.line(xs[2] + bot_w + 2, y_bot + bot_h / 2, gx, y_bot + bot_h / 2)
        c.setDash()


class ProcessDiagram(Flowable):
    def __init__(self, width, height=58):
        super().__init__()
        self.width = width
        self.height = height

    def wrap(self, aw, ah):
        return self.width, self.height

    def draw(self):
        c = self.canv
        c.setFillColor(AMBER_SOFT)
        c.roundRect(0, 0, self.width, self.height, 5, fill=1, stroke=0)
        labels = [
            "Dense retrieval + BM25",
            "Hybrid ranking / RRF",
            "Cross-encoder rerank",
        ]
        n = len(labels)
        gap = 18
        bw = (self.width - 20 - gap * (n - 1) - 8) / n
        y = 12
        h = 28
        for i, lab in enumerate(labels):
            x = 10 + i * (bw + gap)
            c.setFillColor(WHITE)
            c.setStrokeColor(AMBER)
            c.setLineWidth(0.8)
            c.roundRect(x, y, bw, h, 4, fill=1, stroke=1)
            c.setFillColor(NAVY)
            c.setFont("SegoeUI-Bold", 7.6)
            c.drawCentredString(x + bw / 2, y + 10, lab)
            if i < n - 1:
                c.setStrokeColor(AMBER)
                c.setFillColor(AMBER)
                c.setLineWidth(1.1)
                x1, x2, yy = x + bw, x + bw + gap, y + h / 2
                c.line(x1, yy, x2 - 5, yy)
                p = c.beginPath()
                p.moveTo(x2, yy)
                p.lineTo(x2 - 6, yy + 3)
                p.lineTo(x2 - 6, yy - 3)
                p.close()
                c.drawPath(p, fill=1, stroke=0)
        c.setFillColor(AMBER)
        c.setFont("SegoeUI-Bold", 7)
        c.drawString(10, self.height - 12, "PROPOSED — NOT IMPLEMENTED")


class DualHBars(Flowable):
    def __init__(self, width, height=108):
        super().__init__()
        self.width = width
        self.height = height

    def wrap(self, aw, ah):
        return self.width, self.height

    def _chart(self, c, x, y, w, h, title, rows, max_v, unit):
        c.setFillColor(BG)
        c.roundRect(x, y, w, h, 5, fill=1, stroke=0)
        c.setFillColor(NAVY)
        c.setFont("SegoeUI-Bold", 8)
        c.drawString(x + 10, y + h - 14, title)
        label_w = 58
        plot_x = x + 10 + label_w
        plot_w = w - label_w - 78
        for i, (lab, val, color) in enumerate(rows):
            yy = y + h - 36 - i * 28
            c.setFillColor(INK)
            c.setFont("SegoeUI", 7.6)
            c.drawRightString(plot_x - 8, yy + 3, lab)
            c.setFillColor(WHITE)
            c.roundRect(plot_x, yy, plot_w, 14, 3, fill=1, stroke=0)
            c.setFillColor(color)
            c.roundRect(plot_x, yy, plot_w * (val / max_v), 14, 3, fill=1, stroke=0)
            c.setFillColor(NAVY)
            c.setFont("SegoeUI-Bold", 7.8)
            c.drawString(plot_x + plot_w + 6, yy + 3, f"{val:.2f} {unit}")

    def draw(self):
        gap = 10
        w = (self.width - gap) / 2
        self._chart(
            self.canv, 0, 0, w, self.height,
            "Decode throughput  (measured)",
            [("FP16", 84.88, TEAL), ("Q4_K_M", 69.00, HexColor("#8AA3B5"))],
            90.0, "tok/s",
        )
        self._chart(
            self.canv, w + gap, 0, w, self.height,
            "Ollama GPU allocation  (size_vram)",
            [("FP16", 1003.94, HexColor("#8AA3B5")), ("Q4_K_M", 435.22, TEAL)],
            1100.0, "MiB",
        )


class LatencyRow(Flowable):
    def __init__(self, width, height=36):
        super().__init__()
        self.width = width
        self.height = height

    def wrap(self, aw, ah):
        return self.width, self.height

    def draw(self):
        c = self.canv
        c.setFillColor(BG)
        c.roundRect(0, 0, self.width, self.height, 5, fill=1, stroke=0)
        items = [
            ("Median wall latency", None),
            ("FP16", "2.528 s"),
            ("Q4_K_M", "2.574 s"),
            ("Difference", "~46 ms"),
        ]
        slot = self.width / 4
        for i, (lab, val) in enumerate(items):
            x = i * slot
            c.setFillColor(MUTED)
            c.setFont("SegoeUI", 6.8)
            c.drawCentredString(x + slot / 2, 22, lab)
            if val:
                c.setFillColor(NAVY)
                c.setFont("SegoeUI-Bold", 10)
                c.drawCentredString(x + slot / 2, 8, val)
            else:
                c.setFillColor(NAVY)
                c.setFont("SegoeUI-Bold", 8)
                c.drawCentredString(x + slot / 2, 8, "60 sequential runs")


class ArchDiagram(Flowable):
    def __init__(self, width, height=118):
        super().__init__()
        self.width = width
        self.height = height

    def wrap(self, aw, ah):
        return self.width, self.height

    def draw(self):
        c = self.canv
        c.setFillColor(AMBER_SOFT)
        c.roundRect(0, 0, self.width, self.height, 6, fill=1, stroke=0)
        c.setFillColor(AMBER)
        c.setFont("SegoeUI-Bold", 7.2)
        c.drawString(12, self.height - 14, "PROPOSED — NOT IMPLEMENTED OR LOAD-TESTED")
        c.setFillColor(MUTED)
        c.setFont("SegoeUI", 6.8)
        c.drawRightString(self.width - 12, self.height - 14, "Target: one 24 GB GPU  ·  not the tested GTX 1650")

        labels = [
            "20 concurrent\nrequests",
            "API gateway /\nvalidation",
            "Queue &\nscheduler",
            "Continuous\nbatching",
            "Inference\nbackend",
            "Responses",
        ]
        n = len(labels)
        gap = 8
        bw = (self.width - 24 - gap * (n - 1)) / n
        y = 38
        h = 36
        for i, lab in enumerate(labels):
            x = 12 + i * (bw + gap)
            c.setFillColor(WHITE)
            c.setStrokeColor(AMBER)
            c.setLineWidth(0.7)
            c.roundRect(x, y, bw, h, 4, fill=1, stroke=1)
            c.setFillColor(NAVY)
            c.setFont("SegoeUI", 6.6)
            lines = lab.split("\n")
            ty = y + h / 2 + (len(lines) * 8) / 2 - 8
            for line in lines:
                c.drawCentredString(x + bw / 2, ty, line)
                ty -= 8
            if i < n - 1:
                c.setStrokeColor(AMBER)
                c.setFillColor(AMBER)
                c.setLineWidth(1.0)
                x1 = x + bw
                x2 = x + bw + gap
                yy = y + h / 2
                c.line(x1, yy, x2 - 4, yy)
                p = c.beginPath()
                p.moveTo(x2, yy)
                p.lineTo(x2 - 5, yy + 2.6)
                p.lineTo(x2 - 5, yy - 2.6)
                p.close()
                c.drawPath(p, fill=1, stroke=0)

        # observability under scheduler/batching
        ox, ow = 12 + 2 * (bw + gap), bw * 2 + gap
        c.setFillColor(WHITE)
        c.setStrokeColor(TEAL)
        c.setDash(1.5, 1.5)
        c.roundRect(ox, 8, ow, 22, 3, fill=1, stroke=1)
        c.setDash()
        c.setFillColor(TEAL)
        c.setFont("SegoeUI-Bold", 7)
        c.drawCentredString(ox + ow / 2, 15, "Observability  ·  queue depth, p95 latency")
        c.setStrokeColor(TEAL)
        c.setLineWidth(0.8)
        c.line(ox + ow / 2, 30, ox + ow / 2, y)


def failure_card(num, title, meta, body, width) -> RoundedCard:
    inner = Table(
        [
            [P(f"{num}   {title}", "card_title")],
            [P(meta, "caption")],
            [P(body, "card_body")],
        ],
        colWidths=[width - 16],
    )
    inner.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 1),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 3),
        ("BOTTOMPADDING", (0, 2), (-1, 2), 0),
    ]))
    return RoundedCard(inner, width, fill=BG, pad=8, radius=5)


def takeaway_card(kicker, body, width) -> RoundedCard:
    inner = Table(
        [[P(kicker, "take_h")], [P(body, "take_b")]],
        colWidths=[width - 16],
    )
    inner.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (0, 0), 3),
        ("BOTTOMPADDING", (0, 1), (0, 1), 0),
    ]))
    return RoundedCard(inner, width, fill=BG, pad=8, radius=5)


def page1():
    flow = []
    flow.append(P("TNDRR AI/ML ENGINEERING ASSIGNMENT", "kicker"))
    flow.append(P("Task 1 — LoRA Fine-Tuning", "title"))
    flow.append(P(
        "Rank-8 LoRA on Qwen2.5-0.5B-Instruct for four-class AG News labels "
        "(WORLD, SPORTS, BUSINESS, SCI_TECH). Completed experiment with saved artifacts; "
        "not a deployed classifier.",
        "subtitle",
    ))
    flow.append(HeroMetrics(CONTENT_W, 70))
    flow.append(Spacer(1, 4))
    flow.append(P(
        "Macro-F1 rose from <b>0.2058</b> to <b>0.8383</b> on the same 100 articles "
        "(invalid outputs 0 → 0). Training runtime was <b>119.07 s</b> on an NVIDIA Tesla T4. "
        "Baseline predicted WORLD on 88/100 articles (BUSINESS recall 0/25). After LoRA, "
        "SPORTS was 25/25 and BUSINESS 21/25; WORLD recall fell from 84% to 68% (21/25 → 17/25). "
        "Token-level eval_loss (0.2016 → 0.1726) is completion loss, not classification accuracy.",
        "small",
    ))

    flow.append(heading("Experimental setup", 5))
    left = panel_table(
        [
            [P("Item", "th"), P("Value", "th")],
            [P("Model", "cellm"), P("Qwen/Qwen2.5-0.5B-Instruct", "cell")],
            [P("Dataset", "cellm"), P("fancyzhx/ag_news, balanced subsample", "cell")],
            [P("Splits", "cellm"), P("800 / 100 / 100  (25 per class on test)", "cell")],
            [P("Hardware", "cellm"), P("NVIDIA Tesla T4, FP16", "cell")],
        ],
        [52, CONTENT_W / 2 - 56],
        header=True,
    )
    right = panel_table(
        [
            [P("Item", "th"), P("Value", "th")],
            [P("Epochs / LR / batch", "cellm"), P("2  /  2e-4  /  16 effective", "cell")],
            [P("Seed", "cellm"), P("42, stratified sampling", "cell")],
            [P("LoRA", "cellm"), P("r=8, α=16, dropout 0.05, bias none", "cell")],
            [P("Targets / task", "cellm"), P("q_proj, v_proj  ·  CAUSAL_LM", "cell")],
        ],
        [70, CONTENT_W / 2 - 74],
        header=True,
    )
    cols = Table([[left, right]], colWidths=[CONTENT_W / 2 + 4, CONTENT_W / 2 - 4])
    cols.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (0, 0), 8),
        ("RIGHTPADDING", (1, 0), (1, 0), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    flow.append(cols)

    flow.append(heading("Method", 6))
    flow.append(P(
        "The base model is a causal instruction model, not a classification head. Articles "
        "use a fixed chat template; the supervised target is the category string only "
        "(completion-only loss). The unmodified model is the baseline. PEFT injects LoRA "
        "matrices into attention query and value projections; TRL SFTTrainer updates only "
        "those matrices. Evaluation uses greedy decoding (max_new_tokens=12) and exact "
        "string match on the same 100 held-out articles.",
        "body",
    ))
    flow.append(P("W′  =  W  +  (α / r) BA", "eq"))
    flow.append(P(
        "Pretrained weights <i>W</i> stay frozen. The residual is a low-rank product of "
        "learned matrices <i>B</i> and <i>A</i>, scaled by α/r with <b>r = 8</b> and "
        "<b>α = 16</b>. Adapters attach only to <b>q_proj</b> and <b>v_proj</b>. PEFT "
        "reported <b>540,672</b> trainable parameters of 494,573,440 total (0.1093%). "
        "Measured peak PyTorch allocator use on this 0.5B run was 1.95 GiB (notebook "
        "reading; not stored in the JSON metrics).",
        "small",
    ))

    flow.append(heading("Three documented failures  (16 LoRA errors total)", 5))
    gap = 6
    cw = (CONTENT_W - 2 * gap) / 3
    cards = Table(
        [[
            failure_card(
                "1", "Bird flu  ·  index 12",
                "Gold WORLD  ·  baseline WORLD  ·  LoRA SCI_TECH",
                "Malaysia outbreak. A correct WORLD call was overwritten, consistent "
                "with outbreak/science vocabulary overlap — an interpretation, not a "
                "causal claim.",
                cw,
            ),
            failure_card(
                "2", "Plasma-TV retail  ·  index 25",
                "Gold BUSINESS  ·  baseline WORLD  ·  LoRA SCI_TECH",
                "Retailers promoting plasma TVs. Both models are wrong; LoRA moved "
                "the error toward device terms rather than retail.",
                cw,
            ),
            failure_card(
                "3", "Southern Africa study  ·  index 29",
                "Gold SCI_TECH  ·  both models WORLD",
                "Food/water study. Geographic-crisis cues compete with “study” framing. "
                "Files do not establish which cues the model used.",
                cw,
            ),
        ]],
        colWidths=[cw + gap, cw + gap, cw],
    )
    cards.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (0, 0), gap),
        ("RIGHTPADDING", (1, 0), (1, 0), gap),
        ("RIGHTPADDING", (2, 0), (2, 0), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    flow.append(cards)

    flow.append(heading("Theoretical 7B memory comparison  —  not a 7B training run", 5))
    flow.append(P(
        "Parameter arithmetic for a hypothetical 7B model (memory_estimates.json). "
        "Excludes activations, KV cache, CUDA scratch, and fragmentation. The 4,194,304 "
        "LoRA count in that file is an illustrative 32-layer hidden-4096 sketch, "
        "<b>not</b> the 540,672 parameters trained on Qwen2.5-0.5B.",
        "caption",
    ))
    flow.append(HBarChart(
        CONTENT_W,
        [
            ("Full FT (7B)", 78.23, HexColor("#8AA3B5")),
            ("FT + FP32 master", 104.31, NAVY),
            ("FP16 LoRA (7B sketch)", 13.09, TEAL),
            ("4-bit QLoRA (7B sketch)", 3.31, TEAL),
        ],
        max_value=104.31,
        height=96,
        unit="GiB",
    ))
    flow.append(P(
        "THEORETICAL estimates of parameter-related memory, not measured total VRAM. "
        "Measured peak allocator use for the actual 0.5B LoRA run remains 1.95 GiB on a T4.",
        "caption",
    ))
    return flow


def page2():
    flow = []
    flow.append(P("TNDRR AI/ML ENGINEERING ASSIGNMENT", "kicker"))
    flow.append(P("Task 2 — Framework-Free RAG", "title"))
    flow.append(P(
        "Question answering over NIST AI RMF 1.0 without LangChain, LlamaIndex, Chroma, "
        "or FAISS. Embeddings: sentence-transformers/all-MiniLM-L6-v2. Generation: local "
        "Ollama model qwen-q4 — kept out of retrieval scoring.",
        "subtitle",
    ))
    flow.append(heading("Architecture", 2))
    flow.append(P(
        "PyMuPDF extracts every physical PDF page (1-based). Chunk windows stay inside a "
        "single page so citations remain reliable, and content plus special tokens stays "
        "under MiniLM’s 256-token limit. Chunks are L2-normalised NumPy arrays. Cosine "
        "similarity is <font face='Consolas' size='8'>scores = embeddings @ query</font>, "
        "then argsort descending. Top-5 pages come from the retriever, not from generated "
        "text. Source: data/nist_ai_rmf.pdf — 48 physical pages, 106,478 extracted characters.",
        "small",
    ))
    flow.append(Spacer(1, 3))
    flow.append(PipelineDiagram(CONTENT_W, 186))
    flow.append(P(
        "Identical embedding, normalisation, cosine metric, and top-5 cutoff for both configs. "
        "A grounded prompt (temperature 0) is used only when Ollama is reachable; fabricated "
        "LLM citations are not treated as verified sources.",
        "caption",
    ))

    flow.append(heading("Evaluation methodology", 6))
    flow.append(P(
        "Fifteen questions were grounded in extracted text with evidence quotes that "
        "programmatically occur on the listed pages, then marked <i>human_verified=true</i>. "
        "Citations use physical PDF pages, not printed footer numbers (printed “Page 1” is "
        "physical page 6). Gold labels were not altered after evaluation.",
        "body",
    ))
    metrics = panel_table(
        [
            [P("Metric", "th"), P("Definition", "th")],
            [P("Hit@1", "cell"), P("Rank-1 chunk is from a gold physical page.", "cell")],
            [P("Hit@5", "cell"), P("Any top-5 chunk is from a gold physical page.", "cell")],
            [P("MRR", "cell"), P("Mean of 1/rank of the first gold-page chunk in the top 5, else 0. Unitless.", "cell")],
            [P("Quote Hit@5", "cell"), P("Verified evidence quote occurs inside at least one top-5 chunk (stricter).", "cell")],
        ],
        [78, CONTENT_W - 78],
        header=True,
    )
    flow.append(metrics)

    flow.append(heading("Measured comparison  ·  15 verified questions", 8))
    flow.append(GroupedBars(CONTENT_W, 150))
    flow.append(P(
        "Sources: Task-2-RAG-From-Scratch/results/chunk_comparison.csv and retrieval_results.csv. "
        "Page Hit@5 tied at 0.933 (14/15). Config B was higher on Hit@1 (0.533 → 0.600), "
        "MRR (0.706 → 0.739), and quote Hit@5 (0.600 → 0.800) on this set — a measured "
        "result for these 15 questions, not a general law that larger chunks always win. "
        "Page-level Hit@5 overestimates evidence retrieval: config A quote Hit@5 is 0.600 "
        "against page Hit@5 0.933.",
        "caption",
    ))
    return flow


def page3():
    flow = []
    flow.append(P("TNDRR AI/ML ENGINEERING ASSIGNMENT", "kicker"))
    flow.append(P("Retrieval failures and Task 3 benchmark", "title"))
    flow.append(P(
        "Task 2 misses are from the verified 15-question set. Task 3 numbers are from "
        "results/summary.json — 60 sequential Ollama runs on a 4 GB GTX 1650, not a "
        "concurrent-load experiment.",
        "subtitle",
    ))

    flow.append(heading("Two real retrieval failures", 2))
    gap = 8
    cw = (CONTENT_W - gap) / 2
    q14 = failure_card(
        "Q14", "Hit@5 miss  ·  both configs",
        "Gold physical PDF page 43  ·  MEASURED miss",
        "Question: Appendix B privacy-related AI-specific risk versus traditional software. "
        "Evidence: “Privacy risk due to enhanced data aggregation capability for AI systems.” "
        "Config A top-5 pages: 22, 4, 44, 35, 17. Config B: 44, 13, 4, 22, 22. Dense retrieval "
        "matched thematically related privacy passages (Privacy-Enhanced chapter, MEASURE 2.10, "
        "TOC, next Appendix B page) but missed the short gold bullet in a long list on page 43. "
        "Larger chunks ranked adjacent page 44 first and still missed 43.",
        cw,
    )
    q03 = failure_card(
        "Q03", "Hit@1 and quote miss  ·  config A",
        "Gold physical PDF page 9  ·  NOT a Hit@5 page miss",
        "Question: In the AI RMF, what does risk refer to? Config A pages: 47, 9, 44, 15, 9. "
        "Rank 1 was Appendix D (“manage AI risks”). Page 9 entered the top 5, so page Hit@5 "
        "succeeded, but the 120-token window split the definition across two chunks (quote Hit@5 "
        "= 0). Config B retrieved page 9 at rank 1 with the intact quote. Only one unique "
        "Hit@5-miss question existed; Q03 is the second real failure and is labelled accurately.",
        cw,
    )
    pair = Table([[q14, q03]], colWidths=[cw + gap, cw])
    pair.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (0, 0), gap),
        ("RIGHTPADDING", (1, 0), (1, 0), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    flow.append(pair)
    flow.append(P(
        "Page matching scores a chunk as relevant if it shares a gold PDF page, even when the "
        "supporting sentence is missing or split. Quote Hit@5 requires the verified span. "
        "The gap (A: 0.933 vs 0.600; B: 0.933 vs 0.800) is that proxy error, not a scoring bug.",
        "caption",
    ))

    flow.append(heading("Proposed retrieval improvements", 6))
    flow.append(ProcessDiagram(CONTENT_W, 56))
    flow.append(P(
        "BM25 would promote distinctive phrases such as “data aggregation” (Q14) and "
        "“composite measure” (Q03). Hybrid search keeps BM25 and dense candidates; RRF "
        "merges lists without calibrating cosine against BM25. A cross-encoder can reject "
        "thematically related but wrong pages, at higher latency. None of these components "
        "are in the Task 2 codebase.",
        "small",
    ))

    # visual separator
    sep = Table([[""]], colWidths=[CONTENT_W], rowHeights=[1.1])
    sep.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    flow.append(Spacer(1, 8))
    flow.append(sep)

    flow.append(heading("Task 3 — local FP16 vs Q4_K_M  (measured, sequential)", 8))
    flow.append(P(
        "Same pretrained family, two stored precisions: Qwen2.5-0.5B-Instruct GGUF "
        "<b>FP16</b> (Ollama <i>qwen-fp16</i>) versus <b>Q4_K_M</b> (<i>qwen-q4</i>). "
        "Quantized files were downloaded; this project did not train a quantizer. "
        "Controller: 10 prompts × 3 repetitions × 2 models = <b>60</b> sequential HTTP "
        "generations. Settings: num_ctx=2048, num_predict=128, temperature=0, seed=42, "
        "warm-up, models unloaded between variants. Hardware: Windows 11, GTX 1650 4 GB, "
        "Ollama 0.40.16. Throughput is aggregate decode tokens / decode seconds from Ollama "
        "eval counters. GPU memory is Ollama <i>size_vram</i> after warm-up, not a "
        "full-request peak trace.",
        "small",
    ))
    flow.append(Spacer(1, 4))
    flow.append(DualHBars(CONTENT_W, 102))
    flow.append(Spacer(1, 5))
    flow.append(LatencyRow(CONTENT_W, 34))
    flow.append(P(
        "FP16 decode throughput was about 23% higher here; Q4 reduced reported allocation by "
        "about 56.6% (1003.94 → 435.22 MiB). Median latency differed by only ~46 ms. "
        "Throughput and latency need not move together: wall time includes HTTP, prefill, and "
        "variable output length. Truncations (done_reason=length): FP16 6, Q4 3. Ranking is "
        "specific to this GTX 1650 / Ollama build, not a universal FP16-versus-Q4 law. "
        "Do not read size_vram as full-system peak VRAM.",
        "caption",
    ))
    return flow


def page4():
    flow = []
    flow.append(P("TNDRR AI/ML ENGINEERING ASSIGNMENT", "kicker"))
    flow.append(P("Deployment design, findings, and conclusions", "title"))
    flow.append(P(
        "Quality scores are provisional. The 20-user / 24 GB design is a proposal. "
        "Neither is a measured multi-user result.",
        "subtitle",
    ))

    flow.append(heading("Proposed serving architecture  ·  one 24 GB GPU", 2))
    flow.append(ArchDiagram(CONTENT_W, 114))
    flow.append(P(
        "docs/DEPLOYMENT_DESIGN.md sketches a FastAPI gateway, request validation, an inference "
        "scheduler, continuous batching, and an Ollama or vLLM backend, with observability on "
        "queue depth and p95 latency. Memory budgeting treats 20 <b>actively decoding sequences</b> "
        "(not idle TCP connections) at up to 2,048 tokens, using official Qwen2.5-0.5B config "
        "(24 layers, GQA with 2 KV heads). Q4 weights would free VRAM for KV cache relative to "
        "FP16. <b>No FastAPI service, vLLM deployment, or 20-user load test exists in this "
        "repository.</b> Sequential GTX 1650 tok/s must not be extrapolated to 24 GB multi-user capacity.",
        "small",
    ))

    flow.append(heading("Quality findings  ·  provisional", 7))
    flow.append(P(
        "Ten prompts were scored 0–2 (provisional totals in quality_scored.csv): FP16 "
        "<b>7/20</b>, Q4 <b>8/20</b>. These are not finalized accuracy conclusions. "
        "Observed failure types: (1) FP16 P02 arithmetic truncated at 128 tokens before the "
        "answer 30 req/s, while Q4 finished correctly — a completion-budget failure, not proof "
        "FP16 cannot divide; (2) P09 GPU-memory arithmetic — FP16 truncated before 16 GB, Q4 "
        "used the wrong formula; a separate 256-token diagnostic (not in the 60-run summary) "
        "let FP16 finish P02 and P09 correctly while Q4 still produced −16 GB; (3) both models "
        "answered “Yes” on an invalid syllogism (P03); (4) both wrapped JSON in Markdown fences; "
        "(5) both failed alphabetical sorting (P08). Q4 saved reported allocation; it did not "
        "dominate quality or decode speed on this card.",
        "small",
    ))

    flow.append(heading("Key takeaways", 7))
    gap = 7
    tw = (CONTENT_W - 2 * gap) / 3
    takes = Table(
        [[
            takeaway_card(
                "TASK 1",
                "Parameter-efficient tuning improved classification accuracy significantly "
                "(28% → 84% on 100 articles), with a WORLD-recall trade-off (21/25 → 17/25).",
                tw,
            ),
            takeaway_card(
                "TASK 2",
                "Page-based retrieval metrics can overstate evidence quality. Hit@5 tied at "
                "0.933 while quote Hit@5 rose 0.600 → 0.800; chunk boundaries materially affected Q03.",
                tw,
            ),
            takeaway_card(
                "TASK 3",
                "Quantization reduced reported memory allocation (1003.94 → 435.22 MiB), but "
                "FP16 delivered higher decoding throughput on the tested GTX 1650 (84.88 vs 69.00 tok/s).",
                tw,
            ),
        ]],
        colWidths=[tw + gap, tw + gap, tw],
    )
    takes.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (0, 0), gap),
        ("RIGHTPADDING", (1, 0), (1, 0), gap),
        ("RIGHTPADDING", (2, 0), (2, 0), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    flow.append(takes)

    flow.append(heading("Limitations and untested assumptions", 8))
    flow.append(P(
        "Small evaluation sets; a single LoRA recipe; page-level RAG proxy metrics; sequential "
        "4 GB inference; provisional quality labels; untested 24 GB concurrency. Unmeasured: "
        "concurrent batching efficiency; actual KV-cache bytes per sequence on a 24 GB card; "
        "p95 latency under 20 users; OOM headroom; kernel-level Q4 vs FP16 speed on datacenter "
        "GPUs; whether raising num_predict in production would change the truncation pattern "
        "seen at 128 tokens. The wall-clock versus internal-timer gap on the 1650 is unexplained.",
        "small",
    ))

    flow.append(heading("Reproducibility", 6))
    flow.append(P(
        "Python 3.13. <b>Task 1:</b> executed notebook "
        "<font face='Consolas' size='7.5'>Task-1-FineTuning with LoRA/notebooks/LoRA_FineTuning.ipynb</font> "
        "plus results/*.json and adapters/lora_adapter/; reload requires the Hugging Face base model. "
        "<b>Task 2:</b> <font face='Consolas' size='7.5'>python run.py ingest</font> then "
        "<font face='Consolas' size='7.5'>python run.py evaluate</font> in Task-2-RAG-From-Scratch/; "
        "gold labels in evaluation/gold_questions.csv; Ollama qwen-q4 only for generation. "
        "<b>Task 3:</b> download GGUF files (gitignored), ollama create from the Modelfiles, "
        "<font face='Consolas' size='7.5'>python benchmark.py</font>; saved measurements in results/. "
        "A GitHub remote URL is omitted because it is not confirmed in this workspace. "
        "This report is generated from those stored artifacts only.",
        "small",
    ))
    return flow


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=TOP_MARGIN,
        bottomMargin=BOTTOM_MARGIN,
        title="TNDRR AI/ML Engineering Assignment — Combined Report",
        author="TNDRR internship submission",
    )
    story = []
    story.extend(page1())
    story.append(PageBreak())
    story.extend(page2())
    story.append(PageBreak())
    story.extend(page3())
    story.append(PageBreak())
    story.extend(page4())
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
