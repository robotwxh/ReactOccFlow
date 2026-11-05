"""Build the one-page Chinese project overview from its editable Markdown.

Reference searches found no reportlab/PDF layout implementation in OccFM or
IR-WM. This independent document utility uses standard ReportLab components.
See docs/provenance/project_overview_a4_20261101.json.
"""
from __future__ import annotations

from html import escape
from pathlib import Path
import re

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Flowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/pdf"
SOURCE = OUT / "ReactOccFlow_项目概览_A4.md"
PDF = OUT / "ReactOccFlow_项目概览_A4.pdf"
INK = colors.HexColor("#16314B")
ACCENT = colors.HexColor("#147D92")
MUTED = colors.HexColor("#5B6B78")
RULE = colors.HexColor("#D9E3E9")


def markup(text: str, *, bold=False) -> str:
    """Use embedded CJK and Latin fonts for every glyph, with no font substitution."""
    cjk = pdfmetrics.getFont("CJK").face.charToGlyph
    latin_name = "LatinBold" if bold else "Latin"
    latin = pdfmetrics.getFont(latin_name).face.charToGlyph
    runs = []
    for char in text:
        code = ord(char)
        font = "CJK" if code > 127 and code in cjk else latin_name
        if code not in (cjk if font == "CJK" else latin):
            raise ValueError(f"Missing font glyph: {char!r} U+{code:04X}")
        if runs and runs[-1][0] == font:
            runs[-1][1] += char
        else:
            runs.append([font, char])
    return "".join(f'<font name="{font}">{escape(value)}</font>' for font, value in runs)


class Pipeline(Flowable):
    def __init__(self, width):
        super().__init__()
        self.width, self.height = width, 61

    def draw(self):
        gap = 12
        box_width = (self.width - 3 * gap) / 4
        labels = [
            ("历史场景", "研究阶段：oracle"),
            ("候选自车动作", "固定库 / 学习版待补"),
            ("联合未来生成", "DiT + actor（已实现）"),
            ("风险决策与控制", "接口已建 / 闭环待验"),
        ]
        title = ParagraphStyle("box-title", fontName="CJK", fontSize=9.4,
                               leading=12, alignment=TA_CENTER, textColor=INK)
        sub = ParagraphStyle("box-sub", fontName="CJK", fontSize=7.4,
                             leading=10, alignment=TA_CENTER, textColor=MUTED)
        for index, (name, detail) in enumerate(labels):
            x = index * (box_width + gap)
            self.canv.setFillColor(colors.HexColor("#EDF6F7") if index == 2 else colors.HexColor("#F4F7F9"))
            self.canv.setStrokeColor(ACCENT if index == 2 else RULE)
            self.canv.roundRect(x, 5, box_width, 48, 5, fill=1, stroke=1)
            for text, style, y in ((name, title, 31), (detail, sub, 15)):
                p = Paragraph(markup(text), style)
                _, h = p.wrap(box_width - 8, 30)
                p.drawOn(self.canv, x + 4, y - max(0, h - style.leading))
            if index < 3:
                start, end = x + box_width + 2, x + box_width + gap - 2
                self.canv.setStrokeColor(ACCENT)
                self.canv.setFillColor(ACCENT)
                self.canv.line(start, 29, end - 2, 29)
                arrow = self.canv.beginPath()
                arrow.moveTo(end, 29)
                arrow.lineTo(end - 4, 31.2)
                arrow.lineTo(end - 4, 26.8)
                arrow.close()
                self.canv.drawPath(arrow, fill=1, stroke=0)


def footer(canvas, document):
    width, _ = A4
    canvas.setStrokeColor(RULE)
    canvas.line(36, 30, width - 36, 30)
    style = ParagraphStyle("footer", fontName="CJK", fontSize=7.3,
                           leading=9, textColor=MUTED)
    p = Paragraph(markup("2026-11-01  |  项目总体方案与实现边界摘要  |  已实现能力与待验证目标分开记录"), style)
    p.wrap(width - 90, 15)
    p.drawOn(canvas, 36, 16)
    canvas.setFont("Latin", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(width - 36, 16, f"{document.page} / 1")


def main():
    pdfmetrics.registerFont(TTFont("CJK", "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"))
    pdfmetrics.registerFont(TTFont("Latin", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    pdfmetrics.registerFont(TTFont("LatinBold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))
    text = SOURCE.read_text()
    prefix, *sections = re.split(r"^## ", text, flags=re.MULTILINE)
    lines = [line for line in prefix.splitlines() if line.strip()]
    title, subtitle, lead = lines[0].removeprefix("# "), lines[1], lines[2].removeprefix("> ")
    width = A4[0] - 72
    body = ParagraphStyle("body", fontName="CJK", fontSize=10.1, leading=14.5,
                          wordWrap="CJK", textColor=INK, spaceAfter=4)
    section_style = ParagraphStyle("section", fontName="CJK", fontSize=11,
                                   leading=14.5, textColor=INK, wordWrap="CJK")
    story = [Paragraph(markup("PROJECT OVERVIEW", bold=True),
                       ParagraphStyle("kicker", fontName="LatinBold", fontSize=8,
                                      leading=11, textColor=ACCENT, spaceAfter=4)),
             Paragraph(markup(title, bold=True),
                       ParagraphStyle("title", fontName="LatinBold", fontSize=27,
                                      leading=32, textColor=INK, spaceAfter=5)),
             Paragraph(markup(subtitle),
                       ParagraphStyle("subtitle", fontName="CJK", fontSize=13.2,
                                      leading=18, textColor=MUTED, spaceAfter=11))]
    lead_style = ParagraphStyle("lead", parent=body, fontSize=10.6, leading=15.2, spaceAfter=0)
    banner = Table([[Paragraph(markup(lead), lead_style)]], colWidths=[width])
    banner.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EDF6F7")),
                               ("LEFTPADDING", (0, 0), (-1, -1), 10),
                               ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                               ("TOPPADDING", (0, 0), (-1, -1), 8),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    story.extend([banner, Spacer(1, 5), Pipeline(width)])
    for section in sections:
        heading, content = section.split("\n", 1)
        number, label = heading.split(" ", 1)
        story.append(Spacer(1, 7))
        h = Table([[Paragraph(markup(number, bold=True),
                              ParagraphStyle("number", fontName="LatinBold", fontSize=9.3,
                                             leading=14.5, textColor=ACCENT)),
                    Paragraph(markup(label), section_style)]], colWidths=[23, width - 23])
        h.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0),
                               ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                               ("TOPPADDING", (0, 0), (-1, -1), 0),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
        story.append(h)
        story.extend(Paragraph(markup(p.replace("\n", " ").strip()), body)
                     for p in re.split(r"\n\s*\n", content.strip()) if p.strip())
    doc = SimpleDocTemplate(str(PDF), pagesize=A4, leftMargin=36, rightMargin=36,
                            topMargin=29, bottomMargin=41, title="ReactOccFlow 项目整体思路",
                            author="ReactOccFlow", subject="One-page project overview")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    reader = PdfReader(PDF)
    if len(reader.pages) != 1:
        raise RuntimeError(f"Expected one A4 page, produced {len(reader.pages)} pages")
    extracted = "".join(reader.pages[0].extract_text().split())
    for marker in ("ReactOccFlow", "SwinVAE", "DiT", "22,000", "风险", "当前进展"):
        if marker not in extracted:
            raise RuntimeError(f"Missing content in PDF: {marker}")
    print(f"Created {PDF}: one A4 page, {len(extracted)} extracted characters")


if __name__ == "__main__":
    main()
