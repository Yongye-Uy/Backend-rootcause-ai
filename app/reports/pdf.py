import io
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.reports.data import ReportContext

ACCENT = colors.HexColor("#5b3fd9")
MUTED = colors.HexColor("#666666")
BORDER = colors.HexColor("#dddddd")

_stylesheet = getSampleStyleSheet()

TITLE = ParagraphStyle(
    "ReportTitle", parent=_stylesheet["Title"], textColor=ACCENT, fontSize=20, spaceAfter=4
)
META = ParagraphStyle(
    "ReportMeta", parent=_stylesheet["Normal"], textColor=MUTED, fontSize=9, spaceAfter=16
)
SECTION = ParagraphStyle(
    "SectionHeading",
    parent=_stylesheet["Heading2"],
    textColor=ACCENT,
    fontSize=13,
    spaceBefore=16,
    spaceAfter=6,
)
SUBHEADING = ParagraphStyle(
    "SubHeading",
    parent=_stylesheet["Heading3"],
    fontSize=11,
    spaceBefore=8,
    spaceAfter=4,
)
BODY = ParagraphStyle(
    "Body", parent=_stylesheet["Normal"], fontSize=10.5, leading=15, spaceAfter=6, alignment=TA_LEFT
)
QUESTION = ParagraphStyle("Question", parent=BODY, fontName="Helvetica-Bold", spaceAfter=1)
ANSWER = ParagraphStyle("Answer", parent=BODY, textColor=MUTED, spaceAfter=8)
FOOTER = ParagraphStyle("Footer", parent=_stylesheet["Normal"], fontSize=8, textColor=MUTED)

# ReportLab's built-in Helvetica only covers WinAnsiEncoding -- "smart" Unicode
# punctuation that LLMs love to use (non-breaking hyphens, curly quotes, em-dashes)
# renders as a black box glyph otherwise. Normalize to plain ASCII before escaping
# for the mini-XML markup Paragraph expects.
_PUNCT_MAP = {
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "--",
    "‘": "'", "’": "'", "‚": ",", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "…": "...",
    " ": " ",
}


def _clean(text: str) -> str:
    if not text:
        return ""
    for bad, good in _PUNCT_MAP.items():
        text = text.replace(bad, good)
    return escape(text)


def _bullets(items: list[str], style: ParagraphStyle = BODY) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(_clean(item), style)) for item in items],
        bulletType="bullet",
        leftIndent=16,
    )


def _sources(sources) -> list:
    if not sources:
        return [Paragraph("None given.", ANSWER)]
    return [
        Paragraph(f'<link href="{escape(s.url)}" color="#5b3fd9">{_clean(s.title)}</link>', BODY)
        for s in sources
    ]


def generate_pdf_report(context: ReportContext) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title=f"RootCause AI Investigation Report - Session {context.session_id}",
    )

    story: list = []

    story.append(Paragraph("RootCause AI &mdash; Investigation Report", TITLE))
    story.append(
        Paragraph(
            f"Session #{context.session_id} &middot; Generated {context.generated_at.strftime('%B %d, %Y')}",
            META,
        )
    )

    story.append(Paragraph("Original Problem", SECTION))
    story.append(Paragraph(_clean(context.problem_text), BODY))

    story.append(Paragraph("Investigation Questions &amp; Answers", SECTION))
    for qa_round in context.qa_rounds:
        story.append(Paragraph(f"Round {qa_round.round}", SUBHEADING))
        for question, answer in qa_round.pairs:
            story.append(Paragraph(_clean(question), QUESTION))
            story.append(Paragraph(_clean(answer), ANSWER))

    story.append(Paragraph("Confirmed Root Cause", SECTION))
    root_cause_text = _clean(context.root_cause)
    if context.root_cause_rejection_count > 0:
        root_cause_text += (
            f"  <i>(confirmed after {context.root_cause_rejection_count} revision"
            f"{'s' if context.root_cause_rejection_count != 1 else ''})</i>"
        )
    story.append(Paragraph(root_cause_text, BODY))

    story.append(Paragraph("Solutions Considered", SECTION))
    table_data = [["#", "Name", "Cost", "Difficulty", "Time"]]
    for s in context.solutions_summary:
        table_data.append(
            [str(s.rank), _clean(s.name), _clean(s.cost), _clean(s.difficulty), _clean(s.time_estimate)]
        )
    solutions_table = Table(table_data, colWidths=[0.3 * inch, 2.1 * inch, 1.4 * inch, 1.2 * inch, 1.2 * inch])
    solutions_table.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f5ff")]),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(solutions_table)

    sel = context.selected_solution
    story.append(Paragraph(f"Selected Solution: {_clean(sel.name)}", SECTION))
    story.append(Paragraph(_clean(sel.explanation), BODY))
    story.append(Paragraph(f"<b>Resources:</b> {_clean(sel.resources)}", BODY))
    story.append(
        Paragraph(
            f"<b>Cost:</b> {_clean(sel.cost)} &nbsp;&nbsp; <b>Difficulty:</b> {_clean(sel.difficulty)} "
            f"&nbsp;&nbsp; <b>Time:</b> {_clean(sel.time_estimate)}",
            BODY,
        )
    )
    if sel.pros:
        story.append(Paragraph("Advantages", SUBHEADING))
        story.append(_bullets(sel.pros))
    if sel.cons:
        story.append(Paragraph("Disadvantages", SUBHEADING))
        story.append(_bullets(sel.cons))
    if sel.risks:
        story.append(Paragraph("Risks", SUBHEADING))
        story.append(_bullets(sel.risks))
    story.append(Paragraph("Sources", SUBHEADING))
    story.extend(_sources(sel.sources))

    plan = context.plan
    story.append(Paragraph("Implementation Plan", SECTION))
    story.append(Paragraph(_clean(plan.overview), BODY))
    story.append(Paragraph("Requirements", SUBHEADING))
    story.append(Paragraph(_clean(plan.requirements), BODY))

    meta_table = Table(
        [
            ["Tools", "Cost", "Timeline"],
            [_clean(plan.tools), _clean(plan.cost), _clean(plan.timeline)],
        ],
        colWidths=[2.1 * inch, 2.1 * inch, 2.1 * inch],
    )
    meta_table.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("TEXTCOLOR", (0, 0), (-1, 0), ACCENT),
                ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(Spacer(1, 6))
    story.append(meta_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Step-by-Step Instructions", SUBHEADING))
    story.append(
        ListFlowable(
            [ListItem(Paragraph(_clean(step), BODY)) for step in plan.steps],
            bulletType="1",
            leftIndent=16,
        )
    )

    story.append(Paragraph("Possible Problems", SUBHEADING))
    story.append(Paragraph(_clean(plan.possible_problems), BODY))
    story.append(Paragraph("Alternatives", SUBHEADING))
    story.append(Paragraph(_clean(plan.alternatives), BODY))
    story.append(Paragraph("Sources", SUBHEADING))
    story.extend(_sources(plan.sources))

    story.append(Spacer(1, 20))
    story.append(
        Paragraph(
            f"Generated locally from your saved investigation on "
            f"{context.generated_at.strftime('%B %d, %Y at %H:%M UTC')} &mdash; no additional AI calls were made.",
            FOOTER,
        )
    )

    doc.build(story)
    return buffer.getvalue()
