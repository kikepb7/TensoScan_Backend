"""PDF report with ReportLab (ADR-B002-02): header, summary, evolution chart, table and a disclaimer footer.

Every piece of text that comes from data (the user's name, dates, numbers) goes through ``xml.sax.saxutils.escape``
before it reaches a ``Paragraph``, whose mini-markup would otherwise interpret ``<`` and ``&`` (AUD-B-21).
"""

import math
from datetime import datetime, timedelta, tzinfo
from io import BytesIO
from typing import Any
from xml.sax.saxutils import escape

from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.graphics.widgets.markers import makeMarker
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.domain.measurements.model import Measurement
from app.domain.measurements.report import FieldStats, MeasurementsReport

DISCLAIMER = (
    "Este informe es orientativo y no constituye un diagnóstico médico. "
    "Consulta siempre con un profesional sanitario para interpretar tus mediciones."
)

_SYSTOLIC = colors.HexColor("#B91C1C")
_DIASTOLIC = colors.HexColor("#1D4ED8")
_PULSE = colors.HexColor("#B45309")
_HEADER_BACKGROUND = colors.HexColor("#1F2937")
_ZEBRA = colors.HexColor("#F3F4F6")
_GRID = colors.HexColor("#D1D5DB")
_MARKERS_UP_TO = 60  # beyond this many points the markers only add clutter
_X_TICKS = 6


class ReportLabReportRenderer:
    """Implements the ``ReportRenderer`` port."""

    def render(self, report: MeasurementsReport) -> bytes:
        buffer = BytesIO()
        document = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            leftMargin=18 * mm,
            rightMargin=18 * mm,
            topMargin=16 * mm,
            bottomMargin=24 * mm,
            title="Informe de mediciones de tensión arterial",
            author="TensoScan",
        )
        styles = _styles()
        story: list[Any] = [
            *_header(report, styles),
            Spacer(1, 6 * mm),
            *_summary(report, styles),
            Spacer(1, 6 * mm),
            Paragraph("Evolución", styles["heading"]),
            _chart(report, document.width),
            Spacer(1, 6 * mm),
            Paragraph("Mediciones", styles["heading"]),
            _table(report),
        ]
        document.build(story, onFirstPage=_footer, onLaterPages=_footer)
        return buffer.getvalue()


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontSize=18, alignment=0, spaceAfter=6),
        "heading": ParagraphStyle("heading", parent=base["Heading2"], fontSize=12, spaceBefore=2, spaceAfter=4),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontSize=10, leading=14),
    }


def _local(value: datetime, zone: tzinfo) -> datetime:
    return value.astimezone(zone)


def _date_time(value: datetime, zone: tzinfo) -> str:
    return _local(value, zone).strftime("%d/%m/%Y %H:%M")


def _header(report: MeasurementsReport, styles: dict[str, ParagraphStyle]) -> list[Any]:
    zone = report.timezone
    first, last = report.measurements[0].measured_at, report.measurements[-1].measured_at
    start, end = report.range_from or first, report.range_to or last
    period = f"{_local(start, zone):%d/%m/%Y} - {_local(end, zone):%d/%m/%Y}"
    lines = [
        ("Paciente", report.user_name),
        ("Periodo", period),
        ("Generado", f"{_date_time(report.generated_at, zone)} ({zone})"),
    ]
    return [
        Paragraph("Informe de mediciones de tensión arterial", styles["title"]),
        *[Paragraph(f"<b>{label}:</b> {escape(value)}", styles["body"]) for label, value in lines],
    ]


def _summary(report: MeasurementsReport, styles: dict[str, ParagraphStyle]) -> list[Any]:
    summary = report.summary

    def row(label: str, stats: FieldStats) -> list[str]:
        return [label, f"{stats.mean:.1f}", str(stats.minimum), str(stats.maximum)]

    table = Table(
        [
            ["", "Media", "Mínimo", "Máximo"],
            row("Sistólica (mmHg)", summary.systolic),
            row("Diastólica (mmHg)", summary.diastolic),
            row("Pulso (ppm)", summary.pulse),
        ],
        colWidths=[48 * mm, 28 * mm, 28 * mm, 28 * mm],
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("LINEBELOW", (0, 0), (-1, 0), 0.8, _GRID),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _ZEBRA]),
            ]
        )
    )
    return [
        Paragraph("Resumen", styles["heading"]),
        Paragraph(f"Número de lecturas: <b>{summary.count}</b>", styles["body"]),
        Spacer(1, 2 * mm),
        table,
    ]


def _table(report: MeasurementsReport) -> Table:
    header = ["Fecha y hora", "Sistólica (mmHg)", "Diastólica (mmHg)", "Pulso (ppm)"]
    rows = [
        [
            _date_time(m.measured_at, report.timezone),
            str(m.reading.systolic),
            str(m.reading.diastolic),
            str(m.reading.pulse),
        ]
        for m in report.measurements
    ]
    table = Table([header, *rows], colWidths=[50 * mm, 40 * mm, 40 * mm, 30 * mm], repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), _HEADER_BACKGROUND),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _ZEBRA]),
                ("LINEBELOW", (0, 0), (-1, -1), 0.25, _GRID),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _chart(report: MeasurementsReport, available_width: float) -> Drawing:
    """Line chart of the three series against time (x = days since the first measurement)."""
    measurements = report.measurements
    origin = measurements[0].measured_at

    def days(m: Measurement) -> float:
        return (m.measured_at - origin) / timedelta(days=1)

    series = [
        ("Sistólica", _SYSTOLIC, [(days(m), m.reading.systolic) for m in measurements]),
        ("Diastólica", _DIASTOLIC, [(days(m), m.reading.diastolic) for m in measurements]),
        ("Pulso", _PULSE, [(days(m), m.reading.pulse) for m in measurements]),
    ]
    height = 62 * mm
    drawing = Drawing(available_width, height)
    plot = LinePlot()
    plot.x, plot.y = 38, 28
    plot.width, plot.height = available_width - 50, height - 28 - 22
    plot.data = [points for _, _, points in series]
    for index, (_, color, _) in enumerate(series):
        plot.lines[index].strokeColor = color
        plot.lines[index].strokeWidth = 1.4
        if len(measurements) <= _MARKERS_UP_TO:
            plot.lines[index].symbol = makeMarker("Circle", fillColor=color, strokeColor=color, size=3.5)

    span = days(measurements[-1])
    x_min, x_max = (0.0, span) if span > 0 else (-0.5, 0.5)
    plot.xValueAxis.valueMin, plot.xValueAxis.valueMax = x_min, x_max
    plot.xValueAxis.valueSteps = [x_min + (x_max - x_min) * i / _X_TICKS for i in range(_X_TICKS + 1)]
    show_time = (x_max - x_min) < 3
    pattern = "%d/%m %H:%M" if show_time else "%d/%m"
    plot.xValueAxis.labelTextFormat = lambda value: _local(origin + timedelta(days=value), report.timezone).strftime(
        pattern
    )
    plot.xValueAxis.labels.fontSize = 7
    values = [y for _, _, points in series for _, y in points]
    plot.yValueAxis.valueMin = max(0, math.floor((min(values) - 10) / 10) * 10)
    plot.yValueAxis.valueMax = math.ceil((max(values) + 10) / 10) * 10
    plot.yValueAxis.labels.fontSize = 7
    plot.yValueAxis.visibleGrid = True
    plot.yValueAxis.gridStrokeColor = _GRID
    plot.yValueAxis.gridStrokeWidth = 0.4
    drawing.add(plot)

    left = plot.x
    for label, color, _ in series:
        drawing.add(Line(left, height - 8, left + 14, height - 8, strokeColor=color, strokeWidth=1.8))
        drawing.add(Rect(left + 5, height - 10, 4, 4, fillColor=color, strokeColor=color))
        drawing.add(String(left + 18, height - 11, label, fontName="Helvetica", fontSize=8))
        left += 78
    unit = String(available_width - 2, height - 11, "mmHg / ppm", fontName="Helvetica", fontSize=7)
    unit.textAnchor, unit.fillColor = "end", colors.grey
    drawing.add(unit)
    return drawing


def _footer(canvas: Any, document: Any) -> None:
    canvas.saveState()
    style = ParagraphStyle("footer", fontName="Helvetica-Oblique", fontSize=8, leading=10, textColor=colors.grey)
    disclaimer = Paragraph(escape(DISCLAIMER), style)
    disclaimer.wrap(document.width, document.bottomMargin)
    disclaimer.drawOn(canvas, document.leftMargin, 13 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawRightString(A4[0] - document.rightMargin, 8 * mm, f"TensoScan · Página {document.page}")
    canvas.restoreState()
