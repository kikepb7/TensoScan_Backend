"""The legacy HTML report: one plain table, no JavaScript, no external resources.

The previous page loaded Plotly from a CDN and interpolated data into a script (AUD-B-21). Every value is now escaped
with ``html.escape``. The page disappears with the legacy routes (spec B-002, REQ-B002-13).
"""

from collections.abc import Sequence
from html import escape
from zoneinfo import ZoneInfo

from app.domain.measurements.model import Measurement

_STYLE = (
    "body{font-family:Arial,sans-serif;margin:20px}table{border-collapse:collapse;width:100%}"
    "th,td{border:1px solid #ddd;padding:8px;text-align:center}th{background:#4caf50;color:#fff}"
)


def render_measurements_html(measurements: Sequence[Measurement], timezone: ZoneInfo) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{escape(str(m.reading.systolic))}</td>"
        f"<td>{escape(str(m.reading.diastolic))}</td>"
        f"<td>{escape(str(m.reading.pulse))}</td>"
        f"<td>{escape(m.measured_at.astimezone(timezone).strftime('%d-%m-%Y %H:%M'))}</td>"
        "</tr>"
        for m in measurements
    )
    return (
        '<!doctype html><html lang="es"><head><meta charset="utf-8">'
        f"<title>Informe de mediciones</title><style>{_STYLE}</style></head><body>"
        "<h1>Informe de mediciones</h1>"
        '<p><a href="/ocr/user/measurements/pdf">Descargar PDF</a></p>'
        "<table><thead><tr><th>Sistólica</th><th>Diastólica</th><th>Pulso</th><th>Fecha</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></body></html>"
    )
