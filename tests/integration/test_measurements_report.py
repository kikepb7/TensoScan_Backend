"""``GET /v1/measurements/report``: the PDF is parsed back with pypdf to check what a person would read."""

import re
from io import BytesIO
from typing import Any

import httpx
import pytest
from pypdf import PdfReader

from app.infrastructure.reports.reportlab_renderer import DISCLAIMER
from tests.conftest import NEW_USER, bearer, register_and_login
from tests.support.measurements import body, post

REPORT = "/v1/measurements/report"


def pdf_text(response: httpx.Response) -> str:
    reader = PdfReader(BytesIO(response.content))
    return "\n".join(page.extract_text() for page in reader.pages)


def one_line(text: str) -> str:
    return " ".join(text.split())


@pytest.fixture
async def tokens(client: httpx.AsyncClient) -> dict[str, Any]:
    return await register_and_login(client)


async def test_the_report_is_a_pdf_attachment_named_after_the_generation_date(
    client: httpx.AsyncClient, tokens: dict[str, Any]
) -> None:
    await post(client, tokens, body())

    response = await client.get(REPORT, headers=bearer(tokens), params={"format": "pdf"})

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"] == 'attachment; filename="tensoscan-report-20261019.pdf"'
    assert response.content.startswith(b"%PDF")


async def test_the_format_parameter_is_optional_and_defaults_to_pdf(
    client: httpx.AsyncClient, tokens: dict[str, Any]
) -> None:
    await post(client, tokens, body())

    response = await client.get(REPORT, headers=bearer(tokens))

    assert response.headers["content-type"] == "application/pdf"


async def test_times_are_printed_in_the_default_zone_europe_madrid(
    client: httpx.AsyncClient, tokens: dict[str, Any]
) -> None:
    await post(client, tokens, body(measured_at="2026-10-19T06:15:00Z"))  # CEST: UTC+2

    text = pdf_text(await client.get(REPORT, headers=bearer(tokens)))

    assert "19/10/2026 08:15" in text
    assert "19/10/2026 06:15" not in text
    assert "(Europe/Madrid)" in text


@pytest.mark.parametrize(
    ("tz", "expected"),
    [
        ("Europe/Madrid", "19/10/2026 08:15"),
        ("UTC", "19/10/2026 06:15"),
        ("America/New_York", "19/10/2026 02:15"),
        ("Asia/Tokyo", "19/10/2026 15:15"),
    ],
)
async def test_the_tz_parameter_decides_the_local_times(
    client: httpx.AsyncClient, tokens: dict[str, Any], tz: str, expected: str
) -> None:
    await post(client, tokens, body(measured_at="2026-10-19T06:15:00Z"))

    text = pdf_text(await client.get(REPORT, headers=bearer(tokens), params={"tz": tz}))

    assert expected in text
    assert f"({tz})" in text


async def test_winter_and_summer_times_use_their_own_offsets(client: httpx.AsyncClient, tokens: dict[str, Any]) -> None:
    await post(client, tokens, body(measured_at="2026-01-15T12:00:00Z"))  # CET: UTC+1
    await post(client, tokens, body(measured_at="2026-07-15T12:00:00Z"))  # CEST: UTC+2

    text = pdf_text(await client.get(REPORT, headers=bearer(tokens)))

    assert "15/01/2026 13:00" in text
    assert "15/07/2026 14:00" in text


async def test_the_report_contains_header_summary_chart_table_and_disclaimer(
    client: httpx.AsyncClient, tokens: dict[str, Any]
) -> None:
    for index, (systolic, diastolic, pulse) in enumerate([(120, 80, 60), (140, 90, 70), (130, 85, 80)]):
        await post(
            client,
            tokens,
            body(systolic=systolic, diastolic=diastolic, pulse=pulse, measured_at=f"2026-10-1{index}T08:00:00Z"),
        )

    response = await client.get(REPORT, headers=bearer(tokens))
    text = pdf_text(response)

    flat = one_line(text)
    assert "Informe de mediciones de tensión arterial" in flat
    assert "Paciente: Ana García" in flat
    assert "Periodo: 10/10/2026 - 12/10/2026" in flat
    assert "Generado: 19/10/2026 10:00 (Europe/Madrid)" in flat
    assert "Número de lecturas: 3" in flat
    assert "130.0" in text and "85.0" in text and "70.0" in text  # means
    assert {"Sistólica", "Diastólica", "Pulso", "Evolución", "Mediciones"} <= set(text.split("\n"))
    assert "10/10/2026 10:00" in text and "12/10/2026 10:00" in text
    assert one_line(DISCLAIMER) in flat
    axis_labels = [line for line in text.split("\n") if re.fullmatch(r"\d\d/\d\d( \d\d:\d\d)?", line)]
    assert len(axis_labels) >= 5, "the evolution chart (with its date axis) is missing from the PDF"


async def test_every_page_carries_the_disclaimer(client: httpx.AsyncClient, tokens: dict[str, Any]) -> None:
    for index in range(80):
        await post(client, tokens, body(measured_at=f"2026-09-{1 + index // 4:02d}T{(index % 4) * 5 + 1:02d}:00:00Z"))

    response = await client.get(REPORT, headers=bearer(tokens))

    pages = PdfReader(BytesIO(response.content)).pages
    assert len(pages) >= 2
    assert all(one_line(DISCLAIMER) in one_line(page.extract_text()) for page in pages)


async def test_a_single_measurement_still_produces_a_report(client: httpx.AsyncClient, tokens: dict[str, Any]) -> None:
    await post(client, tokens, body())

    response = await client.get(REPORT, headers=bearer(tokens))

    assert response.status_code == 200
    assert "Número de lecturas: 1" in one_line(pdf_text(response))


async def test_from_and_to_limit_the_measurements_in_the_report(
    client: httpx.AsyncClient, tokens: dict[str, Any]
) -> None:
    for day in (10, 11, 12):
        await post(client, tokens, body(measured_at=f"2026-10-{day}T10:00:00Z"))

    response = await client.get(
        REPORT, headers=bearer(tokens), params={"from": "2026-10-11T00:00:00Z", "to": "2026-10-11T23:59:59Z"}
    )
    text = pdf_text(response)

    assert "11/10/2026 12:00" in text
    assert "10/10/2026" not in text
    assert "12/10/2026 12:00" not in text
    assert "Número de lecturas: 1" in one_line(text)


async def test_a_users_report_never_includes_other_users_measurements(
    client: httpx.AsyncClient, tokens: dict[str, Any]
) -> None:
    other = await register_and_login(client, {**NEW_USER, "email": "other@example.com", "name": "Berta"})
    await post(client, tokens, body(systolic=121, measured_at="2026-10-10T10:00:00Z"))
    await post(client, other, body(systolic=199, measured_at="2026-10-11T10:00:00Z"))

    text = pdf_text(await client.get(REPORT, headers=bearer(tokens)))

    assert "Número de lecturas: 1" in one_line(text)
    assert "199" not in text
    assert "Berta" not in text


async def test_text_taken_from_the_data_is_escaped_so_it_cannot_inject_markup(client: httpx.AsyncClient) -> None:
    hostile = {**NEW_USER, "name": "<b>Ana</b> & <i>Co", "last_name": '"><img src=x> &amp; <para'}
    tokens = await register_and_login(client, hostile)
    await post(client, tokens, body())

    response = await client.get(REPORT, headers=bearer(tokens))

    assert response.status_code == 200
    flat = one_line(pdf_text(response))
    assert 'Paciente: <b>Ana</b> & <i>Co "><img src=x> &amp; <para' in flat


@pytest.mark.parametrize("tz", ["Mars/Olympus", "not a zone", "../../etc/passwd", "Europe", "", "UTC+2"])
async def test_an_invalid_tz_is_a_422_on_the_tz_field(
    client: httpx.AsyncClient, tokens: dict[str, Any], tz: str
) -> None:
    await post(client, tokens, body())

    response = await client.get(REPORT, headers=bearer(tokens), params={"tz": tz})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert response.json()["error"]["details"] == [{"field": "tz", "code": "INVALID_TIMEZONE"}]


async def test_the_html_format_does_not_exist_in_v1(client: httpx.AsyncClient, tokens: dict[str, Any]) -> None:
    await post(client, tokens, body())

    response = await client.get(REPORT, headers=bearer(tokens), params={"format": "html"})

    assert response.status_code == 422
    assert response.json()["error"]["details"][0]["field"] == "format"


async def test_no_measurements_in_the_range_is_a_404(client: httpx.AsyncClient, tokens: dict[str, Any]) -> None:
    empty = await client.get(REPORT, headers=bearer(tokens))
    await post(client, tokens, body(measured_at="2026-10-10T10:00:00Z"))

    outside = await client.get(REPORT, headers=bearer(tokens), params={"from": "2026-11-01T00:00:00Z"})

    assert empty.status_code == outside.status_code == 404
    assert outside.json()["error"]["code"] == "NOT_FOUND"


async def test_the_report_requires_authentication(client: httpx.AsyncClient) -> None:
    response = await client.get(REPORT)

    assert response.status_code == 401
