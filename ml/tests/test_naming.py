"""T-B003-06: file-name convention and labels.csv (REQ-B003-09/10)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tensoscan_ml.data.naming import NamingError, Roi, parse_labels_csv, parse_name


def test_a_reading_name_is_parsed_into_device_values_condition_and_take() -> None:
    actual = parse_name("omron-m3__128-82-67__dim__01.mp4")

    assert (actual.device, actual.sys, actual.dia, actual.pul) == ("omron-m3", 128, 82, 67)
    assert (actual.status, actual.condition, actual.take) == ("COMPLETE", "dim", "01")
    assert actual.reading_key == "128-82-67"


def test_an_error_name_has_no_values_and_a_device_error_status() -> None:
    actual = parse_name("omron-m3__error__good__01.mp4")

    assert actual.status == "DEVICE_ERROR"
    assert (actual.sys, actual.dia, actual.pul) == (None, None, None)
    assert actual.reading_key == "error"


def test_photos_follow_the_same_convention() -> None:
    assert parse_name("beurer-bm27__141-94-78__glare__03.JPG").device == "beurer-bm27"


@pytest.mark.parametrize(
    ("name", "code"),
    [
        ("omron-m3_128-82-67_dim_01.mp4", "BAD_NAME"),
        ("omron-m3__128-82__dim__01.mp4", "BAD_NAME"),
        ("omron-m3__128-82-67__dark__01.mp4", "BAD_CONDITION"),
        ("Omron M3__128-82-67__dim__01.mp4", "BAD_DEVICE"),
        ("omron-m3__82-128-67__dim__01.mp4", "IMPLAUSIBLE_READING"),
        ("omron-m3__128-82-500__dim__01.mp4", "IMPLAUSIBLE_READING"),
        ("omron-m3__128-82-67__dim__01.txt", "BAD_EXTENSION"),
    ],
)
def test_names_that_break_the_convention_are_rejected_with_a_code(name: str, code: str) -> None:
    with pytest.raises(NamingError) as error:
        parse_name(name)

    assert error.value.code == code


def _write_csv(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_labels_csv_rows_are_validated_one_by_one(tmp_path: Path) -> None:
    csv_path = _write_csv(
        tmp_path / "labels.csv",
        "file,device,sys,dia,pul,condition\n"
        "IMG_001.jpg,omron-m3,128,82,67,good\n"
        "IMG_002.jpg,omron-m3,error,,,dim\n"
        "IMG_003.jpg,omron-m3,128,,67,good\n"
        "IMG_004.jpg,omron-m3,128,82,67,night\n"
        "IMG_001.jpg,omron-m3,128,82,67,good\n"
        "IMG_005.jpg,omron-m3,70,82,67,good\n",
    )

    rows, problems = parse_labels_csv(csv_path)

    assert [row.file for row in rows] == ["IMG_001.jpg", "IMG_002.jpg"]
    assert rows[1].label.status == "DEVICE_ERROR"
    assert [(p.line, p.code) for p in problems] == [
        (4, "MISSING_VALUE"),
        (5, "BAD_CONDITION"),
        (6, "DUPLICATE_ROW"),
        (7, "IMPLAUSIBLE_READING"),
    ]


def test_labels_csv_accepts_manual_roi_columns(tmp_path: Path) -> None:
    csv_path = _write_csv(
        tmp_path / "labels.csv",
        "file,device,sys,dia,pul,condition,roi_x,roi_y,roi_w,roi_h\n"
        "a.jpg,omron-m3,128,82,67,good,100,200,640,800\n"
        "b.jpg,omron-m3,128,82,67,good,,,,\n"
        "c.jpg,omron-m3,128,82,67,good,10,20,,\n",
    )

    rows, problems = parse_labels_csv(csv_path)

    assert rows[0].roi == Roi(100, 200, 640, 800)
    assert rows[1].roi is None
    assert [p.code for p in problems] == ["BAD_ROI"]


def test_labels_csv_without_the_required_columns_is_an_error(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "labels.csv", "file,sys,dia\na.jpg,128,82\n")

    with pytest.raises(ValueError, match="missing columns"):
        parse_labels_csv(csv_path)
