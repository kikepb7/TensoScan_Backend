from __future__ import annotations

import pytest

from tensoscan_ml import __version__
from tensoscan_ml.cli import COMMANDS, build_parser, main


def test_parser_exposes_every_subcommand_of_the_spec() -> None:
    expected = {"synth", "ingest", "split", "prelabel", "train", "eval", "export", "verify", "release"}

    actual = set(COMMANDS)

    assert expected <= actual


def test_help_for_each_subcommand_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    for name in COMMANDS:
        with pytest.raises(SystemExit) as exit_info:
            parser.parse_args([name, "--help"])

        assert exit_info.value.code == 0
    assert "usage" in capsys.readouterr().out


def test_version_flag_prints_the_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--version"])

    assert __version__ in capsys.readouterr().out


def test_release_is_not_part_of_this_delivery_and_says_so(capsys: pytest.CaptureFixture[str]) -> None:
    actual = main(["release"])

    assert actual == 2
    assert "not implemented" in capsys.readouterr().err
