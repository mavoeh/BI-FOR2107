from __future__ import annotations

import json

from for2107_repro.cli import main
from for2107_repro.progress import ProgressCounter, configure, report, stage


def test_progress_writes_to_stderr_only(capsys) -> None:
    configure(enabled=True)
    report("starting")
    counter = ProgressCounter("items", 2)
    counter.advance()
    counter.advance()
    with stage("example stage"):
        pass
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "starting" in captured.err
    assert "items: 2/2 (100%)" in captured.err
    assert "START example stage" in captured.err
    assert "DONE  example stage" in captured.err
    configure(enabled=False)


def test_large_counter_reports_first_completion(capsys) -> None:
    configure(enabled=True)
    counter = ProgressCounter("participants", 278)
    counter.advance()
    captured = capsys.readouterr()
    assert "participants: 1/278" in captured.err
    configure(enabled=False)


def test_cli_progress_does_not_corrupt_json(capsys) -> None:
    result = main(["preflight", "--config", "configs/smoke.yaml"])
    captured = capsys.readouterr()
    assert result == 0
    assert isinstance(json.loads(captured.out), dict)
    assert "START Preflight checks" in captured.err
    assert "Command complete: preflight" in captured.err


def test_cli_quiet_suppresses_progress(capsys) -> None:
    result = main(["preflight", "--config", "configs/smoke.yaml", "--quiet"])
    captured = capsys.readouterr()
    assert result == 0
    assert isinstance(json.loads(captured.out), dict)
    assert captured.err == ""
