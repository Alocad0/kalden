from pathlib import Path

import pandas as pd
import pytest

from kalden.core.mike import csv_to_dfs0


def _job_row(**overrides):
    row = {column: None for column in csv_to_dfs0.REQUIRED_JOB_COLUMNS}
    row.update(
        job_id="job",
        enabled=True,
        csv_path="input.csv",
        time_column="time",
    )
    row.update(overrides)
    return row


def test_enabled_job_requires_non_blank_id() -> None:
    jobs = pd.DataFrame([_job_row(job_id=float("nan"))])

    with pytest.raises(ValueError, match="missing job_id"):
        list(csv_to_dfs0.iter_jobs(jobs, "config.xlsx"))


def test_duplicate_enabled_job_ids_are_rejected() -> None:
    jobs = pd.DataFrame([_job_row(), _job_row()])

    with pytest.raises(ValueError, match="Duplicate enabled job_id"):
        list(csv_to_dfs0.iter_jobs(jobs, "config.xlsx"))


def test_malformed_numeric_csv_value_is_rejected(tmp_path, monkeypatch) -> None:
    csv_path = tmp_path / "input.csv"
    csv_path.write_text(
        "time,Q,R\n2024-01-01,not-a-number,1\n2024-01-02,2,3\n",
        encoding="utf-8",
    )
    job = csv_to_dfs0.Job(
        "job",
        str(csv_path),
        str(tmp_path / "output.dfs0"),
        "time",
    )
    items = pd.DataFrame(
        [
            {
                "job_id": "job",
                "order": 1,
                "enabled": True,
                "csv_column": "Q",
                "item_name": "Q",
                "eum_type": "Discharge",
                "eum_unit": None,
                "data_value_type": "Instantaneous",
                "scale_factor": 1,
                "offset": 0,
            },
            {
                "job_id": "job",
                "order": 2,
                "enabled": True,
                "csv_column": "R",
                "item_name": "R",
                "eum_type": "Discharge",
                "eum_unit": None,
                "data_value_type": "Instantaneous",
                "scale_factor": 1,
                "offset": 0,
            },
        ]
    )

    monkeypatch.setattr(csv_to_dfs0, "_import_mikeio", lambda: object())
    monkeypatch.setattr(
        csv_to_dfs0,
        "_make_item_info_factory",
        lambda: lambda **kwargs: kwargs,
    )

    with pytest.raises(ValueError, match="non-numeric value"):
        csv_to_dfs0.build_dataset_for_job(job, items)


def test_write_job_does_not_overwrite_without_permission(tmp_path, monkeypatch) -> None:
    output = tmp_path / "output.dfs0"
    output.write_bytes(b"original")
    job = csv_to_dfs0.Job("job", "input.csv", str(output), "time")

    class Dataset:
        def to_dfs(self, path):
            Path(path).write_bytes(b"replacement")

    monkeypatch.setattr(
        csv_to_dfs0,
        "build_dataset_for_job",
        lambda *_: (Dataset(), pd.DataFrame({"value": [1]})),
    )

    with pytest.raises(FileExistsError):
        csv_to_dfs0.write_job(job, pd.DataFrame())
    assert output.read_bytes() == b"original"


def test_failed_write_preserves_existing_output(tmp_path, monkeypatch) -> None:
    output = tmp_path / "output.dfs0"
    output.write_bytes(b"original")
    job = csv_to_dfs0.Job("job", "input.csv", str(output), "time")

    class Dataset:
        def to_dfs(self, path):
            Path(path).write_bytes(b"partial")
            raise OSError("simulated write failure")

    monkeypatch.setattr(
        csv_to_dfs0,
        "build_dataset_for_job",
        lambda *_: (Dataset(), pd.DataFrame({"value": [1]})),
    )

    with pytest.raises(OSError, match="simulated write failure"):
        csv_to_dfs0.write_job(job, pd.DataFrame(), overwrite=True)

    assert output.read_bytes() == b"original"
    assert list(tmp_path.glob(".*.dfs0")) == []
