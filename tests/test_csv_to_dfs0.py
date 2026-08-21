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


def _item_row(**overrides):
    row = {column: None for column in csv_to_dfs0.REQUIRED_ITEM_COLUMNS}
    row.update(
        job_id="job",
        order=1,
        enabled=True,
        csv_column="Q",
        item_name="Q",
        eum_type="Discharge",
        eum_unit=None,
        data_value_type="Instantaneous",
        scale_factor=1,
        offset=0,
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


def test_config_helpers_normalize_spreadsheet_values(tmp_path) -> None:
    config_path = tmp_path / "config.xlsx"

    assert csv_to_dfs0._truthy(" YES ")
    assert not csv_to_dfs0._truthy("off")
    assert csv_to_dfs0._blank_to_none(float("nan")) is None
    assert csv_to_dfs0._pipe_split(" one | two || ") == ["one", "two"]
    assert csv_to_dfs0._normalize_data_value_type("mean-step-forward") == (
        "MeanStepForward"
    )
    assert csv_to_dfs0._path_from_config(
        "results/output",
        str(tmp_path),
        ".dfs0",
    ) == str(tmp_path / "results" / "output.dfs0")
    assert csv_to_dfs0._config_dir(str(config_path)) == str(tmp_path)


def test_load_config_reports_missing_required_columns(monkeypatch) -> None:
    def fake_read_excel(path, sheet_name):
        if sheet_name == "Jobs":
            return pd.DataFrame({"job_id": ["job"]})
        return pd.DataFrame(columns=csv_to_dfs0.REQUIRED_ITEM_COLUMNS)

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    with pytest.raises(ValueError, match="Missing required columns in Jobs"):
        csv_to_dfs0.load_config("config.xlsx")


def test_iter_jobs_resolves_relative_paths_and_defaults(tmp_path) -> None:
    jobs = pd.DataFrame(
        [
            _job_row(
                csv_path="input/source.csv",
                output_dfs0_path=None,
                delimiter=";",
                decimal=",",
                skiprows="2",
                header_row="1",
                na_values="NA | missing",
                dayfirst="yes",
                drop_rows_all_nan=None,
            ),
            _job_row(job_id="disabled", enabled=False),
        ]
    )

    result = list(csv_to_dfs0.iter_jobs(jobs, str(tmp_path / "config.xlsx")))

    assert len(result) == 1
    job = result[0]
    assert Path(job.csv_path) == tmp_path / "input" / "source.csv"
    assert Path(job.output_path) == tmp_path / "input" / "source.dfs0"
    assert job.delimiter == ";"
    assert job.decimal == ","
    assert job.skiprows == 2
    assert job.header_row == 1
    assert job.na_values == ["NA", "missing"]
    assert job.dayfirst
    assert job.drop_rows_all_nan


def test_build_dataset_sorts_transforms_and_drops_empty_rows(
    tmp_path,
    monkeypatch,
) -> None:
    csv_path = tmp_path / "input.csv"
    csv_path.write_text(
        "time,Q\n"
        "2024-01-02,2\n"
        "2024-01-03,\n"
        "2024-01-01,1\n",
        encoding="utf-8",
    )
    job = csv_to_dfs0.Job(
        "job",
        str(csv_path),
        str(tmp_path / "output.dfs0"),
        "time",
    )
    items = pd.DataFrame([_item_row(scale_factor=10, offset=1)])
    captured = {}

    class FakeMikeio:
        @staticmethod
        def from_pandas(data, items):
            captured["data"] = data.copy()
            captured["items"] = items
            return "dataset"

    monkeypatch.setattr(csv_to_dfs0, "_import_mikeio", lambda: FakeMikeio())
    monkeypatch.setattr(
        csv_to_dfs0,
        "_make_item_info_factory",
        lambda: lambda **kwargs: kwargs,
    )

    dataset, data = csv_to_dfs0.build_dataset_for_job(job, items)

    assert dataset == "dataset"
    assert data.index.tolist() == pd.to_datetime(
        ["2024-01-01", "2024-01-02"]
    ).tolist()
    assert data["Q"].tolist() == [11.0, 21.0]
    assert captured["data"].equals(data)
    assert captured["items"][0]["item_name"] == "Q"


@pytest.mark.parametrize(
    ("csv_text", "message"),
    [
        (
            "time,Q\nnot-a-time,1\n2024-01-02,2\n",
            "unparseable timestamps",
        ),
        (
            "time,Q\n2024-01-01,1\n2024-01-01,2\n",
            "duplicate timestamps",
        ),
    ],
)
def test_build_dataset_rejects_invalid_time_axis(
    tmp_path,
    monkeypatch,
    csv_text: str,
    message: str,
) -> None:
    csv_path = tmp_path / "input.csv"
    csv_path.write_text(csv_text, encoding="utf-8")
    job = csv_to_dfs0.Job(
        "job",
        str(csv_path),
        str(tmp_path / "output.dfs0"),
        "time",
    )
    monkeypatch.setattr(csv_to_dfs0, "_import_mikeio", lambda: object())
    monkeypatch.setattr(
        csv_to_dfs0,
        "_make_item_info_factory",
        lambda: lambda **kwargs: kwargs,
    )

    with pytest.raises(ValueError, match=message):
        csv_to_dfs0.build_dataset_for_job(job, pd.DataFrame([_item_row()]))
